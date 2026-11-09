"""
Mede a velocidade do sistema no notebook (i5-1135G7, sem GPU dedicada),
com frames reais dos videos da pasta video/. Tres partes:

A) Latencia dos modelos por formato/dispositivo (ms por frame, ponta a
   ponta: pre-processamento + rede + pos-processamento):
   PyTorch na CPU x OpenVINO na CPU x OpenVINO na GPU integrada (Iris Xe).

B) Loop do app antigo (app/streamlit_app.py, sem o Streamlit): a cada
   volta faz seek para o frame "do relogio", roda o modelo (PyTorch, 960),
   desenha as caixas e codifica a imagem (o st.image manda PNG ao
   navegador). O video so mostra os frames que o loop conseguiu processar:
   por isso ficava travado ("camera lenta" com saltos).

C) Pipeline novo (epi/detector.py como o app/servidor.py usa: pose +
   rastreamento + EPIs em paralelo na GPU), lendo o video em sequencia.
   No app novo o video e tocado pelo navegador a velocidade normal; esta
   parte mede quantos frames por segundo a analise acompanha.

Saidas em results/benchmark_velocidade/:
  latencia_modelos.csv, loop_app_antigo.csv, pipeline_novo.csv

Uso (a partir da pasta TCC; feche o app antes para nao disputar a GPU):
    venv_win\\Scripts\\python scripts/benchmark_velocidade.py
    venv_win\\Scripts\\python scripts/benchmark_velocidade.py --partes A --frames 60
"""

import argparse
import csv
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from epi.detector import Detector, carregar_yolo  # noqa: E402

SAIDA = "results/benchmark_velocidade"
OV = "models/openvino"
INICIO_S = 100  # pula a abertura dos videos


def videos():
    return sorted(p for p in glob.glob("video/*") if p.lower().endswith((".mp4", ".webm", ".mov")))


def amostra_frames(caminho, n, passo=4):
    cap = cv2.VideoCapture(caminho)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(INICIO_S * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    frames = []
    while len(frames) < n:
        for _ in range(passo - 1):
            cap.grab()
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def cronometrar(fn, frames, aquecer=5):
    for f in frames[:aquecer]:
        fn(f)
    ts = []
    for f in frames:
        t = time.perf_counter()
        fn(f)
        ts.append((time.perf_counter() - t) * 1000)
    return np.median(ts), np.percentile(ts, 90)


def parte_a(frames, saida):
    casos = [
        ("EPI v6 (imgsz 960)", "PyTorch", "CPU", "models/best_v6.pt", "cpu", 960),
        ("EPI v6 (imgsz 960)", "OpenVINO", "CPU", f"{OV}/v6_frame_544x960_openvino_model", "intel:cpu", [544, 960]),
        ("EPI v6 (imgsz 960)", "OpenVINO", "GPU Iris Xe", f"{OV}/v6_frame_544x960_openvino_model", "intel:gpu", [544, 960]),
        ("Pessoas+pose YOLO26n (imgsz 640)", "PyTorch", "CPU", "yolo26n-pose.pt", "cpu", 640),
        ("Pessoas+pose YOLO26n (imgsz 640)", "OpenVINO", "CPU", f"{OV}/pessoa_384x640_openvino_model", "intel:cpu", [384, 640]),
        ("Pessoas+pose YOLO26n (imgsz 640)", "OpenVINO", "GPU Iris Xe", f"{OV}/pessoa_384x640_openvino_model", "intel:gpu", [384, 640]),
    ]
    linhas = []
    for modelo, formato, disp, caminho, device, imgsz in casos:
        if not os.path.exists(caminho):
            print(f"  (pulado, falta {caminho} - rode scripts/exportar_openvino.py)")
            continue
        m = carregar_yolo(caminho)
        med, p90 = cronometrar(lambda f: m.predict(f, imgsz=imgsz, device=device, verbose=False), frames)
        linhas.append([modelo, formato, disp, f"{med:.1f}", f"{p90:.1f}", f"{1000 / med:.1f}"])
        print(f"  {modelo:34s} {formato:8s} {disp:12s} {med:6.1f} ms (p90 {p90:6.1f})  {1000 / med:5.1f} FPS")
    gravar(saida, "latencia_modelos.csv", ["modelo", "formato", "dispositivo", "ms_mediana", "ms_p90", "fps"], linhas)


def parte_b(segundos, saida):
    """Replica o run_live_loop do app antigo (seek por frame + predict + plot + PNG)."""
    m = YOLO("models/best_v6.pt")
    linhas = []
    for v in videos():
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        base = int(INICIO_S * fps)
        m.predict(np.zeros((480, 854, 3), np.uint8), imgsz=960, verbose=False)
        t0 = time.time()
        n, t_seek, t_pred, t_desenho, frames_mostrados = 0, [], [], [], []
        while time.time() - t0 < segundos:
            alvo = base + int((time.time() - t0) * fps)
            a = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, alvo)
            ok, frame = cap.read()
            b = time.perf_counter()
            r = m.predict(frame, conf=0.40, imgsz=960, verbose=False)
            c = time.perf_counter()
            rgb = cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB)
            cv2.imencode(".png", rgb)
            d = time.perf_counter()
            t_seek.append((b - a) * 1000)
            t_pred.append((c - b) * 1000)
            t_desenho.append((d - c) * 1000)
            frames_mostrados.append(alvo)
            n += 1
        dur = time.time() - t0
        exibidos_fps = n / dur
        linhas.append([os.path.basename(v)[:40], f"{fps:.2f}", f"{exibidos_fps:.1f}", f"{100 * exibidos_fps / fps:.1f}",
                       f"{np.median(t_seek):.1f}", f"{np.median(t_pred):.1f}", f"{np.median(t_desenho):.1f}",
                       f"{np.median(np.diff(frames_mostrados)):.0f}"])
        print(f"  {os.path.basename(v)[:40]:40s} video {fps:.1f} fps -> mostra {exibidos_fps:4.1f} fps "
              f"({100 * exibidos_fps / fps:.0f}% dos frames) | seek+leitura {np.median(t_seek):.0f} ms, "
              f"modelo {np.median(t_pred):.0f} ms, desenho+PNG {np.median(t_desenho):.0f} ms")
    gravar(saida, "loop_app_antigo.csv", ["video", "fps_video", "fps_exibido", "pct_frames_exibidos", "ms_seek_leitura",
                                         "ms_modelo", "ms_desenho_png", "salto_mediano_frames"], linhas)


def medir_pipeline(det, video, n_frames):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(INICIO_S * fps))
    det.resetar_rastreamento()
    ts, ts_leitura = [], []
    for i in range(n_frames):
        a = time.perf_counter()
        ok, frame = cap.read()  # leitura sequencial, sem seek
        b = time.perf_counter()
        if not ok:
            break
        det.processar(frame, rastrear=True)
        c = time.perf_counter()
        if i >= 5:
            ts.append((c - a) * 1000)
            ts_leitura.append((b - a) * 1000)
    return fps, np.median(ts), np.median(ts_leitura)


def parte_c(n_frames, saida, repeticoes=3):
    """Cada config roda `repeticoes` vezes em rodizio (a ordem gira a cada rodada), com pausa entre
    medicoes: o notebook esquenta e reduz o clock (CPU e GPU dividem o mesmo limite de energia),
    entao medir uma config inteira e depois a outra favoreceria a primeira."""
    configs = [
        ("OpenVINO GPU, paralelo (app)", dict(modelo_epi=f"{OV}/v6_frame_544x960_openvino_model",
                                              modelo_pessoa=f"{OV}/pessoa_384x640_openvino_model", paralelo=True)),
        ("OpenVINO GPU, sequencial", dict(modelo_epi=f"{OV}/v6_frame_544x960_openvino_model",
                                          modelo_pessoa=f"{OV}/pessoa_384x640_openvino_model")),
        ("PyTorch CPU, sequencial", dict(modelo_epi="models/best_v6.pt", modelo_pessoa="yolo26n-pose.pt")),
    ]
    dets = {nome: Detector(conf=0.15, iou=0.6, intel="gpu", **kw) for nome, kw in configs}
    medidas = {}
    for rodada in range(repeticoes):
        ordem = configs[rodada % len(configs):] + configs[:rodada % len(configs)]
        for nome, _ in ordem:
            for v in videos():
                time.sleep(5)  # pausa para esfriar entre medicoes
                fps, med, leit = medir_pipeline(dets[nome], v, n_frames)
                medidas.setdefault((nome, v), []).append((med, leit, fps))
                print(f"  rodada {rodada + 1}: {nome:30s} {os.path.basename(v)[:30]:30s} {med:5.1f} ms/frame", flush=True)
    linhas = []
    for (nome, v), ms in medidas.items():
        meds = [m[0] for m in ms]
        med, fps = float(np.median(meds)), ms[0][2]
        cap_fps = 1000 / med
        linhas.append([nome, os.path.basename(v)[:40], f"{fps:.2f}", f"{np.median([m[1] for m in ms]):.1f}", f"{med:.1f}",
                       f"{min(meds):.1f}", f"{max(meds):.1f}", f"{cap_fps:.1f}", f"{min(100, 100 * cap_fps / fps):.0f}"])
        print(f"  {nome:30s} {os.path.basename(v)[:30]:30s} {med:5.1f} ms/frame (min {min(meds):.1f}, max {max(meds):.1f})"
              f" -> {cap_fps:4.1f} FPS ({min(100, 100 * cap_fps / fps):.0f}% dos frames de um video de {fps:.0f} fps)")
    gravar(saida, "pipeline_novo.csv", ["config", "video", "fps_video", "ms_leitura", "ms_por_frame_mediana",
                                       "ms_min_rodadas", "ms_max_rodadas", "fps_analise", "pct_frames_analisaveis"], linhas)


def gravar(saida, nome, cab, linhas):
    with open(os.path.join(saida, nome), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partes", default="ABC")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--segundos", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)
    if "A" in args.partes:
        print("A) latencia dos modelos")
        frames = amostra_frames(videos()[0], args.frames)
        parte_a(frames, SAIDA)
    if "B" in args.partes:
        print("B) loop do app antigo (Streamlit)")
        parte_b(args.segundos, SAIDA)
    if "C" in args.partes:
        print("C) pipeline novo")
        parte_c(args.frames, SAIDA)
    print(f"resultados em {SAIDA}/")


if __name__ == "__main__":
    main()
