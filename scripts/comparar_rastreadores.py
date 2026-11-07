"""
Compara os rastreadores de pessoas que vem com o Ultralytics em trechos dos
videos, com as MESMAS deteccoes (modelo de pose YOLO26n): a diferenca entre
eles e so como ligam as deteccoes de um frame ao outro.

Sem gabarito de identidade (ninguem anotou "quem e quem" frame a frame), as
medidas sao indiretas, mas dizem o que importa para o app:
  ids_confirmados   trilhas que duraram >= 0,5 s (as que viram "Pessoa N")
  ids_por_pessoa    ids_confirmados / pessoas reais do trecho (contadas a
                    olho); 1,0 = cada pessoa manteve um so ID. Acima disso
                    = a pessoa "trocou de numero" (fragmentacao)
  duracao_media_s   duracao media das trilhas confirmadas
  trilhas_curtas    trilhas < 0,5 s (ruido: deteccao relampago)
  ms_rastreador     custo extra por frame em relacao a so detectar (medido com
                    uma segunda copia do modelo, sem rastreador acoplado - no
                    mesmo objeto, o predict() tambem rodaria o rastreador)

Saida: results/rastreadores/comparacao.csv

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/comparar_rastreadores.py
"""

import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics.trackers.basetrack import BaseTrack  # noqa: E402

from epi.detector import carregar_yolo, eh_openvino  # noqa: E402

SAIDA = "results/rastreadores"
POSE_OV = "models/openvino/pessoa_384x640_openvino_model"
STEMAC = "video/Chefe Secreto - Stemac Parte 2 - Alan Mendes (480p, h264, youtube).mp4"
ALUMINIO = ("video/YTDown.com_YouTube_Media_EirrzjjAf8Y_DE-ONDE-VEM-O-ALUMÍNIO-Boravê-com-Mari-Fulfaro-"
            "Manual-do-Mundo_001_1080p.mp4")
# planos continuos (sem corte de cena) achados por scripts/mapear_videos.py, com as
# pessoas reais contadas a olho em 6 miniaturas de cada plano:
# (video, inicio_s, duracao_s, pessoas reais, observacao)
TRECHOS = [
    (ALUMINIO, 194, 33, 1, "apresentador, camera fecha no rosto no fim"),
    (ALUMINIO, 391, 31, 1, "apresentador com protetor facial"),
    (STEMAC, 13, 20, 2, "palco; empilhadeira passa na frente (oclusao)"),
    (ALUMINIO, 254, 21, 1, "operario pequeno andando na sala de cubas"),
    (STEMAC, 321, 10, 2, "dois funcionarios no estoque"),
    (STEMAC, 909, 9, 2, "escritorio, duas pessoas sentadas"),
    (STEMAC, 946, 8, 2, "duas pessoas em pe, creditos na tela"),
    (STEMAC, 0, 12, 2, "palco; camera passa atras de pilares (oclusao)"),
]
RASTREADORES = ["bytetrack.yaml", "botsort.yaml", "tracktrack.yaml", "epi/tracktrack_sem_gmc.yaml", "ocsort.yaml"]


def rodar(modelo, modelo_det, disp, video, inicio, dur, rastreador):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio * fps))
    pred = getattr(modelo, "predictor", None)
    if pred is not None and hasattr(pred, "trackers"):
        del pred.trackers  # forca criar o rastreador novo (persist=True reaproveitaria o anterior)
    BaseTrack._count = 0
    vistos, ts_track, ts_det = {}, [], []
    for i in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        r = modelo.track(frame, persist=True, tracker=rastreador, imgsz=[384, 640], classes=[0], conf=0.1,
                         device=disp, verbose=False)[0]
        ts_track.append((time.perf_counter() - t) * 1000)
        if i % 10 == 0:  # custo so da deteccao, para separar o do rastreador
            t = time.perf_counter()
            modelo_det.predict(frame, imgsz=[384, 640], classes=[0], conf=0.1, device=disp, verbose=False)
            ts_det.append((time.perf_counter() - t) * 1000)
        if r.boxes is not None and r.boxes.id is not None:
            for tid, s in zip(r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                if s >= 0.15:
                    vistos.setdefault(tid, []).append(i)
    dur_trilhas = np.array([(max(f) - min(f) + 1) / fps for f in vistos.values()]) if vistos else np.zeros(0)
    return dur_trilhas, float(np.median(ts_track) - np.median(ts_det))


def main():
    os.makedirs(SAIDA, exist_ok=True)
    caminho = POSE_OV if eh_openvino(POSE_OV) else "yolo26n-pose.pt"
    modelo, modelo_det = carregar_yolo(caminho), carregar_yolo(caminho)
    disp = "intel:gpu" if eh_openvino(POSE_OV) else "cpu"
    linhas, total = [], {}
    for video, inicio, dur, reais, obs in TRECHOS:
        for rast in RASTREADORES:
            d, ms = rodar(modelo, modelo_det, disp, video, inicio, dur, rast)
            conf_ = d[d >= 0.5]
            nome_r = os.path.basename(rast).replace(".yaml", "")
            linhas.append([os.path.basename(video)[:30], inicio, dur, reais, obs, nome_r, len(conf_),
                           f"{len(conf_) / reais:.2f}", f"{conf_.mean() if len(conf_) else 0:.1f}", int((d < 0.5).sum()),
                           f"{ms:.1f}"])
            t = total.setdefault(nome_r, [0, 0, 0, []])
            t[0] += len(conf_)
            t[1] += reais
            t[2] += int((d < 0.5).sum())
            t[3].append(ms)
            print(f"  {os.path.basename(video)[:24]:24s} {inicio:4d}s  {nome_r:20s} ids={len(conf_):3d} "
                  f"({len(conf_) / reais:.2f}/pessoa)  duracao media {conf_.mean() if len(conf_) else 0:5.1f}s  "
                  f"curtas={int((d < 0.5).sum()):3d}  +{ms:.1f} ms", flush=True)
    with open(os.path.join(SAIDA, "comparacao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "inicio_s", "duracao_s", "pessoas_reais", "observacao", "rastreador", "ids_confirmados",
                    "ids_por_pessoa", "duracao_media_s", "trilhas_curtas", "ms_rastreador"])
        w.writerows(linhas)
    with open(os.path.join(SAIDA, "resumo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rastreador", "ids_confirmados", "pessoas_reais", "ids_por_pessoa", "trilhas_curtas", "ms_rastreador_mediana"])
        for rast, (ids, reais, curtas, mss) in total.items():
            w.writerow([rast, ids, reais, f"{ids / reais:.2f}", curtas, f"{np.median(mss):.1f}"])
            print(f"TOTAL {rast:12s} {ids} IDs para {reais} pessoas = {ids / reais:.2f} por pessoa | curtas {curtas} | "
                  f"+{np.median(mss):.1f} ms")
    print(f"resultados em {SAIDA}/comparacao.csv e resumo.csv")


if __name__ == "__main__":
    main()
