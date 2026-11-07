"""
Escolhe frames de um video para anotar a mao (aprendizado ativo) e ja
gera as pre-anotacoes, para no Roboflow so corrigir em vez de desenhar.

Por que nao pegar frames aleatorios: a maioria mostraria o que o modelo ja
acerta (capacete grande, colete). Aqui cada frame candidato (1 por segundo)
ganha uma nota de "informatividade", maior quando:
  - ha punho visivel (pose) sem luva detectada com conf >= 0.4 na pessoa;
  - ha tornozelo visivel sem bota detectada com conf >= 0.4 na pessoa;
  - ha deteccoes na faixa de duvida (0.15 a 0.45) - o modelo hesitou.
So entram frames com contexto de trabalho (alguem com capacete ou colete
detectado) e nota >= --nota-min: em entrevista de escritorio, reuniao
ou carro, mao sem luva e verdade, nao erro do modelo, e o frame pouco ensina.
So contam pessoas grandes o bastante para anotar (altura >= 20% do frame).
Depois escolhe os de maior nota, com pelo menos --espaco segundos entre si,
sem frames quase iguais (correlacao da miniatura > 0.85) e no maximo
--max-por-plano frames do mesmo plano continuo (videos editados repetem o
mesmo enquadramento por varios segundos; mais frames dele ensinam pouco).

Vazamento: se o video ja tem frames anotados (results/mapa_videos/
origem_frames_anotados.csv, gerado por scripts/mapear_videos.py), pula
--margem segundos em volta de cada um - senao um frame novo de treino
poderia ser vizinho de um frame do teste real (mesma cena).

Pre-anotacao: uniao do v5 e do v6 em 960 (NMS por classe), conf >= 0.25,
que errar por excesso e mais barato (apagar caixa e mais rapido que desenhar).

Saida em <saida>/<nome_do_video>/:
  images/*.jpg  labels/*.txt (formato YOLO, classes na ordem do projeto)
  classes.txt   selecionados.csv (nota de cada frame)   folha.jpg (miniaturas)

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/selecionar_frames_anotacao.py --video "video/<arquivo>.mp4" --n 120
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from epi import CLASSES  # noqa: E402
from epi.conformidade import partes_visiveis  # noqa: E402
from epi.detector import associar, carregar_yolo, extrair, juntar, nms_por_classe  # noqa: E402

LUVA, BOTA = CLASSES.index("gloves"), CLASSES.index("boots")


def segundos_ja_anotados(video):
    caminho = "results/mapa_videos/origem_frames_anotados.csv"
    if not os.path.exists(caminho):
        print("  aviso: rode scripts/mapear_videos.py antes, para evitar frames vizinhos dos ja anotados")
        return []
    with open(caminho, encoding="utf-8") as f:
        return [int(r["segundo"]) for r in csv.DictReader(f) if r["video"] == os.path.basename(video)]


def miniatura(frame):
    m = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def nota_frame(frame, m_pose, modelos_epi):
    h, w = frame.shape[:2]
    rp = m_pose.predict(frame, imgsz=640, conf=0.4, classes=[0], verbose=False)[0]
    pc, ps, _ = extrair(rp)
    kps = rp.keypoints.data.cpu().numpy() if rp.keypoints is not None and len(pc) else np.zeros((0, 17, 3))
    grandes = (pc[:, 3] - pc[:, 1]) >= 0.20 * h if len(pc) else np.zeros(0, bool)
    partes = [m.predict(frame, imgsz=960, conf=0.15, iou=0.6, verbose=False)[0] for m in modelos_epi]
    ec, es, ek = nms_por_classe(*juntar(*[extrair(r) for r in partes]), iou=0.5)
    dono = associar(ec, pc, w, h)
    nota = 0.0
    for j in np.flatnonzero(grandes):
        vis = partes_visiveis(kps[j], w, h)
        seus = dono == j
        tem = lambda c: bool(((ek == c) & seus & (es >= 0.4)).any())  # noqa: E731
        nota += (vis[LUVA] and not tem(LUVA)) + (vis[BOTA] and not tem(BOTA))
    nota += 0.5 * float(((es >= 0.15) & (es < 0.45)).sum())
    # contexto de trabalho: se alguem tem capacete ou colete, mao/pe sem luva/bota
    # provavelmente e erro do modelo; em entrevista de escritorio e negativo verdadeiro
    # (oculos nao conta: oculos de grau no escritorio viram "goggles" e enganam o filtro)
    contexto = bool((np.isin(ek, [CLASSES.index(c) for c in ("helmet", "vest")]) & (es >= 0.4)).any())
    return nota, contexto, int(grandes.sum()), (ec, es, ek)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--espaco", type=float, default=3.0, help="segundos minimos entre frames escolhidos")
    ap.add_argument("--inicio", type=float, default=0)
    ap.add_argument("--fim", type=float, default=None)
    ap.add_argument("--saida", default="novos_frames_v7")
    ap.add_argument("--conf-pre", type=float, default=0.25)
    ap.add_argument("--margem", type=float, default=20, help="segundos evitados em volta de frames ja anotados")
    ap.add_argument("--nota-min", type=float, default=1.0)
    ap.add_argument("--sem-contexto", action="store_true", help="aceita frames sem ninguem de EPI")
    ap.add_argument("--max-por-plano", type=int, default=3)
    args = ap.parse_args()
    evitar = segundos_ja_anotados(args.video)
    if evitar:
        print(f"  evitando +-{args.margem:.0f} s em volta de {len(evitar)} frames ja anotados deste video")

    m_pose = carregar_yolo("yolo26n-pose.pt")
    modelos_epi = [carregar_yolo("models/best.pt"), carregar_yolo("models/best_v6.pt")]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fim = int((args.fim or total / fps) * fps)
    candidatos = []
    i = int(args.inicio * fps)
    while i < min(total, fim):
        if any(abs(i / fps - s) <= args.margem for s in evitar):
            i += int(fps)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        nota, contexto, n_pessoas, dets = nota_frame(frame, m_pose, modelos_epi)
        if n_pessoas and nota >= args.nota_min and (contexto or args.sem_contexto):
            candidatos.append((nota, i, n_pessoas, dets, miniatura(frame)))
        if int(i / fps) % 120 == 0:
            print(f"  {i / fps:6.0f} s  candidatos: {len(candidatos)}", flush=True)
        i += int(fps)  # 1 candidato por segundo

    # plano continuo: candidatos vizinhos no tempo com miniaturas parecidas (correlacao >= 0.5)
    plano, p_atual = {}, 0
    for k, c in enumerate(candidatos):
        if k and float(c[4] @ candidatos[k - 1][4]) < 0.5:
            p_atual += 1
        plano[c[1]] = p_atual
    candidatos.sort(key=lambda c: -c[0])
    escolhidos, por_plano = [], {}
    for c in candidatos:
        if len(escolhidos) >= args.n:
            break
        if por_plano.get(plano[c[1]], 0) >= args.max_por_plano:
            continue
        if any(abs(c[1] - e[1]) < args.espaco * fps for e in escolhidos):
            continue
        if any(float(c[4] @ e[4]) > 0.85 for e in escolhidos):
            continue
        escolhidos.append(c)
        por_plano[plano[c[1]]] = por_plano.get(plano[c[1]], 0) + 1
    escolhidos.sort(key=lambda c: c[1])

    base = os.path.splitext(os.path.basename(args.video))[0][:40].strip().replace(" ", "_")
    pasta = os.path.join(args.saida, base)
    for d in ("images", "labels"):
        os.makedirs(os.path.join(pasta, d), exist_ok=True)
    with open(os.path.join(pasta, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    miniaturas = []
    with open(os.path.join(pasta, "selecionados.csv"), "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["arquivo", "frame", "segundo", "nota", "pessoas", "caixas_pre"])
        for nota, idx, n_pessoas, (ec, es, ek), _ in escolhidos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            h, w = frame.shape[:2]
            nome = f"{base}_f{idx:06d}"
            cv2.imwrite(os.path.join(pasta, "images", nome + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok_conf = es >= args.conf_pre
            with open(os.path.join(pasta, "labels", nome + ".txt"), "w") as fl:
                for (x1, y1, x2, y2), k in zip(ec[ok_conf], ek[ok_conf]):
                    fl.write(f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            wcsv.writerow([nome + ".jpg", idx, f"{idx / fps:.1f}", f"{nota:.1f}", n_pessoas, int(ok_conf.sum())])
            mini = cv2.resize(frame, (240, int(240 * h / w)))
            cv2.putText(mini, f"{idx / fps:.0f}s nota {nota:.1f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            miniaturas.append(mini)
    if miniaturas:
        hm = max(m.shape[0] for m in miniaturas)
        miniaturas = [cv2.copyMakeBorder(m, 0, hm - m.shape[0], 0, 0, cv2.BORDER_CONSTANT) for m in miniaturas]
        linhas = [np.hstack(miniaturas[k:k + 8] + [np.zeros_like(miniaturas[0])] * (8 - len(miniaturas[k:k + 8])))
                  for k in range(0, len(miniaturas), 8)]
        cv2.imwrite(os.path.join(pasta, "folha.jpg"), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(escolhidos)} frames (de {len(candidatos)} candidatos com contexto e nota >= {args.nota_min}) em {pasta}/")


if __name__ == "__main__":
    main()
