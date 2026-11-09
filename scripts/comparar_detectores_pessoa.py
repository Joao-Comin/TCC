"""
Compara detectores de pessoa (COCO) para o app: no teste real, que fracao
das caixas de EPI anotadas cai dentro de alguma pessoa detectada (mesma
regra de associacao do app, epi/detector.py), e o tempo na CPU.

EPI fora de qualquer pessoa nao entra na conformidade de ninguem, entao
essa fracao e o teto do que a associacao consegue. Os modelos de pose cobrem
um pouco menos, mas trazem os pontos do corpo usados na visibilidade.

Saida: results/detectores_pessoa/comparacao.csv

Uso (a partir da pasta TCC; baixa os pesos que faltarem):
    venv_win\\Scripts\\python scripts/comparar_detectores_pessoa.py
"""

import csv
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from epi.detector import associar  # noqa: E402
from epi.metricas import ler_rotulos  # noqa: E402

TESTE = "novos_frames_dataset/real_world_test"
SAIDA = "results/detectores_pessoa"
MODELOS = ["yolo11n.pt", "yolo11n-pose.pt", "yolo26n.pt", "yolo26n-pose.pt", "yolo11s-pose.pt"]


def main():
    os.makedirs(SAIDA, exist_ok=True)
    dados = []
    for p in sorted(glob.glob(os.path.join(TESTE, "images", "*"))):
        im = cv2.imread(p)
        h, w = im.shape[:2]
        dados.append((im, ler_rotulos(os.path.join(TESTE, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)))
    linhas = []
    for nome in MODELOS:
        m = YOLO(nome)
        for im, _ in dados[:2]:
            m.predict(im, imgsz=640, classes=[0], verbose=False)
        cobertos = total = n_pessoas = 0
        ts = []
        for im, (gc, gk) in dados:
            h, w = im.shape[:2]
            t = time.perf_counter()
            r = m.predict(im, imgsz=640, conf=0.3, classes=[0], verbose=False)[0]
            ts.append((time.perf_counter() - t) * 1000)
            pc = r.boxes.xyxy.cpu().numpy()
            n_pessoas += len(pc)
            cobertos += int((associar(gc, pc, w, h) >= 0).sum())
            total += len(gc)
        linhas.append([nome, n_pessoas, cobertos, total, f"{100 * cobertos / total:.1f}", f"{np.median(ts):.0f}"])
        print(f"  {nome:16s} pessoas={n_pessoas:3d}  EPIs dentro de pessoa {cobertos}/{total} = "
              f"{100 * cobertos / total:.1f}%  {np.median(ts):.0f} ms (CPU)")
    with open(os.path.join(SAIDA, "comparacao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "pessoas_detectadas", "epis_dentro_de_pessoa", "epis_anotados", "pct", "ms_cpu_mediana"])
        w.writerows(linhas)


if __name__ == "__main__":
    main()
