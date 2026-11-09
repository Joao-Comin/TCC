"""
Analise de erros por instancia anotada (GT) no teste real: para cada caixa
anotada, o que o modelo fez com ela (conf >= 0.25, como no app):

  acerto        caixa da mesma classe com IoU >= 0.5
  conf_baixa    so achou com conf < 0.25 (o limiar cortou)
  localizacao   achou a classe certa, mas com IoU entre 0.1 e 0.5
  outra_classe  caixa de OUTRA classe no lugar (IoU >= 0.5)
  perdido       nada no lugar

Alem da contagem, grava uma folha com o recorte de cada luva e bota (verde =
acerto, vermelho = erro) e o tamanho da caixa em pixels no frame original -
foi assim que se viu que muitas luvas perdidas NAO sao pequenas (luvas
escuras, de couro, coloridas, em close) e que as botas perdidas sao escuras,
borradas pelo movimento e parcialmente encobertas.

Saida em results/analise_erros/:
  categorias.csv          contagem por modelo x classe x categoria
  instancias.csv          uma linha por caixa anotada (tamanho, categoria)
  folha_luvas_botas_<modelo>.jpg

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/analisar_erros.py
"""

import argparse
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from epi import CLASSES, NOMES_PT  # noqa: E402
from epi.detector import Detector  # noqa: E402
from epi.metricas import iou_matriz, ler_rotulos  # noqa: E402

TESTE = "novos_frames_dataset/real_world_test"
SAIDA = "results/analise_erros"
CATEGORIAS = ["acerto", "conf_baixa", "localizacao", "outra_classe", "perdido"]
LIMIAR = 0.25


def categoria(j, gk, pc, ps, pk, iou):
    c = gk[j]
    mesma = pk == c
    alta = mesma & (ps >= LIMIAR)
    if alta.any() and iou[j][alta].max() >= 0.5:
        return "acerto"
    if mesma.any() and iou[j][mesma].max() >= 0.5:
        return "conf_baixa"
    if alta.any() and iou[j][alta].max() >= 0.1:
        return "localizacao"
    outra = (pk != c) & (ps >= LIMIAR)
    if outra.any() and iou[j][outra].max() >= 0.5:
        return "outra_classe"
    return "perdido"


def ladrilho(im, caixa, cat, classe):
    x1, y1, x2, y2 = caixa.astype(int)
    h, w = im.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    m = max(bw, bh)
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
    rec = im[Y1:Y2, X1:X2].copy()
    cv2.rectangle(rec, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (60, 200, 60) if cat == "acerto" else (40, 40, 230), 2)
    esc = 200 / max(rec.shape[:2])
    rec = cv2.resize(rec, (max(1, int(rec.shape[1] * esc)), max(1, int(rec.shape[0] * esc))))
    t = np.full((232, 200, 3), 24, np.uint8)
    t[:rec.shape[0], :rec.shape[1]] = rec
    txt = f"{NOMES_PT[classe]} - {cat.replace('_', ' ')} - {bw}x{bh}px"
    cv2.putText(t, txt, (4, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["v6=models/best_v6.pt", "v5=models/best.pt"])
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(TESTE, "images", "*")))
    linhas_cat, linhas_inst = [], []
    for spec in args.modelos:
        nome, caminho = spec.split("=")
        det = Detector(caminho, None, imgsz=args.imgsz, conf=0.05, iou=0.6)
        cont = {c: dict.fromkeys(CATEGORIAS, 0) for c in CLASSES}
        ladrilhos = []
        for p in imgs:
            im = cv2.imread(p)
            h, w = im.shape[:2]
            gc, gk = ler_rotulos(os.path.join(TESTE, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)
            pc, ps, pk = det.processar(im)["epis"]
            iou = iou_matriz(gc, pc) if len(gc) and len(pc) else np.zeros((len(gc), len(pc)))
            for j in range(len(gc)):
                cat = categoria(j, gk, pc, ps, pk, iou)
                classe = CLASSES[gk[j]]
                cont[classe][cat] += 1
                bw, bh = gc[j][2] - gc[j][0], gc[j][3] - gc[j][1]
                linhas_inst.append([nome, os.path.basename(p)[:9], classe, f"{bw:.0f}", f"{bh:.0f}",
                                    f"{bw * args.imgsz / w:.0f}", cat])
                if classe in ("gloves", "boots"):
                    ladrilhos.append(ladrilho(im, gc[j], cat, classe))
        for classe, d in cont.items():
            linhas_cat.append([nome, classe] + [d[c] for c in CATEGORIAS])
        print(nome, {c: d for c, d in cont.items()})
        if ladrilhos:
            vazio = np.full_like(ladrilhos[0], 24)
            filas = [np.hstack(ladrilhos[k:k + 6] + [vazio] * (6 - len(ladrilhos[k:k + 6])))
                     for k in range(0, len(ladrilhos), 6)]
            cv2.imwrite(os.path.join(SAIDA, f"folha_luvas_botas_{nome}.jpg"), np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 88])

    with open(os.path.join(SAIDA, "categorias.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "classe"] + CATEGORIAS)
        w.writerows(linhas_cat)
    with open(os.path.join(SAIDA, "instancias.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "imagem", "classe", "largura_px", "altura_px", f"largura_px_em_{args.imgsz}", "categoria"])
        w.writerows(linhas_inst)
    print(f"resultados em {SAIDA}/")


if __name__ == "__main__":
    main()
