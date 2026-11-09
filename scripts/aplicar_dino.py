"""
Aplica em datasets/novos_unificado/ o resultado do notebook
notebooks/pre_anotar_luvas_colab.ipynb (dino_deteccoes.csv).

Limiares calibrados conferindo a olho 30 recortes por faixa de score
(folhas em diag_tmp/calibracao/):

  gloves   >= 0.50  ~90% certas  -> vira caixa (tag dino-gloves)
  gloves   0.35-0.50  ~40% certas  -> sem caixa, so tag talvez-gloves
  goggles  >= 0.60  ~85% certas  -> vira caixa (tag dino-goggles)
  goggles  0.50-0.60  ~40% certas  -> sem caixa, so tag talvez-goggles
  goggles  < 0.50                  -> ignorado (quase tudo mascara/abafador)

Grava no relatorio.csv as colunas dino_gloves, dino_goggles (caixas
adicionadas), talvez_gloves e talvez_goggles (0/1), que o
scripts/subir_roboflow.py usa para as tags.

Pode rodar de novo sem duplicar caixas: deteccao que ja tem caixa parecida
da mesma classe no rotulo (IoU >= 0.5) e pulada.

Uso:
    venv_win\\Scripts\\python scripts/aplicar_dino.py
    venv_win\\Scripts\\python scripts/aplicar_dino.py --dataset-dir datasets/yolo_v2 --csv datasets/yolo_v2/dino_deteccoes_v2.csv

Se o dataset nao tiver relatorio.csv (caso do dataset da v2), um novo e
criado com uma linha por imagem.
"""

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict

DATASET_DIR = "datasets/novos_unificado"
CLASS_ID = {"gloves": 2, "goggles": 4}
IOU_DUP = 0.5


def xyxy(xc, yc, w, h):
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gloves-min", type=float, default=0.50)
    ap.add_argument("--goggles-min", type=float, default=0.60)
    ap.add_argument("--talvez-gloves-min", type=float, default=0.35)
    ap.add_argument("--talvez-goggles-min", type=float, default=0.50)
    ap.add_argument("--dataset-dir", default=DATASET_DIR)
    ap.add_argument("--csv", default=None, help="padrao: <dataset-dir>/dino_deteccoes.csv")
    args = ap.parse_args()
    dataset_dir = args.dataset_dir
    dino_csv = args.csv or os.path.join(dataset_dir, "dino_deteccoes.csv")
    relatorio = os.path.join(dataset_dir, "relatorio.csv")
    caixa_min = {"gloves": args.gloves_min, "goggles": args.goggles_min}
    talvez_min = {"gloves": args.talvez_gloves_min, "goggles": args.talvez_goggles_min}

    caixas = defaultdict(list)    # (split, arquivo) -> [(classe, xc, yc, w, h)]
    talvez = defaultdict(set)     # classe -> {arquivo}
    with open(dino_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cls, score = r["classe"], float(r["score"])
            if score >= caixa_min[cls]:
                caixas[(r["split"], r["arquivo"])].append((cls, *(float(r[k]) for k in ("xc", "yc", "w", "h"))))
            elif score >= talvez_min[cls]:
                talvez[cls].add(r["arquivo"])

    adicionadas = defaultdict(Counter)  # arquivo -> classe -> n
    for (split, arquivo), dets in caixas.items():
        lbl = os.path.join(dataset_dir, split, "labels", os.path.splitext(arquivo)[0] + ".txt")
        linhas = [l for l in open(lbl).read().splitlines() if l.strip()] if os.path.exists(lbl) else []
        existentes = [(int(l.split()[0]), xyxy(*map(float, l.split()[1:]))) for l in linhas]
        novas = []
        for cls, *c in dets:
            cid = CLASS_ID[cls]
            if any(e_cid == cid and iou(xyxy(*c), e) >= IOU_DUP for e_cid, e in existentes):
                continue
            existentes.append((cid, xyxy(*c)))
            novas.append(f"{cid} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f}")
            adicionadas[arquivo][cls] += 1
        if novas:
            with open(lbl, "w") as f:
                f.write("\n".join(linhas + novas) + "\n")

    if os.path.exists(relatorio):
        with open(relatorio, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [{"arquivo": os.path.basename(p), "split": p.replace("\\", "/").split("/")[-3]}
                for p in sorted(glob.glob(os.path.join(dataset_dir, "*", "images", "*")))]
    campos = [c for c in rows[0].keys() if c != "suspeita_goggles"]
    for c in ("dino_gloves", "dino_goggles", "talvez_gloves", "talvez_goggles"):
        if c not in campos:
            campos.append(c)
    for r in rows:
        a = r["arquivo"]
        for cls in ("gloves", "goggles"):
            r[f"dino_{cls}"] = str(int(r.get(f"dino_{cls}") or 0) + adicionadas[a][cls])
            # "talvez" so faz sentido se a imagem nao ganhou caixa dessa classe
            r[f"talvez_{cls}"] = "1" if a in talvez[cls] and r[f"dino_{cls}"] == "0" else "0"
    with open(relatorio, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for cls in ("gloves", "goggles"):
        n_caixas = sum(c[cls] for c in adicionadas.values())
        n_imgs = sum(1 for c in adicionadas.values() if c[cls])
        n_talvez = sum(1 for r in rows if r[f"talvez_{cls}"] == "1")
        por_ds = Counter(a.split("_")[0] for a, c in adicionadas.items() if c[cls])
        print(f"{cls}: +{n_caixas} caixas em {n_imgs} imagens {dict(por_ds)} | talvez-{cls}: {n_talvez} imagens")


if __name__ == "__main__":
    main()
