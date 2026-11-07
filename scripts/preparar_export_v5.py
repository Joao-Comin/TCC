"""
Prepara o export YOLOv11 do projeto TCC-EPI do Roboflow (dataset novo
unificado, ja com as revisoes feitas la) para entrar no treino da v5:

1. Remapeia as classes pelo NOME (o Roboflow exporta em ordem alfabetica:
   boots, gloves, goggles, helmet, vest) para a nossa ordem.
2. Tira as imagens duvidosas que nao foram revisadas: marcadas talvez-gloves
   (ou talvez-goggles) no relatorio.csv e que continuam SEM caixa dessa
   classe. Nelas o DINO viu o EPI com score baixo; como a revisao de ~100
   delas achou luva real em quase todas, mante-las sem caixa ensinaria o
   modelo que luva e fundo. As que ganharam caixa na revisao ficam.

Saida: datasets/novos_v5/ (train/valid/test) e novos_v5.zip na raiz, para
subir no Drive (MyDrive/TCC/yolo_dataset/novos_v5.zip).

Uso:
    venv_win\\Scripts\\python scripts/preparar_export_v5.py <pasta_do_export>
"""

import csv
import glob
import os
import re
import shutil
import sys
import zipfile
from collections import Counter

import yaml

RELATORIO = "datasets/novos_unificado/relatorio.csv"
OUT_DIR = "datasets/novos_v5"
ZIP_PATH = "novos_v5.zip"
OUR_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

# epids_00001.jpg vira epids_00001_jpg.rf.<hash>.jpg no export
RF_NOME = re.compile(r"^(.+)_(jpg|jpeg|png)\.rf\.[0-9a-f]+\.\w+$", re.IGNORECASE)


def nome_original(arquivo):
    m = RF_NOME.match(arquivo)
    return f"{m.group(1)}.{m.group(2)}" if m else arquivo


def main():
    export_dir = sys.argv[1]
    with open(os.path.join(export_dir, "data.yaml"), encoding="utf-8") as f:
        nomes_rf = yaml.safe_load(f)["names"]
    if isinstance(nomes_rf, dict):
        nomes_rf = [nomes_rf[k] for k in sorted(nomes_rf)]
    remap = {i: OUR_CLASSES.index(n) for i, n in enumerate(nomes_rf)}
    print("classes do export:", nomes_rf, "-> remap", remap)

    with open(RELATORIO, encoding="utf-8") as f:
        rel = {r["arquivo"]: r for r in csv.DictReader(f)}

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    mantidas, tiradas, sem_relatorio = Counter(), Counter(), 0
    caixas = Counter()
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split, "images"))
        os.makedirs(os.path.join(OUT_DIR, split, "labels"))
        for img in sorted(glob.glob(os.path.join(export_dir, split, "images", "*"))):
            arq = os.path.basename(img)
            lbl = os.path.join(export_dir, split, "labels", os.path.splitext(arq)[0] + ".txt")
            linhas = []
            if os.path.exists(lbl):
                for l in open(lbl):
                    p = l.split()
                    if len(p) == 5:
                        linhas.append([remap[int(p[0])]] + p[1:])
            classes = {c for c, *_ in linhas}

            r = rel.get(nome_original(arq))
            if r is None:
                sem_relatorio += 1
            elif (r.get("talvez_gloves") == "1" and 2 not in classes) or \
                 (r.get("talvez_goggles") == "1" and 4 not in classes):
                tiradas[split] += 1
                continue

            mantidas[split] += 1
            for c, *_ in linhas:
                caixas[OUR_CLASSES[c]] += 1
            shutil.copy(img, os.path.join(OUT_DIR, split, "images", arq))
            with open(os.path.join(OUT_DIR, split, "labels", os.path.splitext(arq)[0] + ".txt"), "w") as f:
                f.write("".join(" ".join(map(str, l)) + "\n" for l in linhas))

    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(OUR_CLASSES)}\nnames: {OUR_CLASSES}\n")

    print("mantidas:", dict(mantidas), "| tiradas (duvidosas):", dict(tiradas))
    if sem_relatorio:
        print(f"AVISO: {sem_relatorio} imagens sem linha no relatorio (mantidas)")
    print("caixas por classe:", dict(caixas))

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(OUT_DIR)))
    print(f"{ZIP_PATH}: {os.path.getsize(ZIP_PATH) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
