"""
Monta a v2 "limpa" para o treino da v6, a partir de datasets/yolo_v2 ja com
as caixas do Grounding DINO aplicadas (scripts/aplicar_dino.py):

1. Mantem so as fontes que anotam varias classes e estao mais perto do
   dominio do video: testcasque (5 classes), ensetest (helmet/vest/boots,
   maior fonte de botas) e glovesds (luvas). Saem gogglesds (so oculos,
   frames de video de review com filtro) e mschaberle (so capacete, fotos
   de jornal, coletes visiveis sem caixa).
2. Tira as imagens da faixa duvidosa de LUVA do DINO (talvez_gloves no
   relatorio.csv) que continuam sem caixa de luva. Diferente do dataset
   novo, aqui as de oculos duvidosos (talvez_goggles) ficam: tira-las
   removia ~1/4 das imagens e junto muitas botas, a classe mais fraca;
   oculos sem caixa custa menos que perder essas botas.

Saida: datasets/yolo_v2_limpo/ (train/val/test) e yolo_v2_limpo.zip na raiz,
para subir em MyDrive/TCC/yolo_dataset/.

Uso:
    venv_win\\Scripts\\python scripts/preparar_v2_limpo.py
"""

import csv
import os
import shutil
import zipfile
from collections import Counter

SRC = "datasets/yolo_v2"
OUT_DIR = "datasets/yolo_v2_limpo"
ZIP_PATH = "yolo_v2_limpo.zip"
FONTES = ("testcasque_", "ensetest_", "glovesds_")
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]


def main():
    with open(os.path.join(SRC, "relatorio.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    mantidas, fora_fonte, tiradas, caixas = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        arq, split = r["arquivo"], r["split"]
        fonte = arq.split("_")[0]
        if not arq.startswith(FONTES):
            fora_fonte[fonte] += 1
            continue
        lbl = os.path.join(SRC, split, "labels", os.path.splitext(arq)[0] + ".txt")
        linhas = [l for l in open(lbl).read().splitlines() if l.split()] if os.path.exists(lbl) else []
        classes = {int(l.split()[0]) for l in linhas}
        if r.get("talvez_gloves") == "1" and 2 not in classes:
            tiradas[fonte] += 1
            continue
        for d in ("images", "labels"):
            os.makedirs(os.path.join(OUT_DIR, split, d), exist_ok=True)
        shutil.copy(os.path.join(SRC, split, "images", arq), os.path.join(OUT_DIR, split, "images", arq))
        with open(os.path.join(OUT_DIR, split, "labels", os.path.basename(lbl)), "w") as f:
            f.write("".join(l + "\n" for l in linhas))
        mantidas[split] += 1
        for l in linhas:
            caixas[CLASSES[int(l.split()[0])]] += 1

    print("fontes retiradas:", dict(fora_fonte))
    print("tiradas por faixa duvidosa do DINO:", dict(tiradas))
    print("mantidas por split:", dict(mantidas), "| total:", sum(mantidas.values()))
    print("caixas por classe:", dict(caixas))

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(OUT_DIR)))
    print(f"{ZIP_PATH}: {os.path.getsize(ZIP_PATH) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
