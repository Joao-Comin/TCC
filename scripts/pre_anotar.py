"""
Pre-anotacao automatica: roda o modelo atual (best.pt) sobre as imagens novas
e gera arquivos de anotacao YOLO ja preenchidos, para serem CORRIGIDOS a mao
em vez de desenhados do zero.

Saida em pre_anotado/:
  - imagens (copiadas)
  - <nome>.txt  (anotacoes no formato YOLO)
  - classes.txt (nomes das classes, exigido pelo LabelImg/Roboflow)

Confianca baixa de proposito: e mais rapido apagar uma caixa errada do que
desenhar uma que faltou.
"""

import glob
import os
import shutil
from collections import Counter

from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
SRC_DIR = "novos_frames"
OUT_DIR = "pre_anotado"
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

CONF = 0.20
IMGSZ = 960


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    images = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    images += sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))

    counts = Counter()
    sem_deteccao = 0

    for idx, img_path in enumerate(images, 1):
        # nome limpo, sem espacos/acentos (evita dor de cabeca nas ferramentas)
        novo_nome = f"frame_{idx:03d}"
        ext = os.path.splitext(img_path)[1].lower()
        shutil.copy(img_path, os.path.join(OUT_DIR, novo_nome + ext))

        results = model.predict(img_path, conf=CONF, imgsz=IMGSZ, verbose=False)
        boxes = results[0].boxes

        linhas = []
        for cls_id, xywhn in zip(boxes.cls, boxes.xywhn):
            c = int(cls_id)
            x, y, w, h = xywhn.tolist()
            linhas.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts[CLASSES[c]] += 1

        if not linhas:
            sem_deteccao += 1

        with open(os.path.join(OUT_DIR, novo_nome + ".txt"), "w") as f:
            f.write("\n".join(linhas))

    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES))

    print(f"\n{len(images)} imagens processadas -> {OUT_DIR}/")
    print(f"\nCaixas pre-anotadas por classe:")
    total = 0
    for c in CLASSES:
        print(f"  - {c}: {counts[c]}")
        total += counts[c]
    print(f"  TOTAL: {total} caixas")
    print(f"\nImagens sem nenhuma deteccao (precisam ser anotadas do zero): {sem_deteccao}")


if __name__ == "__main__":
    main()
