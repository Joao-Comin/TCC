"""
Mede, por dataset de origem, quantos EPIs aparecem na imagem mas NAO tem
caixa no rotulo (anotacao incompleta ensina o modelo que aquele EPI e fundo).

Usa o Grounding DINO (detector que acha objetos pelo nome, independente do
nosso best.pt - entao enxerga tambem o que o nosso modelo erra, tipo luva).
E lento no CPU (~8 s/imagem), por isso roda numa AMOSTRA por dataset, nao no
dataset inteiro. Ele tambem erra (marca bone como capacete, camisa como
colete), entao os numeros sao estimativa - conferir amostras a olho antes de
decidir algo.

Saida: diag_tmp/auditoria/deteccoes.csv (uma linha por caixa do DINO sem
caixa correspondente no rotulo) e resumo impresso no final.

Uso:
    venv_win\\Scripts\\python scripts/auditar_rotulos.py --por-dataset 60
"""

import argparse
import csv
import glob
import os
import random
from collections import Counter, defaultdict

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DATASET_DIR = "datasets/novos_unificado"
OUT_DIR = "diag_tmp/auditoria"
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]
PROMPTS = {"hard hat": "helmet", "safety vest": "vest", "glove": "gloves", "boot": "boots", "safety goggles": "goggles"}
PREFIXOS = ["epids", "ppecv", "vestds", "autogl", "sgogg"]

SCORE_MIN = 0.35
IOU_MATCH = 0.3


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0


def ler_rotulo(path, w, h):
    caixas = []
    if os.path.exists(path):
        for linha in open(path):
            p = linha.split()
            if len(p) == 5:
                c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
                caixas.append((CLASSES[c], ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)))
    return caixas


def classe_do_texto(texto):
    # o DINO as vezes junta dois prompts ("hard hat safety goggles"); nesses casos descarta
    achadas = [cls for prompt, cls in PROMPTS.items() if prompt in texto]
    return achadas[0] if len(achadas) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-dataset", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    amostra = []
    for pref in PREFIXOS:
        imgs = sorted(glob.glob(os.path.join(DATASET_DIR, "train", "images", f"{pref}_*")))
        amostra += rng.sample(imgs, min(args.por_dataset, len(imgs)))

    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
    texto = ". ".join(PROMPTS) + "."

    faltando = defaultdict(Counter)       # prefixo -> classe -> caixas sem rotulo
    imgs_com_falta = defaultdict(Counter)  # prefixo -> classe -> imagens com >=1 caixa sem rotulo
    rotuladas = defaultdict(Counter)       # prefixo -> classe -> caixas no rotulo

    with open(os.path.join(OUT_DIR, "deteccoes.csv"), "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["arquivo", "classe", "score", "x1", "y1", "x2", "y2"])
        for i, img_path in enumerate(amostra, 1):
            nome = os.path.basename(img_path)
            pref = nome.split("_")[0]
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            gt = ler_rotulo(os.path.join(DATASET_DIR, "train", "labels", os.path.splitext(nome)[0] + ".txt"), w, h)
            for cls, _ in gt:
                rotuladas[pref][cls] += 1

            inp = proc(images=im, text=texto, return_tensors="pt")
            with torch.no_grad():
                res = model(**inp)
            r = proc.post_process_grounded_object_detection(
                res, inp.input_ids, threshold=SCORE_MIN, text_threshold=0.25, target_sizes=[(h, w)]
            )[0]

            classes_faltando = set()
            for txt, score, box in zip(r["text_labels"], r["scores"], r["boxes"]):
                cls = classe_do_texto(txt)
                if cls is None:
                    continue
                box = box.tolist()
                if any(c == cls and iou(box, b) >= IOU_MATCH for c, b in gt):
                    continue
                faltando[pref][cls] += 1
                classes_faltando.add(cls)
                out.writerow([nome, cls, round(float(score), 3), *[round(v) for v in box]])
            for cls in classes_faltando:
                imgs_com_falta[pref][cls] += 1
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(amostra)}", flush=True)

    n = Counter(os.path.basename(p).split("_")[0] for p in amostra)
    print("\n% de imagens da amostra com pelo menos 1 EPI detectado SEM caixa no rotulo")
    print(f"{'dataset':8}" + "".join(f"{c:>9}" for c in CLASSES))
    for pref in PREFIXOS:
        print(f"{pref:8}" + "".join(f"{100 * imgs_com_falta[pref][c] / n[pref]:8.0f}%" for c in CLASSES))
    print("\ncaixas no rotulo (amostra), para comparar")
    for pref in PREFIXOS:
        print(f"{pref:8}" + "".join(f"{rotuladas[pref][c]:>9}" for c in CLASSES))


if __name__ == "__main__":
    main()
