"""
Unifica os datasets baixados em ~/Desktop/Novo Dataset/ nas 5 classes do
projeto, gerando uma pasta pronta para subir no Roboflow e revisar a mao.

O que faz, por dataset:
1. Remapeia as classes pelo NOME (cada dataset tem nomes/ordem diferentes;
   o VEST usa ate '0'..'4' - identificados olhando recortes das caixas).
   Classes que nao se encaixam (person, no-*, ear_protection, spec...) tem a
   caixa descartada, mas a imagem continua.
2. Remove as copias de augmentation do Roboflow (mesma foto com sufixo
   .rf.<hash>) - senao o treino ve a mesma foto varias vezes e a mesma foto
   pode cair em train e valid ao mesmo tempo.
3. Pre-anota com o modelo atual (best.pt) as classes que o dataset NAO
   anota. Ex.: o seafty_goggles so marca oculos - um capacete sem caixa ali
   ensinaria o modelo que capacete e "fundo". As caixas pre-anotadas sao
   listadas no relatorio.csv para revisar primeiro.

Saida em datasets/novos_unificado/ (train/valid/test + data.yaml) e
datasets/novos_unificado/relatorio.csv.

Uso:
    venv_win\\Scripts\\python scripts/unify_novos_datasets.py
"""

import csv
import glob
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import yaml
from ultralytics import YOLO

SRC_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "Novo Dataset")
OUT_DIR = "datasets/novos_unificado"
MODEL_PATH = "models/best.pt"

FINAL_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

PRE_CONF = 0.35
IMGSZ = 640
SEED = 0

# cover: classes que o dataset anota de forma confiavel (as outras sao
#        pre-anotadas pelo modelo). "per_image" = o dataset e uma colcha de
#        retalhos, entao so confia nas classes que aparecem em cada imagem.
# max_images: limite de fotos unicas (None = todas).
DATASETS = [
    {
        "pattern": "EPI dataset*",
        "prefix": "epids",
        "map": {"helmet": "helmet", "vest": "vest", "glasses": "goggles"},
        "cover": {"helmet", "vest", "goggles"},
        "max_images": None,
    },
    {
        "pattern": "PPE Detection Using CV*",
        "prefix": "ppecv",
        "map": {"helmet": "helmet", "vest": "vest", "goggles": "goggles"},
        "cover": {"helmet", "vest", "goggles"},
        "max_images": None,
    },
    {
        "pattern": "VEST.v1i*",
        "prefix": "vestds",
        "map": {
            "0": "boots", "Safety-Boot": "boots",
            "2": "gloves", "Gloves": "gloves",
            "3": "helmet", "Helmet": "helmet", "helmet": "helmet",
            "4": "vest", "Vest": "vest", "vest": "vest", "Safety-Vest": "vest",
            "Reflective-Vest": "vest", "Yelek": "vest",
            "safety vest - v1 2023-11-19 5-12pm": "vest",
            "Goggles": "goggles",
            # "1" e "Glass" ficam de fora: marcam a regiao dos olhos mesmo em
            # quem esta SEM oculos - o modelo pre-anota goggles no lugar
        },
        "cover": "per_image",
        "max_images": None,
    },
    {
        "pattern": "autolabeling-glasses*",
        "prefix": "autogl",
        "map": {"helmet": "helmet", "vest": "vest", "gloves": "gloves",
                "with glasses": "goggles"},
        "cover": {"helmet", "vest", "gloves", "goggles"},
        "max_images": None,
    },
    {
        "pattern": "seafty_goggles*",
        "prefix": "sgogg",
        # "spec" = oculos de grau comum, nao e EPI: a caixa sai e a imagem
        # fica como exemplo negativo de goggles
        "map": {"goggles": "goggles"},
        "cover": {"goggles"},
        # 6.4 mil fotos, muitas de camera de seguranca escura, e goggles ja
        # esta forte na v2 - limita para nao afogar as outras classes
        "max_images": 2000,
    },
]


def find_dir(pattern):
    dirs = [d for d in glob.glob(os.path.join(SRC_ROOT, pattern)) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"Nenhuma pasta para '{pattern}' em {SRC_ROOT}")
    return sorted(dirs)[0]


def load_names(ds_dir):
    with open(os.path.join(ds_dir, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def read_boxes(lbl_path, names, class_map):
    """Le um label YOLO e devolve [(classe_final, x, y, w, h)].
    Poligonos (segmentacao) viram a caixa que os envolve."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            final = class_map.get(names[int(parts[0])])
            if final is None:
                continue
            vals = list(map(float, parts[1:]))
            if len(vals) == 4:
                x, y, w, h = vals
            else:
                xs, ys = vals[0::2], vals[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            boxes.append((final, x, y, w, h))
    return boxes


def collect_unique(ds_dir):
    """Uma entrada por foto original: {chave: (split, img_path, lbl_path)}.
    Copias de augmentation (.rf.<hash>) da mesma foto sao descartadas."""
    unique = {}
    for split in ("train", "valid", "test"):
        for img in sorted(glob.glob(os.path.join(ds_dir, split, "images", "*"))):
            base = os.path.basename(img)
            key = re.split(r"\.rf\.", base)[0]
            if key in unique:
                continue
            lbl = os.path.join(ds_dir, split, "labels", os.path.splitext(base)[0] + ".txt")
            unique[key] = (split, img, lbl)
    return unique


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(OUT_DIR, split, "images"))
        os.makedirs(os.path.join(OUT_DIR, split, "labels"))

    model = YOLO(MODEL_PATH)
    rng = random.Random(SEED)
    report = []
    totals = defaultdict(Counter)

    for cfg in DATASETS:
        ds_dir = find_dir(cfg["pattern"])
        names = load_names(ds_dir)
        entries = collect_unique(ds_dir)
        items = [(k, *v) for k, v in entries.items()]

        cap = cfg["max_images"]
        if cap and len(items) > cap:
            # prioriza fotos que tem pelo menos uma caixa util
            com = [i for i in items if read_boxes(i[3], names, cfg["map"])]
            sem = [i for i in items if i not in com]
            rng.shuffle(com)
            rng.shuffle(sem)
            items = (com + sem)[:cap]

        print(f"\n[{cfg['prefix']}] {os.path.basename(ds_dir)}: "
              f"{len(entries)} fotos unicas, usando {len(items)}")

        for idx, (key, split, img, lbl) in enumerate(items, 1):
            boxes = read_boxes(lbl, names, cfg["map"])
            if cfg["cover"] == "per_image":
                cover = {b[0] for b in boxes}
            else:
                cover = cfg["cover"]
            missing = [c for c in FINAL_CLASSES if c not in cover]

            pre = []
            if missing:
                res = model.predict(img, conf=PRE_CONF, imgsz=IMGSZ, verbose=False)[0]
                for c, xywhn in zip(res.boxes.cls, res.boxes.xywhn):
                    cname = FINAL_CLASSES[int(c)]
                    if cname in missing:
                        pre.append((cname, *xywhn.tolist()))

            new_base = f"{cfg['prefix']}_{idx:05d}"
            ext = os.path.splitext(img)[1].lower()
            shutil.copy(img, os.path.join(OUT_DIR, split, "images", new_base + ext))
            with open(os.path.join(OUT_DIR, split, "labels", new_base + ".txt"), "w") as f:
                f.write("\n".join(
                    f"{FINAL_CLASSES.index(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                    for c, x, y, w, h in boxes + pre
                ))

            for b in boxes:
                totals[cfg["prefix"]][b[0]] += 1
            for b in pre:
                totals[cfg["prefix"]][b[0] + " (pre)"] += 1
            report.append({
                "arquivo": new_base + ext,
                "split": split,
                "dataset": os.path.basename(ds_dir),
                "original": os.path.basename(img),
                "caixas_originais": len(boxes),
                "caixas_pre_anotadas": len(pre),
                "classes_pre_anotadas": " ".join(sorted({b[0] for b in pre})),
            })
            if idx % 500 == 0:
                print(f"  {idx}/{len(items)}")

    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(FINAL_CLASSES)}\nnames: {FINAL_CLASSES}\n")

    # pre-anotadas primeiro: sao as que mais precisam de revisao
    report.sort(key=lambda r: -r["caixas_pre_anotadas"])
    with open(os.path.join(OUT_DIR, "relatorio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

    print("\n===== Resumo de caixas por dataset =====")
    for prefix, cnt in totals.items():
        print(f"  {prefix}: {dict(sorted(cnt.items()))}")
    print(f"\nTotal: {len(report)} imagens -> {OUT_DIR}/")
    print(f"Imagens com caixas pre-anotadas (revisar primeiro): "
          f"{sum(1 for r in report if r['caixas_pre_anotadas'])}")


if __name__ == "__main__":
    main()
