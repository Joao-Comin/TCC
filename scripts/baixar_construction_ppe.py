"""
Baixa as imagens do dataset Construction-PPE (Ultralytics Platform) a partir
do manifesto .ndjson exportado (URLs assinadas, com expiracao). Mantem os
splits originais (train/val/test) e converte anotacoes para .txt YOLO,
descartando as classes fora do nosso escopo (Person, none, no_*).

As caixas no ndjson ja vem no formato [class_id, x_center, y_center, w, h]
normalizado, ou seja, ja e formato YOLO - so precisa filtrar/remapear classe.
"""

import json
import os
import time
import urllib.request
import urllib.error

NDJSON_PATH = r"C:\Users\João Victor Comin\Downloads\construction-ppe.ndjson"
OUT_DIR = "construction_ppe_raw"

# so as 5 classes do nosso escopo atual; indices no dataset de origem -> nome
SOURCE_CLASSES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles"}
OUR_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]


def download_one(url, dest_path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            return True
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"  FALHOU: {dest_path} ({e})")
                return False
            time.sleep(1)
    return False


def main():
    header = None
    images = []
    with open(NDJSON_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["type"] == "dataset":
                header = obj
            else:
                images.append(obj)

    print(f"Total de imagens no manifesto: {len(images)}")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    ok, fail = 0, 0
    for i, img in enumerate(images, 1):
        split = img.get("split", "train")
        fname = img["file"]
        url = img["url"]

        img_dest = os.path.join(OUT_DIR, split, "images", fname)
        lbl_dest = os.path.join(OUT_DIR, split, "labels", os.path.splitext(fname)[0] + ".txt")

        if os.path.exists(img_dest):
            ok += 1
        else:
            success = download_one(url, img_dest)
            if success:
                ok += 1
            else:
                fail += 1
                continue

        # anotacoes: mantem so as classes do nosso escopo (0-4), remapeando por nome
        boxes = img.get("annotations", {}).get("boxes", [])
        lines = []
        for b in boxes:
            cls_id = b[0]
            if cls_id not in SOURCE_CLASSES:
                continue
            class_name = SOURCE_CLASSES[cls_id]
            new_idx = OUR_CLASSES.index(class_name)
            x, y, w, h = b[1], b[2], b[3], b[4]
            lines.append(f"{new_idx} {x} {y} {w} {h}")

        with open(lbl_dest, "w") as f:
            f.write("\n".join(lines))

        if i % 100 == 0:
            print(f"  {i}/{len(images)} processadas ({ok} ok, {fail} falhas)")

    print(f"\nConcluido: {ok} imagens baixadas, {fail} falhas.")


if __name__ == "__main__":
    main()
