import os
import random
import shutil

SEED = 0
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

SRC_IMAGES = "datasets/final/images"
SRC_LABELS = "datasets/final/labels"
OUT_DIR = "datasets/final_split"


def main():
    random.seed(SEED)

    bases = [os.path.splitext(f)[0] for f in os.listdir(SRC_IMAGES)]
    random.shuffle(bases)

    n = len(bases)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])

    splits = {
        "train": bases[:n_train],
        "val": bases[n_train:n_train + n_val],
        "test": bases[n_train + n_val:],
    }

    for split, split_bases in splits.items():
        img_out = os.path.join(OUT_DIR, split, "images")
        lbl_out = os.path.join(OUT_DIR, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for base in split_bases:
            for ext in (".jpg", ".jpeg", ".png"):
                img_src = os.path.join(SRC_IMAGES, base + ext)
                if os.path.exists(img_src):
                    shutil.copy(img_src, os.path.join(img_out, base + ext))
                    break

            lbl_src = os.path.join(SRC_LABELS, base + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(lbl_out, base + ".txt"))
            else:
                open(os.path.join(lbl_out, base + ".txt"), "w").close()

        print(f"{split}: {len(split_bases)} imagens")

    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print("Concluído! Dataset dividido em", OUT_DIR)


if __name__ == "__main__":
    main()
