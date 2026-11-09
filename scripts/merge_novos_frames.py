"""
Integra as imagens anotadas manualmente (exportadas do Roboflow) num pacote
pronto para mesclar no dataset principal DENTRO DO COLAB (o dataset grande
de ~12 mil imagens nao fica mais neste disco local, so no Drive).

Faz 3 coisas:
1. Remapeia os indices de classe (o Roboflow exporta em ordem alfabetica,
   diferente da nossa ordem original) usando o NOME da classe como chave,
   nunca o indice cru.
2. Separa ~20% das imagens novas como "conjunto de teste de video real"
   (novos_frames_dataset/real_world_test/) - nunca entra no treino, serve so
   para medir o modelo antes/depois de forma honesta nesse tipo de imagem.
3. O restante (~80%) recebe oversampling (copiado varias vezes com nomes
   distintos) e vai para novos_frames_dataset/train_extra/ - sem isso, 100
   imagens novas morreriam afogadas em ~12 mil imagens antigas e o treino
   praticamente as ignoraria.

Depois, zipar novos_frames_dataset/ e subir pro Drive junto do dataset
principal; no Colab, copiar train_extra/ para dentro de yolo/train/ antes
de treinar.
"""

import glob
import os
import random
import shutil

import yaml

EXPORT_DIR = "roboflow_export"
OUR_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

OUT_DIR = "novos_frames_dataset"
TRAIN_DIR = os.path.join(OUT_DIR, "train_extra")
TEST_REAL_DIR = os.path.join(OUT_DIR, "real_world_test")

SEED = 0
TEST_FRACTION = 0.20
OVERSAMPLE_FACTOR = 8


def load_roboflow_classes():
    with open(os.path.join(EXPORT_DIR, "data.yaml")) as f:
        data = yaml.safe_load(f)
    return data["names"]


def collect_pairs():
    """Junta imagens+labels dos 3 splits do Roboflow num pool unico."""
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(EXPORT_DIR, split, "images")
        lbl_dir = os.path.join(EXPORT_DIR, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for img_path in glob.glob(os.path.join(img_dir, "*")):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            pairs.append((img_path, lbl_path if os.path.exists(lbl_path) else None))
    return pairs


def remap_label_lines(lbl_path, roboflow_classes):
    if lbl_path is None:
        return []
    with open(lbl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    new_lines = []
    for line in lines:
        parts = line.split()
        old_idx = int(parts[0])
        class_name = roboflow_classes[old_idx]
        new_idx = OUR_CLASSES.index(class_name)
        new_lines.append(" ".join([str(new_idx)] + parts[1:]))
    return new_lines


def main():
    roboflow_classes = load_roboflow_classes()
    print("Classes no export do Roboflow:", roboflow_classes)
    print("Nossas classes:", OUR_CLASSES)

    pairs = collect_pairs()
    print(f"\nTotal de imagens novas encontradas: {len(pairs)}")

    random.seed(SEED)
    random.shuffle(pairs)

    n_test = int(len(pairs) * TEST_FRACTION)
    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]

    print(f"-> {len(test_pairs)} reservadas para datasets/real_world_test/ (nunca treina)")
    print(f"-> {len(train_pairs)} vao para treino, com oversample {OVERSAMPLE_FACTOR}x "
          f"= {len(train_pairs) * OVERSAMPLE_FACTOR} copias")

    # 1) conjunto de teste de video real (sem oversample, sem augmentation)
    os.makedirs(os.path.join(TEST_REAL_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(TEST_REAL_DIR, "labels"), exist_ok=True)
    for img_path, lbl_path in test_pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(TEST_REAL_DIR, "images", base + ext))
        lines = remap_label_lines(lbl_path, roboflow_classes)
        with open(os.path.join(TEST_REAL_DIR, "labels", base + ".txt"), "w") as f:
            f.write("\n".join(lines))

    # 2) treino, com oversample
    train_img_dir = os.path.join(TRAIN_DIR, "images")
    train_lbl_dir = os.path.join(TRAIN_DIR, "labels")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    for img_path, lbl_path in train_pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        lines = remap_label_lines(lbl_path, roboflow_classes)

        for rep in range(OVERSAMPLE_FACTOR):
            new_base = f"realframe_{base}_rep{rep}"
            shutil.copy(img_path, os.path.join(train_img_dir, new_base + ext))
            with open(os.path.join(train_lbl_dir, new_base + ".txt"), "w") as f:
                f.write("\n".join(lines))

    print("\nConcluido.")
    print(f"  datasets/real_world_test/: {len(test_pairs)} imagens (avaliacao)")
    print(f"  {TRAIN_DIR}/: +{len(train_pairs) * OVERSAMPLE_FACTOR} imagens (treino, com repeticao)")


if __name__ == "__main__":
    main()
