import os
import glob
import shutil
import yaml

# Mapeamento de classes originais -> finais
CLASS_MAP = {
    "helmet": "helmet",
    "hard_hat": "helmet",
    "boots": "boots",
    "shoes": "boots",
    "gloves": "gloves",
    "Gloves": "gloves",
    "goggles": "goggles",
    "Goggles": "goggles",
    "eyewear": "goggles",
    "glasses": "goggles",
    "vest": "vest",
    # "person" não entra: já é bem coberto pelo YOLO pré-treinado em COCO,
    # não vale a pena diluir o dataset customizado com poucas instâncias próprias
    # classes descartadas não entram no mapa (ex: "No-Gloves", "person"/"Person")
}

FINAL_CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]

BASE_DIR = "datasets/originals"


def find_dataset_dir(base_dir, pattern):
    """Encontra a pasta do dataset pelo início do nome (evita erro de nome exato)."""
    matches = glob.glob(os.path.join(base_dir, pattern))
    dirs = [m for m in matches if os.path.isdir(m)]  # ignora os .zip
    if not dirs:
        raise FileNotFoundError(
            f"Nenhuma pasta encontrada para o padrão: {pattern} em {base_dir}"
        )
    if len(dirs) > 1:
        print(f"Aviso: múltiplas pastas encontradas para '{pattern}', usando: {dirs[0]}")
    return dirs[0]


# Caminhos reais das pastas (resolvidos dinamicamente)
DATASETS = {
    "testcasque": find_dataset_dir(BASE_DIR, "PPE detection*"),
    "mschaberle": find_dataset_dir(BASE_DIR, "EPI.v1-hardhat*"),
    "ensetest": find_dataset_dir(BASE_DIR, "EPI.v2-final_version*"),
    "glovesds": find_dataset_dir(BASE_DIR, "Gloves.v1i*"),
    "gogglesds": find_dataset_dir(BASE_DIR, "Goggles.v3i*"),
}

OUTPUT_IMAGES = "datasets/final/images"
OUTPUT_LABELS = "datasets/final/labels"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


def load_classes(dataset_path):
    """Procura recursivamente o data.yaml dentro do dataset"""
    for root, dirs, files in os.walk(dataset_path):
        if "data.yaml" in files:
            yaml_path = os.path.join(root, "data.yaml")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return {i: name for i, name in enumerate(data["names"])}
    raise FileNotFoundError(f"data.yaml não encontrado em {dataset_path}")


def process_dataset(dataset_path, prefix):
    dataset_classes = load_classes(dataset_path)
    print(f"[{prefix}] classes originais: {dataset_classes}")

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            img_src = os.path.join(img_dir, img_file)
            lbl_src = os.path.join(lbl_dir, lbl_file)

            # Novo nome para evitar conflitos
            new_base = f"{prefix}_{split}_{base}"
            ext = os.path.splitext(img_file)[1]  # preserva extensão original (jpg, png, etc.)
            img_dst = os.path.join(OUTPUT_IMAGES, new_base + ext)
            lbl_dst = os.path.join(OUTPUT_LABELS, new_base + ".txt")

            # Copiar imagem
            shutil.copy(img_src, img_dst)

            # Processar label
            if os.path.exists(lbl_src):
                with open(lbl_src, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x, y, w, h = parts
                    cls_id = int(cls_id)

                    original_class = dataset_classes.get(cls_id)
                    if original_class not in CLASS_MAP:
                        continue
                    final_class = CLASS_MAP[original_class]
                    final_id = FINAL_CLASSES.index(final_class)

                    new_lines.append(f"{final_id} {x} {y} {w} {h}")

                with open(lbl_dst, "w") as f:
                    f.write("\n".join(new_lines))
            else:
                # Sem label = imagem "background" (sem objetos), cria .txt vazio
                open(lbl_dst, "w").close()


# Executa para cada dataset
for prefix, path in DATASETS.items():
    process_dataset(path, prefix)

# Criar data.yaml final
with open("datasets/final/data.yaml", "w") as f:
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("nc: {}\n".format(len(FINAL_CLASSES)))
    f.write("names: {}\n".format(FINAL_CLASSES))

print("Concluído! Dataset final em datasets/final/")