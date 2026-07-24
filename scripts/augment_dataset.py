import os
import cv2
import albumentations as A
from collections import Counter

# Classes finais
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles", "person"]

# Classes minoritárias que queremos aumentar
TARGET_CLASSES = {"gloves", "goggles", "person"}

LABELS_DIR = "datasets/final/labels"
IMAGES_DIR = "datasets/final/images"
OUTPUT_IMAGES = "datasets/augmented/images"
OUTPUT_LABELS = "datasets/augmented/labels"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# Transformações de augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
])

def augment_image(img_path, lbl_path, base_name):
    image = cv2.imread(img_path)
    with open(lbl_path, "r") as f:
        lines = f.readlines()

    # Verifica se contém alguma classe alvo
    has_target = any(CLASSES[int(line.split()[0])] in TARGET_CLASSES for line in lines)
    if not has_target:
        return

    # Aplica transformação
    augmented = transform(image=image)
    aug_img = augmented["image"]

    # Novo nome
    new_base = base_name + "_aug"
    ext = os.path.splitext(img_path)[1]
    img_out = os.path.join(OUTPUT_IMAGES, new_base + ext)
    lbl_out = os.path.join(OUTPUT_LABELS, new_base + ".txt")

    # Salva imagem
    cv2.imwrite(img_out, aug_img)

    # Copia labels sem alteração (bounding boxes continuam válidas)
    with open(lbl_out, "w") as f:
        f.writelines(lines)

def main():
    for lbl_file in os.listdir(LABELS_DIR):
        base = os.path.splitext(lbl_file)[0]
        lbl_path = os.path.join(LABELS_DIR, lbl_file)
        img_path = os.path.join(IMAGES_DIR, base + ".jpg")
        if not os.path.exists(img_path):
            continue
        augment_image(img_path, lbl_path, base)

if __name__ == "__main__":
    main()
    print("✅ Augmentation concluída! Imagens salvas em datasets/augmented/")
