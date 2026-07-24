import os
from collections import Counter

# Pastas a serem auditadas
LABELS_DIRS = ["datasets/final/labels", "datasets/augmented/labels"]
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles", "person"]

def audit_labels():
    counts = Counter()
    empty_files = 0

    for lbl_dir in LABELS_DIRS:
        if not os.path.exists(lbl_dir):
            continue
        for lbl_file in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                lines = f.readlines()
            if not lines:
                empty_files += 1
                continue
            for line in lines:
                cls_id = int(line.split()[0])
                counts[CLASSES[cls_id]] += 1

    print("📊 Auditoria combinada (final + augmented):")
    for cls in CLASSES:
        print(f"- {cls}: {counts[cls]} exemplos")
    print(f"\nImagens sem anotação: {empty_files}")

if __name__ == "__main__":
    audit_labels()
