import os
from collections import Counter

LABELS_DIR = "datasets/final/labels"
CLASSES = ["helmet", "vest", "gloves", "boots", "goggles", "person"]

def audit_labels():
    counts = Counter()
    empty_files = 0

    for lbl_file in os.listdir(LABELS_DIR):
        with open(os.path.join(LABELS_DIR, lbl_file), "r") as f:
            lines = f.readlines()
        if not lines:
            empty_files += 1
            continue
        for line in lines:
            cls_id = int(line.split()[0])
            counts[CLASSES[cls_id]] += 1

    print("Auditoria do dataset final:")
    for cls in CLASSES:
        print(f"- {cls}: {counts[cls]} exemplos")
    print(f"\nImagens sem anotação: {empty_files}")

if __name__ == "__main__":
    audit_labels()
