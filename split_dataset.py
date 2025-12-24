import os
import shutil
import random

# === KONFIGURASI ===
base_dir = "data/cnn_raw"       # folder sumber gambar mentah
output_dir = "data/cnn_dataset" # folder hasil pembagian
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

# Buat folder output jika belum ada
for split in ["train", "val", "test"]:
    for label in os.listdir(base_dir):
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

# === PEMBAGIAN DATASET ===
for label in os.listdir(base_dir):
    label_path = os.path.join(base_dir, label)
    if not os.path.isdir(label_path):
        continue

    images = [img for img in os.listdir(label_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratio["train"] * total)
    val_end = train_end + int(split_ratio["val"] * total)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Salin file ke folder tujuan
    for file_list, split in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        for img in file_list:
            src = os.path.join(label_path, img)
            dst = os.path.join(output_dir, split, label, img)
            shutil.copy2(src, dst)

print("✅ Dataset berhasil dibagi menjadi train/val/test (70/15/15)\n")

# === CEK JUMLAH GAMBAR ===
def count_images(folder):
    count_dict = {}
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        if os.path.isdir(path):
            count_dict[label] = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count_dict

print("📊 Jumlah gambar setelah pembagian:")
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split)
    counts = count_images(split_path)
    total_split = sum(counts.values())
    print(f"\n📁 {split.upper()} (total {total_split} gambar):")
    for label, count in counts.items():
        print(f"   - {label}: {count}")
