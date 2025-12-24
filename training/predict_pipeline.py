# training/predict_pipeline.py
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image, UnidentifiedImageError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# 1️⃣ LOAD MODEL CNN DAN RF
# ===============================
cnn_model = tf.keras.models.load_model("models/model_cnn.h5")
rf_model = joblib.load("models/model_rf.pkl")

encoders = joblib.load("models/encoders_rf.pkl")
le_daun = encoders['le_daun']
le_suhu = encoders['le_suhu']
le_kelembapan = encoders['le_kelembapan']
le_hama = encoders['le_hama']

dataset = pd.read_csv("data/dataset_hama_cabai.csv")

# Normalisasi kolom text di dataset (untuk pencocokan yang stabil)
for col in ["daun", "suhu", "kelembapan", "hama"]:
    dataset[col] = dataset[col].astype(str).str.strip().str.lower()

# ===============================
# 2️⃣ FUNGSI UNTUK BACA & PREPROCESS GAMBAR
# ===============================
def load_and_preprocess_image(image_path, target_size=(150, 150), max_pixels=4000*4000):
    """
    - Resize & convert image safely using PIL sebelum dikirim ke keras.
    - Meminimalkan risiko OOM & handle file corrupt.
    """
    try:
        with Image.open(image_path) as im:
            im.verify()  # cek integritas dulu
        # reopen (verify() moves file pointer)
        with Image.open(image_path) as im:
            # convert to RGB
            im = im.convert("RGB")
            # resize keeping aspect ratio then fit to target_size to reduce memory
            im.thumbnail((max(target_size), max(target_size)))
            im = im.resize(target_size)
            arr = np.asarray(im).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            return arr
    except UnidentifiedImageError:
        logger.exception("Gagal membaca gambar: format tidak dikenali.")
        raise
    except Exception:
        logger.exception("Gagal preprocess gambar.")
        raise

# ===============================
# 3️⃣ FUNGSI UNTUK CNN PREDIKSI GAMBAR DAUN (aman)
# ===============================
def predict_leaf_label(image_path):
    try:
        img_array = load_and_preprocess_image(image_path, target_size=(150, 150))
        prediction = cnn_model.predict(img_array)
        class_index = int(np.argmax(prediction, axis=1)[0])
        # ambil class labels dari dataset (normalisasi sudah dilakukan di atas)
        class_labels = sorted(dataset["daun"].unique())
        if class_index < 0 or class_index >= len(class_labels):
            raise IndexError(f"class_index {class_index} out of range ({len(class_labels)})")
        daun_label = class_labels[class_index]
        return daun_label
    except Exception:
        logger.exception("Error saat prediksi daun dengan CNN.")
        raise

# ===============================
# 4️⃣ PREDIKSI RF + AMBIL MITIGASI (robust)
# ===============================
def predict_rf(daun_label, suhu_label, kelembapan_label):
    # normalisasi input text agar cocok dengan dataset/encoder
    daun_label_norm = str(daun_label).strip().lower()
    suhu_label_norm = str(suhu_label).strip().lower()
    kelembapan_label_norm = str(kelembapan_label).strip().lower()

    try:
        daun_enc = le_daun.transform([daun_label_norm])[0]
        suhu_enc = le_suhu.transform([suhu_label_norm])[0]
        kelembapan_enc = le_kelembapan.transform([kelembapan_label_norm])[0]
    except ValueError as e:
        logger.exception("Label text tidak dikenali oleh encoder: %s", e)
        raise

    input_data = pd.DataFrame({
        "daun": [daun_enc],
        "suhu": [suhu_enc],
        "kelembapan": [kelembapan_enc]
    })

    try:
        pred_encoded = rf_model.predict(input_data)
        pred_hama = le_hama.inverse_transform(pred_encoded)[0].strip().lower()
    except Exception:
        logger.exception("Error saat prediksi Random Forest.")
        raise

    # cari mitigasi spesifik (semua kolom sudah lower-case)
    match = dataset[
        (dataset["daun"] == daun_label_norm) &
        (dataset["suhu"] == suhu_label_norm) &
        (dataset["kelembapan"] == kelembapan_label_norm) &
        (dataset["hama"] == pred_hama)
    ]

    if not match.empty:
        mitigasi = match.iloc[0]["mitigasi"]
    else:
        alternatif = dataset[dataset["hama"] == pred_hama]
        if not alternatif.empty:
            mitigasi = alternatif.sample(1).iloc[0]["mitigasi"]
        else:
            mitigasi = "Mitigasi belum tersedia untuk kondisi ini."

    return pred_hama, mitigasi

# ===============================
# 5️⃣ FUNGSI UTAMA PIPELINE
# ===============================
def predict_pipeline(image_path, suhu_label, kelembapan_label):
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError("image_path tidak ditemukan atau sudah dihapus.")

    try:
        daun_label = predict_leaf_label(image_path)
        logger.info("Prediksi daun: %s", daun_label)
        hama, mitigasi = predict_rf(daun_label, suhu_label, kelembapan_label)
        logger.info("Prediksi hama: %s", hama)
    except Exception as e:
        logger.exception("Pipeline gagal: %s", e)
        raise

    return {
        "daun": daun_label,
        "hama": hama,
        "mitigasi": mitigasi
    }

# debug test
if __name__ == "__main__":
    image_path = "data/cnn_dataset/train/keriting/Curl Virus00061.jpg"
    print(predict_pipeline(image_path, "panas", "tinggi"))
