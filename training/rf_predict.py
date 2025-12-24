import joblib
import pandas as pd
import os

# ==== 1. Path model & dataset ====
model_path = "models/model_rf.pkl"
encoder_path = "models/encoders_rf.pkl"
data_path = "data/dataset_hama_cabai.csv"

# ==== 2. Pastikan semua file ada ====
for path in [model_path, encoder_path, data_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {path}")

# ==== 3. Load model & encoder ====
rf_model = joblib.load(model_path)
encoders = joblib.load(encoder_path)
le_daun = encoders['le_daun']
le_suhu = encoders['le_suhu']
le_kelembapan = encoders['le_kelembapan']
le_hama = encoders['le_hama']

# ==== 4. Load dataset untuk ambil mitigasi ====
df = pd.read_csv(data_path)

# ==== 5. Fungsi prediksi ====
def prediksi_hama(daun, suhu, kelembapan):
    # Ubah teks input menjadi angka via encoder
    try:
        daun_encoded = le_daun.transform([daun])[0]
        suhu_encoded = le_suhu.transform([suhu])[0]
        kelembapan_encoded = le_kelembapan.transform([kelembapan])[0]
    except ValueError as e:
        return f"❌ Input tidak dikenali: {e}"

    # Buat dataframe input
    input_df = pd.DataFrame([[daun_encoded, suhu_encoded, kelembapan_encoded]],
                            columns=['daun', 'suhu', 'kelembapan'])

    # Prediksi hama
    pred_hama_encoded = rf_model.predict(input_df)[0]
    pred_hama = le_hama.inverse_transform([pred_hama_encoded])[0]

    # Ambil mitigasi yang cocok dari dataset
    mitigasi_row = df[
        (df['daun'] == daun) &
        (df['suhu'] == suhu) &
        (df['kelembapan'] == kelembapan) &
        (df['hama'] == pred_hama)
    ]

    if not mitigasi_row.empty:
        mitigasi = mitigasi_row.iloc[0]['mitigasi']
    else:
        mitigasi = "Mitigasi spesifik tidak ditemukan di dataset."

    return {
        "daun": daun,
        "suhu": suhu,
        "kelembapan": kelembapan,
        "prediksi_hama": pred_hama,
        "mitigasi": mitigasi
    }

# ==== 6. Contoh penggunaan ====
if __name__ == "__main__":
    # contoh hasil CNN
    daun_cnn = "keriting"  
    suhu_input = "panas"
    kelembapan_input = "tinggi"

    hasil = prediksi_hama(daun_cnn, suhu_input, kelembapan_input)
    print("\n=== HASIL PREDIKSI RANDOM FOREST ===")
    for k, v in hasil.items():
        print(f"{k.capitalize():<15}: {v}")
