import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt

# ==== 1. Load model dan encoder ====
model_path = "models/model_rf.pkl"
encoder_path = "models/encoders_rf.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ Model atau encoder belum dilatih. Jalankan rf_train.py dulu.")

rf_model = joblib.load(model_path)
encoders = joblib.load(encoder_path)

le_daun = encoders['le_daun']
le_suhu = encoders['le_suhu']
le_kelembapan = encoders['le_kelembapan']
le_hama = encoders['le_hama']

# ==== 2. Baca dataset ====
data_path = "data/dataset_hama_cabai.csv"
df = pd.read_csv(data_path)

# ==== 3. Pisahkan fitur & target ====
X = df[['daun', 'suhu', 'kelembapan']].copy()
y = df['hama']

# ==== 4. Encode data sesuai encoder yang sudah disimpan ====
X['daun'] = le_daun.transform(X['daun'])
X['suhu'] = le_suhu.transform(X['suhu'])
X['kelembapan'] = le_kelembapan.transform(X['kelembapan'])
y_encoded = le_hama.transform(y)

# ==== 5. Prediksi menggunakan model yang sudah dilatih ====
y_pred = rf_model.predict(X)

# ==== 6. Evaluasi ====
acc = accuracy_score(y_encoded, y_pred)
print(f"\n✅ Akurasi keseluruhan (dataset penuh): {acc:.4f}")
print("\n📊 Laporan klasifikasi:\n")
print(classification_report(y_encoded, y_pred, target_names=le_hama.classes_))

# ==== 7. Confusion Matrix ====
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_hama.classes_)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest Hama Cabai")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()
