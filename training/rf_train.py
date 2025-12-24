import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels

# ==================================================
# 1. Pastikan folder models ada
# ==================================================
os.makedirs("models", exist_ok=True)

# ==================================================
# 2. Baca dataset
# ==================================================
data_path = "data/dataset_hama_cabai.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di path: {data_path}")

df = pd.read_csv(data_path)

# ==================================================
# 3. Validasi kolom
# ==================================================
required_cols = ['daun', 'suhu', 'kelembapan', 'hama']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.")

# ==================================================
# 4. Pisahkan fitur & target
# ==================================================
X = df[['daun', 'suhu', 'kelembapan']].copy()
y = df['hama']

# ==================================================
# 5. Label Encoding
# ==================================================
le_daun = LabelEncoder()
le_suhu = LabelEncoder()
le_kelembapan = LabelEncoder()
le_hama = LabelEncoder()

X['daun'] = le_daun.fit_transform(X['daun'])
X['suhu'] = le_suhu.fit_transform(X['suhu'])
X['kelembapan'] = le_kelembapan.fit_transform(X['kelembapan'])
y_encoded = le_hama.fit_transform(y)

# ==================================================
# 6. VISUALISASI:
#    OOB ERROR vs JUMLAH POHON
#    untuk berbagai max_features (SAMPAI 100 POHON)
# ==================================================
n_estimators_range = range(10, 101, 10)
max_features_list = ['sqrt', 'log2', None]

oob_errors = {
    'sqrt': [],
    'log2': [],
    'None': []
}

for mf in max_features_list:
    for n in n_estimators_range:
        rf_tmp = RandomForestClassifier(
            n_estimators=n,
            max_features=mf,
            oob_score=True,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        rf_tmp.fit(X, y_encoded)

        oob_pred = np.argmax(rf_tmp.oob_decision_function_, axis=1)
        oob_error = 1 - accuracy_score(y_encoded, oob_pred)
        oob_errors[str(mf)].append(oob_error)

# ----- Plot (mirip contoh kamu)
plt.figure(figsize=(8, 5))

plt.plot(
    n_estimators_range,
    oob_errors['sqrt'],
    linestyle='--',
    linewidth=3,
    marker='o',
    label="RandomForestClassifier, max_features='sqrt'"
)


plt.plot(
    n_estimators_range,
    oob_errors['log2'],
    label="RandomForestClassifier, max_features='log2'"
)

plt.plot(
    n_estimators_range,
    oob_errors['None'],
    label="RandomForestClassifier, max_features=None"
)

plt.xlabel('n_estimators')
plt.ylabel('OOB error rate')
plt.title('OOB Error vs Number of Trees (Random Forest)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==================================================
# 7. TRAIN MODEL FINAL (SESUAI HASIL ANALISIS)
# ==================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X, y_encoded)

# ==================================================
# 8. Evaluasi OOB (MODEL FINAL)
# ==================================================
oob_acc = rf_model.oob_score_
print(f"\n✅ OOB Accuracy (Model Final): {oob_acc:.4f}\n")

# ==================================================
# 9. Classification Report (OOB)
# ==================================================
oob_pred_final = np.argmax(
    rf_model.oob_decision_function_,
    axis=1
)

labels = unique_labels(y_encoded, oob_pred_final)
target_names = [le_hama.classes_[i] for i in labels]

print("📊 Laporan klasifikasi (OOB - Model Final):\n")
print(classification_report(
    y_encoded,
    oob_pred_final,
    labels=labels,
    target_names=target_names
))

# ==================================================
# 10. Cross Validation (pembanding)
# ==================================================
print("\n🔍 Cross-Validation (cv=5)...")
cv_scores = cross_val_score(rf_model, X, y_encoded, cv=5)

print(f"Akurasi per fold : {np.round(cv_scores, 4)}")
print(f"Rata-rata        : {cv_scores.mean():.4f}")
print(f"Standar deviasi  : {cv_scores.std():.4f}")

# ==================================================
# 11. VISUALISASI: Cross-Validation
# ==================================================
plt.figure(figsize=(7, 4))
plt.bar(range(1, len(cv_scores) + 1), cv_scores)
plt.axhline(
    cv_scores.mean(),
    linestyle='--',
    label=f"Mean = {cv_scores.mean():.3f}"
)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy per Fold')
plt.legend()
plt.tight_layout()
plt.show()

# ==================================================
# 12. VISUALISASI: Feature Importance
# ==================================================
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(6, 4))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance Random Forest')
plt.tight_layout()
plt.show()

# ==================================================
# 13. Simpan model & encoder
#    (NAMA SAMA DENGAN KODE AWAL)
# ==================================================
joblib.dump(rf_model, "models/model_rf.pkl")
joblib.dump({
    'le_daun': le_daun,
    'le_suhu': le_suhu,
    'le_kelembapan': le_kelembapan,
    'le_hama': le_hama
}, "models/encoders_rf.pkl")

print("\n💾 Model & encoder berhasil disimpan.")
