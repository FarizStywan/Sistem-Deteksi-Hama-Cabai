import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os

# ==== 1. Load model ====
model_path = "models/model_cnn.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model CNN belum ditemukan. Jalankan cnn_train.py dulu.")

cnn_model = tf.keras.models.load_model(model_path)

# ==== 2. Siapkan data generator untuk evaluasi ====
data_dir = "data/cnn_dataset/test"  # atau "train" tergantung dataset kamu
img_size = (150, 150)
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0/255.0)

test_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ==== 3. Prediksi seluruh gambar test ====
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

y_pred_probs = cnn_model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==== 4. Evaluasi ====
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Akurasi CNN pada data test: {acc:.4f}")
print("\n📊 Laporan klasifikasi:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ==== 5. Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - CNN Deteksi Daun Cabai")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()
