# modelling.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Autolog aktif (opsional jika menggunakan log_model eksplisit, tapi bisa dibiarkan)
mlflow.autolog()

# Path ke file dataset (sesuai dengan struktur di repository GitHub)
# Ini adalah path relatif dari direktori kerja MLProject
data_path = 'namadataset_preprocessing/heart_clean.csv' # <--- Path disesuaikan dengan struktur di dalam MLProject

try:
    # Menggunakan mlflow.log_param untuk mencatat path data
    mlflow.log_param("data_path", data_path)
    df = pd.read_csv(data_path)
    print(f"✅ File dataset berhasil dibaca dari {data_path}")
except FileNotFoundError:
    print(f"Error: File dataset tidak ditemukan di {data_path}")
    print("Pastikan file heart_clean.csv berada di folder 'namadataset_preprocessing/' di dalam folder MLProject di repository GitHub Anda.")
    # Keluar dari skrip jika file tidak ditemukan
    exit(1) # Keluar dengan kode status non-nol untuk menandakan kegagalan

# Asumsi kolom target adalah 'target'
if 'target' not in df.columns:
    print("Error: Kolom 'target' tidak ditemukan di dataset.")
    exit(1)

X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log parameter split data
mlflow.log_param("test_size", 0.2)
mlflow.log_param("random_state", 42)

with mlflow.start_run(run_name="RandomForest_Autolog"): # Pastikan run name ini unik atau diatur dinamis jika banyak run
    print("MLflow run started.")

    # Definisikan model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Log parameter model
    mlflow.log_param("n_estimators", 100)

    # Train model
    print("Memulai pelatihan model...")
    model.fit(X_train, y_train)
    print("✅ Pelatihan model selesai.")

    # Evaluate
    print("Memulai evaluasi model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    # Log metrik akurasi
    mlflow.log_metric("accuracy", acc)
    print("✅ Evaluasi model selesai.")


    # --- TAMBAHKAN BAGIAN LOGGING MODEL EKSPLISIT INI ---
    # Ini akan memastikan model dicatat sebagai artefak
    print("Logging model explicitly...")
    try:
        mlflow.sklearn.log_model(
            sk_model=model,          # Objek model scikit-learn Anda
            artifact_path="model",   # Nama folder artefak di dalam run (akan jadi .../run_id/artifacts/model/)
            # registered_model_name="RandomForestHeartDiseaseModel" # Opsional: Nama model terdaftar di MLflow Registry
        )
        print("✅ Model logging complete.")
    except Exception as e:
        print(f"Error during explicit model logging: {e}")

    # --- AKHIR BAGIAN LOGGING MODEL EKSPLISIT ---

print("✅ Modelling.py script finished execution.") # Pesan ini ada di luar with mlflow.start_run
