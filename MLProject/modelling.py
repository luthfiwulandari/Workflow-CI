# modelling.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Menonaktifkan autologging untuk menghindari konflik logging parameter
mlflow.autolog(disable=True) # <--- Autologging dinonaktifkan

# Path ke file dataset (sesuai dengan struktur di repository GitHub)
# Ini adalah path relatif dari direktori kerja MLProject
data_path = 'namadataset_preprocessing/heart_clean.csv' # Path disesuaikan

try:
    df = pd.read_csv(data_path)
    print(f"✅ File dataset berhasil dibaca dari {data_path}")
except FileNotFoundError:
    print(f"Error: File dataset tidak ditemukan di {data_path}")
    print("Pastikan file heart_clean.csv berada di folder 'namadataset_preprocessing/' di dalam folder MLProject di repository GitHub Anda.")
    exit(1) # Keluar dari skrip jika file tidak ditemukan

# Asumsi kolom target adalah 'target'
if 'target' not in df.columns:
    print("Error: Kolom 'target' tidak ditemukan di dataset.")
    exit(1)

X = df.drop('target', axis=1)
y = df['target']

# Split data - Dilakukan di luar with run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run(run_name="RandomForest_ExplicitLog"): # Mengganti nama run untuk membedakan
    print("MLflow run started.")

    # --- LOGGING PARAMETER EKSPLISIT DI DALAM with run ---
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

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
    # --- LOGGING METRIK EKSPLISIT DI DALAM with run ---
    mlflow.log_metric("accuracy", acc)
    print("✅ Evaluasi model selesai.")


    # --- LOGGING MODEL EKSPLISIT INI (di dalam with run) ---
    print("Logging model explicitly...")
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            # registered_model_name="RandomForestHeartDiseaseModel" # Opsional
        )
        print("✅ Model logging complete.")
    except Exception as e:
        print(f"Error during explicit model logging: {e}")
    # --- AKHIR LOGGING MODEL EKSPLISIT ---

print("✅ Modelling.py script finished execution.") # Pesan ini ada di luar with mlflow.start_run
