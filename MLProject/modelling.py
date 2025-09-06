
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import argparse

# Autolog aktif
mlflow.autolog()


def parse_args():
    parser = argparse.ArgumentParser()
    # Define the data_path argument matching the MLProject parameter
    # Update the default path to the correct location
    parser.add_argument("--data_path", type=str, default="Membangun_model/namadataset_preprocessing/heart_clean.csv", help="Path to the dataset file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Use the data_path from arguments
    data_path = args.data_path
    print(f"Loading data from: {data_path}") # Print the path being used

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return


    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="RandomForest_Autolog_Project"):
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")



    print("✅ Modelling.py script updated for MLflow Project.")


if __name__ == "__main__":
    main()
