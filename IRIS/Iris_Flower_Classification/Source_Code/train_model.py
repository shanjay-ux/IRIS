import os
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_model():
    print("=" * 60)
    print("      IRIS FLOWER CLASSIFICATION - MODEL TRAINING      ")
    print("=" * 60)

    # 1. Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    csv_3000_path = os.path.join("data", "iris_3000.csv")
    csv_150_path = os.path.join("data", "iris.csv")

    # Load 3,000 record dataset if available, otherwise load 150 record dataset or sklearn load_iris
    if os.path.exists(csv_3000_path):
        print(f"[OK] Loading 3,000-record dataset from '{csv_3000_path}'")
        df = pd.read_csv(csv_3000_path)
    elif os.path.exists(csv_150_path):
        print(f"[OK] Loading dataset from '{csv_150_path}'")
        df = pd.read_csv(csv_150_path)
    else:
        print("[OK] Generating dataset from Scikit-Learn load_iris...")
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
        target_names = list(iris.target_names)
        df["species"] = df["target"].map(lambda idx: target_names[idx])
        df.to_csv(csv_150_path, index=False)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]
    target_names = sorted(list(y.unique()))

    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[OK] Dataset shape: {df.shape} ({len(df)} total rows)")
    print(f"[OK] Data split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)

    # Model Evaluation
    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    acc_percentage = float(acc * 100)

    print("-" * 60)
    print(f"Model Accuracy on Test Set: {acc_percentage:.2f}% ({sum(y_pred == y_test)}/{len(y_test)} correct)")
    print("-" * 60)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Package Model & Artifacts for Persistence
    model_payload = {
        "model": rf_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "target_names": target_names,
        "accuracy": float(acc_percentage),
        "species_metadata": {
            "setosa": {
                "display_name": "Iris Setosa",
                "description": "Known for small petals and distinct sepal size. Found in arctic climates.",
                "color": "#4f46e5"
            },
            "versicolor": {
                "display_name": "Iris Versicolor",
                "description": "Medium-sized iris with blue-purple blossoms. Native to eastern North America.",
                "color": "#06b6d4"
            },
            "virginica": {
                "display_name": "Iris Virginica",
                "description": "Largest of the three species with vibrant violet petals. Flourishes in wetlands.",
                "color": "#10b981"
            }
        }
    }

    model_path = os.path.join("models", "iris_model.pkl")
    joblib.dump(model_payload, model_path)
    print(f"[OK] Model artifact successfully serialized to '{model_path}'")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save_model()
