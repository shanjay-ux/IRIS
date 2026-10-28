import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_realistic_iris_3000():
    print("=" * 65)
    print("   GENERATING 3,000 RECORD REALISTIC IRIS DATASET   ")
    print("=" * 65)

    np.random.seed(42)

    # Statistical parameters derived from authentic Iris dataset (Mean, Covariance)
    species_params = {
        "setosa": {
            "mean": [5.006, 3.428, 1.462, 0.246],
            "cov": [
                [0.124, 0.099, 0.016, 0.010],
                [0.099, 0.144, 0.012, 0.009],
                [0.016, 0.012, 0.030, 0.006],
                [0.010, 0.009, 0.006, 0.011]
            ]
        },
        "versicolor": {
            "mean": [5.936, 2.770, 4.260, 1.326],
            "cov": [
                [0.266, 0.085, 0.182, 0.056],
                [0.085, 0.098, 0.083, 0.041],
                [0.182, 0.083, 0.221, 0.073],
                [0.056, 0.041, 0.073, 0.039]
            ]
        },
        "virginica": {
            "mean": [6.588, 2.974, 5.552, 2.026],
            "cov": [
                [0.404, 0.094, 0.303, 0.049],
                [0.094, 0.104, 0.071, 0.048],
                [0.303, 0.071, 0.305, 0.048],
                [0.049, 0.048, 0.048, 0.075]
            ]
        }
    }

    target_records_per_species = 1000
    rows = []

    for species_name, params in species_params.items():
        generated_count = 0
        seen_combos = set()
        
        while generated_count < target_records_per_species:
            # Sample from Multivariate Normal Distribution
            sample = np.random.multivariate_normal(params["mean"], params["cov"])
            
            # Add small realistic micro-variations
            sl = round(float(np.clip(sample[0], 4.0, 8.0)), 2)
            sw = round(float(np.clip(sample[1], 2.0, 4.5)), 2)
            pl = round(float(np.clip(sample[2], 1.0, 7.0)), 2)
            pw = round(float(np.clip(sample[3], 0.1, 2.5)), 2)

            combo = (sl, sw, pl, pw)

            # Ensure uniqueness
            if combo not in seen_combos:
                seen_combos.add(combo)
                rows.append({
                    "sepal_length": sl,
                    "sepal_width": sw,
                    "petal_length": pl,
                    "petal_width": pw,
                    "species": species_name
                })
                generated_count += 1

    df = pd.DataFrame(rows)

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Verification checks
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_count = df.duplicated(subset=["sepal_length", "sepal_width", "petal_length", "petal_width"]).sum()
    null_count = df.isna().sum().sum()

    print(f"[OK] Generated dataset shape: {df.shape}")
    print(f"[OK] Total Data Rows (excluding header): {total_rows}")
    print(f"[OK] Total Columns: {total_cols} ({list(df.columns)})")
    print(f"[OK] Duplicate Records: {duplicate_count}")
    print(f"[OK] Missing Values: {null_count}")

    # Class breakdown
    print("\nClass Breakdown:")
    print(df["species"].value_counts())

    # Save to disk
    paths_to_save = [
        os.path.join("data", "iris_3000.csv"),
        os.path.join("Iris_Flower_Classification", "Dataset", "iris_3000.csv")
    ]

    for p in paths_to_save:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[OK] Saved 3,000 row CSV file to: '{p}'")

    # -------------------------------------------------------------
    # ML Validation Training on 3,000 Records
    # -------------------------------------------------------------
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100

    print("-" * 65)
    print(f"[OK] ML Training Validation on 3,000 samples:")
    print(f"    Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"    Test Accuracy: {acc:.2f}%")
    print("-" * 65)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("=" * 65)

if __name__ == "__main__":
    generate_realistic_iris_3000()
