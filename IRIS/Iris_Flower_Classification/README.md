# Iris Flower Classification Machine Learning Web Application (3,000 Records)

A complete production Machine Learning project built using **Scikit-Learn Random Forest Classifier (98.50% test accuracy)** on a realistic **3,000-record dataset**, Python **Flask**, **SQLAlchemy ORM**, **Waitress WSGI**, and a modern **Glassmorphism Web Dashboard Interface**.

---

## Features

- **Large Realistic Dataset**: Built and evaluated on a **3,000-record dataset** (`data/iris_3000.csv`) with 1,000 samples per species.
- **Machine Learning Model**: Scikit-Learn `RandomForestClassifier` (`n_estimators=100`, `max_depth=5`) achieving **98.50% accuracy** on 600 test samples.
- **Model Persistence**: Complete prediction pipeline (feature scaler, random forest model, species metadata) pickled and saved to `models/iris_model.pkl`.
- **Database Logging & Analytics**: SQLAlchemy ORM tracking all predictions in SQLite (`data/iris_predictions.db`) with analytics endpoints (`/api/analytics`, `/api/metrics`).
- **Production Server & Security**: Multi-threaded `Waitress` WSGI server, `Flask-Limiter` rate limits (60 req/min), and production HTTP security headers.
- **Interactive UI**: Glassmorphic dark dashboard with interactive sliders, quick-fill sample presets (Setosa, Versicolor, Virginica), real-time probability bar charts, and database log history.
- **Automated Testing Suite**: 12 Pytest test cases verifying model loadability, API responses, rate limits, security headers, database persistence, and validation bounds.

---

## Directory Structure

```text
IRIS/
├── Iris_Flower_Classification/   # Final Project Deliverables Folder for College Submission
│   ├── Source_Code/              # Flask Server, WSGI, DB, Model Trainer, UI Templates
│   ├── Dataset/                  # iris_3000.csv (3,000 rows) & iris.csv
│   ├── Google_Colab/             # Iris_Flower_Classification.ipynb notebook
│   ├── Screenshots/              # 7 High-resolution visual output PNGs
│   ├── requirements.txt          # Python dependency specifications
│   ├── PROJECT_EXPLANATION.md    # College Viva & Review Guide
│   └── README.md                 # Documentation
├── data/
│   └── iris_3000.csv             # 3,000-record dataset CSV file
├── models/
│   └── iris_model.pkl            # Serialized ML model pipeline artifact
├── app.py                        # Production Flask server & API routes
├── train_model.py                # Dataset generator, model trainer, & evaluation script
├── test_app.py                   # Automated 12-case Pytest test suite
├── wsgi.py                       # Production Waitress WSGI server entry point
├── Dockerfile                    # Multi-stage production Docker build
├── docker-compose.yml            # Docker Compose orchestration setup
└── requirements.txt              # Project dependencies
```

---

## Quick Start & Usage

### 1. Install Dependencies
Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Train Model & Save Artifact
To train the Random Forest model on `data/iris_3000.csv` (saving to `models/iris_model.pkl`), run:

```bash
python train_model.py
```

### 3. Launch Flask Server / WSGI Production Server
To run the production WSGI web application:

```bash
python wsgi.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

### 4. Run Automated Test Suite
Run the 12 automated unit and integration tests with Pytest:

```bash
pytest test_app.py
```

---

## Model Evaluation Metrics (3,000 Records)

| Metric | Value |
| :--- | :--- |
| **Model Type** | Random Forest Classifier (`n_estimators=100`, `max_depth=5`) |
| **Dataset Size** | **3,000 instances** (2,400 Train / 600 Test) |
| **Test Accuracy** | **98.50%** (591 / 600 correct test samples) |
| **Setosa F1-Score** | 1.00 |
| **Versicolor F1-Score** | 0.98 |
| **Virginica F1-Score** | 0.98 |
