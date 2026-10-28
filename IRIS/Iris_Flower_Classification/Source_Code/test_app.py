import os
import pytest
import joblib
from app import create_app
from config import TestingConfig

@pytest.fixture
def app_instance():
    """Create Flask application fixture configured for testing."""
    app = create_app("testing")
    yield app

@pytest.fixture
def client(app_instance):
    """Create Flask test client fixture."""
    return app_instance.test_client()

def test_model_file_exists(app_instance):
    """Ensure serialized model artifact exists and accuracy is >= 95%."""
    model_path = app_instance.config["MODEL_PATH"]
    assert os.path.exists(model_path), f"Model file '{model_path}' should exist."
    payload = joblib.load(model_path)
    assert "model" in payload
    assert "scaler" in payload
    assert payload["accuracy"] >= 95.0

def test_health_endpoint(client):
    """Test /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data

def test_metrics_endpoint(client):
    """Test /api/metrics endpoint."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["app_name"] == "iris-ml-classifier"
    assert "total_inferences" in data
    assert "species_distribution" in data

def test_analytics_endpoint(client):
    """Test /api/analytics endpoint."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "analytics" in data
    assert "history" in data

def test_index_page(client):
    """Test home dashboard HTML rendering."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Iris Flower Classifier" in response.data

def test_predict_setosa_db_logging(client):
    """Test Setosa prediction and database logging."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predicted_species"] == "setosa"
    assert data["confidence"] > 80.0
    assert "log_id" in data

def test_predict_versicolor(client):
    """Test Versicolor prediction."""
    payload = {
        "sepal_length": 6.0,
        "sepal_width": 2.7,
        "petal_length": 4.0,
        "petal_width": 1.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predicted_species"] == "versicolor"

def test_predict_virginica(client):
    """Test Virginica prediction."""
    payload = {
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.5,
        "petal_width": 2.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predicted_species"] == "virginica"

def test_predict_missing_feature(client):
    """Test missing feature validation error."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "field_errors" in data

def test_predict_out_of_bounds(client):
    """Test out-of-bounds feature validation error."""
    payload = {
        "sepal_length": 99.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False

def test_security_headers(client):
    """Test production security headers."""
    response = client.get("/")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in response.headers

def test_404_error_handler(client):
    """Test API 404 error handler."""
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False
    assert data["error"] == "Endpoint not found"
