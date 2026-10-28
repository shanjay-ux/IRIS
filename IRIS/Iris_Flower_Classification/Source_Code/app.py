import os
import time
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import get_config
from logger import setup_logger
from database import db
from train_model import train_and_save_model

# App start time for uptime calculations
START_TIME = time.time()

# Feature input constraints for validation
FEATURE_LIMITS = {
    "sepal_length": {"min": 4.0, "max": 8.0, "default": 5.1, "step": 0.1, "label": "Sepal Length (cm)"},
    "sepal_width":  {"min": 2.0, "max": 4.5, "default": 3.5, "step": 0.1, "label": "Sepal Width (cm)"},
    "petal_length": {"min": 1.0, "max": 7.0, "default": 1.4, "step": 0.1, "label": "Petal Length (cm)"},
    "petal_width":  {"min": 0.1, "max": 2.5, "default": 0.2, "step": 0.1, "label": "Petal Width (cm)"}
}

PRESETS = {
    "setosa": {
        "name": "Iris Setosa Sample",
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2
    },
    "versicolor": {
        "name": "Iris Versicolor Sample",
        "sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.0, "petal_width": 1.2
    },
    "virginica": {
        "name": "Iris Virginica Sample",
        "sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0
    }
}

def load_or_train_model(model_path):
    """Load pickled model payload or trigger training automatically if missing."""
    if not os.path.exists(model_path):
        print(f"[!] Model artifact missing at '{model_path}'. Running training pipeline...")
        train_and_save_model()
    return joblib.load(model_path)

def create_app(config_name=None):
    """Application factory for Flask web server."""
    app = Flask(__name__)
    config_obj = get_config(config_name)
    app.config.from_object(config_obj)

    # Initialize structured logger
    setup_logger(app)
    app.logger.info(f"Starting Iris ML Web App in mode: '{config_obj.__name__}'")

    # Initialize Database
    db.init_db(app.config["SQLALCHEMY_DATABASE_URI"])
    app.logger.info("Database initialized successfully.")

    # Initialize Rate Limiter
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[app.config.get("RATELIMIT_DEFAULT", "100 per minute")],
        storage_uri=app.config.get("RATELIMIT_STORAGE_URI", "memory://")
    )

    # Load Model Artifact
    model_path = app.config["MODEL_PATH"]
    model_payload = load_or_train_model(model_path)
    
    app.model = model_payload["model"]
    app.scaler = model_payload["scaler"]
    app.feature_names = model_payload["feature_names"]
    app.target_names = model_payload["target_names"]
    app.model_accuracy = model_payload["accuracy"]
    app.species_metadata = model_payload.get("species_metadata", {})

    # Security Response Headers Middleware
    @app.after_request
    def apply_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self' 'unsafe-inline';"
        return response

    # ----------------------------------------------------
    # ROUTES
    # ----------------------------------------------------

    @app.route("/")
    def index():
        """Render main interactive dashboard."""
        analytics = db.get_analytics_summary()
        return render_template(
            "index.html",
            accuracy=app.model_accuracy,
            feature_limits=FEATURE_LIMITS,
            presets=PRESETS,
            species_info=app.species_metadata,
            analytics=analytics
        )

    @app.route("/predict", methods=["POST"])
    @limiter.limit("60 per minute")
    def predict():
        """API endpoint for receiving measurements and producing species classification."""
        try:
            data = request.get_json() if request.is_json else request.form.to_dict()

            input_values = []
            errors = {}

            for feature in app.feature_names:
                if feature not in data or data[feature] is None or str(data[feature]).strip() == "":
                    errors[feature] = f"Missing value for {feature}"
                    continue

                try:
                    val = float(data[feature])
                    limits = FEATURE_LIMITS[feature]
                    if val < limits["min"] or val > limits["max"]:
                        errors[feature] = f"Must be between {limits['min']} and {limits['max']} cm"
                    else:
                        input_values.append(val)
                except (ValueError, TypeError):
                    errors[feature] = "Must be a valid numerical number"

            if errors:
                app.logger.warning(f"Validation failed for client {request.remote_addr}: {errors}")
                return jsonify({
                    "success": False,
                    "error": "Validation failed",
                    "field_errors": errors
                }), 400

            # Transform input attributes
            features_df = pd.DataFrame([input_values], columns=app.feature_names)
            scaled_features = app.scaler.transform(features_df)

            # Predict probabilities
            prediction_idx = app.model.predict(scaled_features)[0]
            probabilities = app.model.predict_proba(scaled_features)[0]

            predicted_species = str(prediction_idx)
            probs_dict = {
                app.target_names[i]: float(round(probabilities[i] * 100, 2))
                for i in range(len(app.target_names))
            }
            top_confidence = float(round(max(probabilities) * 100, 2))
            meta = app.species_metadata.get(predicted_species.lower(), {})

            # Log prediction to Database
            client_ip = request.remote_addr
            log_record = db.log_prediction(
                sepal_length=input_values[0],
                sepal_width=input_values[1],
                petal_length=input_values[2],
                petal_width=input_values[3],
                predicted_species=predicted_species,
                confidence=top_confidence,
                client_ip=client_ip
            )

            app.logger.info(
                f"Prediction success: {predicted_species} ({top_confidence}%) for IP: {client_ip}"
            )

            return jsonify({
                "success": True,
                "predicted_species": predicted_species,
                "display_name": meta.get("display_name", f"Iris {predicted_species.title()}"),
                "confidence": top_confidence,
                "probabilities": probs_dict,
                "species_metadata": meta,
                "log_id": log_record["id"],
                "inputs": {
                    "sepal_length": input_values[0],
                    "sepal_width": input_values[1],
                    "petal_length": input_values[2],
                    "petal_width": input_values[3]
                }
            }), 200

        except Exception as e:
            app.logger.error(f"Prediction exception: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"Internal prediction error: {str(e)}"
            }), 500

    @app.route("/api/analytics", methods=["GET"])
    def get_analytics():
        """Return aggregate prediction analytics and recent history."""
        summary = db.get_analytics_summary()
        recent_history = db.get_recent_history(limit=10)
        return jsonify({
            "success": True,
            "analytics": summary,
            "history": recent_history
        }), 200

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint for liveness & readiness probes."""
        uptime_seconds = round(time.time() - START_TIME, 2)
        return jsonify({
            "status": "healthy",
            "uptime_seconds": uptime_seconds,
            "model_loaded": app.model is not None,
            "model_type": "RandomForestClassifier",
            "accuracy": f"{app.model_accuracy:.2f}%",
            "features": app.feature_names,
            "classes": app.target_names
        }), 200

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """Production metrics endpoint."""
        summary = db.get_analytics_summary()
        return jsonify({
            "app_name": "iris-ml-classifier",
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "model_accuracy": app.model_accuracy,
            "total_inferences": summary["total_inferences"],
            "average_confidence": summary["average_confidence"],
            "species_distribution": summary["species_distribution"]
        }), 200

    # ----------------------------------------------------
    # CUSTOM ERROR HANDLERS
    # ----------------------------------------------------

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Endpoint not found"}), 404
        return render_template("index.html", error_message="Page Not Found"), 404

    @app.errorhandler(429)
    def rate_limit_error(error):
        app.logger.warning(f"Rate limit exceeded by IP: {request.remote_addr}")
        return jsonify({
            "success": False,
            "error": "Rate limit exceeded. Please slow down your requests (Max 60 per minute)."
        }), 429

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal server error: {str(error)}")
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"success": False, "error": "Internal server error"}), 500
        return render_template("index.html", error_message="Internal Server Error"), 500

    return app

# Default instance for development runner
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
