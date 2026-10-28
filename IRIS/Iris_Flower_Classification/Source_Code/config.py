import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-12345")
    DEBUG = False
    TESTING = False
    
    # Model Artifact Path
    MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.pkl")
    
    # Database Configuration
    DATABASE_PATH = os.path.join(BASE_DIR, "data", "iris_predictions.db")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI", f"sqlite:///{DATABASE_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Log Configuration
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    LOG_FILE = os.path.join(LOG_DIR, "app.log")
    
    # Rate Limiter Configuration
    RATELIMIT_DEFAULT = "100 per minute"
    RATELIMIT_STORAGE_URI = "memory://"

class DevelopmentConfig(Config):
    """Development environment settings."""
    DEBUG = True

class TestingConfig(Config):
    """Testing environment settings."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

class ProductionConfig(Config):
    """Production environment settings."""
    DEBUG = False
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-production-key-iris-ml-2026")

# Environment map
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": ProductionConfig
}

def get_config(env_name=None):
    """Retrieve config class based on FLASK_ENV or argument."""
    if not env_name:
        env_name = os.getenv("FLASK_ENV", "production").lower()
    return config_by_name.get(env_name, ProductionConfig)
