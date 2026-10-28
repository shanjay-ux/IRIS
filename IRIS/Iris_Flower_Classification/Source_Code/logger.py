import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(app, log_file_path=None):
    """Configure structured logging with file rotation and stdout streaming."""
    if not log_file_path:
        log_file_path = app.config.get("LOG_FILE", "logs/app.log")
        
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Set log level
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    app.logger.setLevel(log_level)

    # Prevent duplicate handlers
    if not app.logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s in %(filename)s:%(lineno)d] - %(message)s"
        )

        # 1. Rotating File Handler (Max 5MB per log file, keep 3 backup files)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        # 2. Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        app.logger.addHandler(file_handler)
        app.logger.addHandler(console_handler)

    app.logger.info(f"Logger initialized. Active log file: '{log_file_path}'")
