import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

Base = declarative_base()

class PredictionLog(Base):
    """Database ORM model for storing prediction logs."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_species = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    client_ip = Column(String(50), nullable=True)

    def to_dict(self):
        """Serialize log record to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sepal_length": self.sepal_length,
            "sepal_width": self.sepal_width,
            "petal_length": self.petal_length,
            "petal_width": self.petal_width,
            "predicted_species": self.predicted_species,
            "confidence": self.confidence,
            "client_ip": self.client_ip
        }

class Database:
    """Database connection and session manager."""
    def __init__(self, db_uri=None):
        self.engine = None
        self.Session = None
        if db_uri:
            self.init_db(db_uri)

    def init_db(self, db_uri):
        """Initialize database engine and create tables if they do not exist."""
        # Ensure SQLite data directory exists if using local file URI
        if db_uri.startswith("sqlite:///") and ":memory:" not in db_uri:
            db_path = db_uri.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.engine = create_engine(db_uri, connect_args={"check_same_thread": False} if "sqlite" in db_uri else {})
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

    def get_session(self):
        """Get thread-scoped database session."""
        if not self.Session:
            raise RuntimeError("Database not initialized. Call init_db first.")
        return self.Session()

    def log_prediction(self, sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence, client_ip=None):
        """Record a single prediction entry to database."""
        session = self.get_session()
        try:
            log_entry = PredictionLog(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                predicted_species=predicted_species,
                confidence=confidence,
                client_ip=client_ip
            )
            session.add(log_entry)
            session.commit()
            return log_entry.to_dict()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_recent_history(self, limit=10):
        """Fetch latest prediction logs."""
        session = self.get_session()
        try:
            logs = session.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
            return [log.to_dict() for log in logs]
        finally:
            session.close()

    def get_analytics_summary(self):
        """Compute aggregate prediction analytics."""
        session = self.get_session()
        try:
            total_count = session.query(func.count(PredictionLog.id)).scalar() or 0
            avg_confidence = session.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
            
            # Group by species
            species_counts = dict(
                session.query(
                    PredictionLog.predicted_species,
                    func.count(PredictionLog.id)
                ).group_by(PredictionLog.predicted_species).all()
            )

            # Determine most common species
            most_common = max(species_counts, key=species_counts.get) if species_counts else "None"

            return {
                "total_inferences": total_count,
                "average_confidence": round(float(avg_confidence), 2),
                "species_distribution": species_counts,
                "most_common_species": most_common
            }
        finally:
            session.close()

db = Database()
