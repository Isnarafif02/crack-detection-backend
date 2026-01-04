from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./crack_detection.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    detections = relationship("DetectionResult", back_populates="user", cascade="all, delete-orphan")

class DetectionResult(Base):
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Images (base64)
    original_image = Column(Text, nullable=False)
    normal_image = Column(Text, nullable=False)
    flipped_image = Column(Text, nullable=False)
    rotated_image = Column(Text, nullable=False)
    cropped_image = Column(Text, nullable=False)
    mask_image = Column(Text, nullable=False)
    
    # Metrics
    inference_time = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    mAP = Column(Float, nullable=False)
    crack_length = Column(Float, nullable=False)
    crack_area = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="detections")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)
