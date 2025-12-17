from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CameraSegment(Base):
    __tablename__ = "camera_segments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False)
    start = Column(String, nullable=False)  # ISO datetime string
    end = Column(String, nullable=False)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)

class BoxDetection(Base):
    __tablename__ = "box_detections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_path = Column(String, nullable=False)
    detection_data = Column(Text)  # JSON string or details

class LabelInterval(Base):
    __tablename__ = "label_intervals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False)
    label_name = Column(String, nullable=False)
    start = Column(String, nullable=False)
    end = Column(String, nullable=False)
