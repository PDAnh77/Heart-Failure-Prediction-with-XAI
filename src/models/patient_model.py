from database import Base
from sqlalchemy import Column, Integer, String

class Patient(Base):
    __tablename__ = "patient_info"
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
