from sqlalchemy import Column, Integer, String
from utils.sqlite_utils import Base

class ModelManagement(Base):
    __tablename__ = 'model_management'
    ID = Column(Integer, primary_key=True)
    NAME = Column(String)
    PATH = Column(String)