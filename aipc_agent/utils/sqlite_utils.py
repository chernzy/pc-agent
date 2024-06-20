from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

class SqliteSqlalchemy(object):
    def __init__(self):
        engine = create_engine("sqlite:///./sqlite.db", echo=True)
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()