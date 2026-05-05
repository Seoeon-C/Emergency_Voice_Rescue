import uuid
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

_cur = Path(__file__).resolve().parent
load_dotenv(_cur / ".env")

# SQLite (기본/개발용) or PostgreSQL (운영: DATABASE_URL 환경변수 설정)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_cur}/zones.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

ZONE_LABELS = ["산", "공사장", "저수지", "강", "논"]


class Zone(Base):
    __tablename__ = "zones"

    id         = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    name       = Column(String,   nullable=False)
    label      = Column(String,   nullable=True)   # 산/공사장/저수지/강/논
    coord      = Column(String,   nullable=True)   # "37.5665° N, 126.9780° E"
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
