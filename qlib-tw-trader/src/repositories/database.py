from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'data.db'}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
    cursor.close()


class Base(DeclarativeBase):
    pass


def get_session() -> Session:
    """取得資料庫 Session"""
    return SessionLocal()


def init_db():
    """初始化資料庫，建立所有表和索引"""
    from src.repositories import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_training_factor_results_factor_id "
            "ON training_factor_results(factor_id)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_training_runs_status "
            "ON training_runs(status)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_training_runs_week_id "
            "ON training_runs(week_id)"
        ))
        conn.commit()
