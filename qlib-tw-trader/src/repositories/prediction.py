"""每日預測記錄存取"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import DailyPrediction


class PredictionRepository:
    """每日預測 CRUD"""

    def __init__(self, session: Session):
        self._session = session

    def get_by_date(self, trade_date: date) -> DailyPrediction | None:
        """依交易日期取得預測"""
        stmt = select(DailyPrediction).where(
            DailyPrediction.trade_date == trade_date
        )
        return self._session.execute(stmt).scalar()

    def list_recent(self, limit: int = 30) -> list[DailyPrediction]:
        """取得最近的預測記錄（按日期降序）"""
        stmt = (
            select(DailyPrediction)
            .order_by(DailyPrediction.trade_date.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def create(self, prediction: DailyPrediction) -> DailyPrediction:
        """建立預測記錄"""
        self._session.add(prediction)
        self._session.commit()
        self._session.refresh(prediction)
        return prediction
