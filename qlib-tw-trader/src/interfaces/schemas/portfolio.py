from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """預測請求"""

    model_id: int
    top_k: int = 10
    trade_date: date | None = None  # 預計交易日期，None = 使用最新資料日期的下一天


class PredictionSignal(BaseModel):
    """預測訊號"""

    rank: int
    symbol: str
    name: str | None
    score: float


class PredictionsResponse(BaseModel):
    """預測回應"""

    trade_date: str  # 預計交易日期
    feature_date: str  # 實際使用的特徵資料日期
    model_name: str
    signals: list[PredictionSignal]


class TodayPredictionDetail(BaseModel):
    """今日預測詳情"""

    trade_date: str
    feature_date: str
    model_name: str
    model_week: str
    is_fallback: bool
    is_incremental: bool
    incremental_days: int | None
    signals: list[PredictionSignal]
    created_at: str | None


class TodayPredictionStatus(BaseModel):
    """今日預測狀態"""

    today: str
    week_id: str
    has_prediction: bool
    prediction: TodayPredictionDetail | None
    model_available: bool
    model_name: str | None
    model_week: str | None
    is_fallback: bool
    message: str | None


class PredictionHistoryItem(BaseModel):
    """歷史預測摘要"""

    trade_date: str
    feature_date: str
    model_name: str
    model_week: str
    is_fallback: bool
    is_incremental: bool
    incremental_days: int | None
    signal_count: int
    top_picks: list[PredictionSignal]
    created_at: str | None


class PredictionHistoryResponse(BaseModel):
    """歷史預測列表"""

    items: list[PredictionHistoryItem]
    total: int
