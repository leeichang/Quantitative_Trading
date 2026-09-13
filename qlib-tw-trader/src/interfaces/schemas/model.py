from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel


class Period(BaseModel):
    """期間"""

    start: date
    end: date


class ModelMetrics(BaseModel):
    """模型指標"""

    ic: float | None
    icir: float | None


class FactorSummary(BaseModel):
    """因子摘要（用於模型詳情）"""

    id: str
    name: str
    display_name: str | None
    category: str
    ic_value: float | None = None  # 該因子在此模型中的 IC 值


class SelectionInfo(BaseModel):
    """因子選擇資訊"""

    method: str | None = None  # 選擇方法（如 composite_threshold_v1）
    config: dict | None = None  # 選擇配置
    stats: dict | None = None  # 選擇統計


class ModelResponse(BaseModel):
    """模型詳情回應"""

    id: str
    name: str | None
    status: str  # queued / running / completed / failed
    trained_at: datetime
    factor_count: int | None
    factors: list[str]  # 向後兼容：選中因子名稱列表
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    training_duration_seconds: int | None
    candidate_factors: list[FactorSummary] = []  # 候選因子
    selected_factors: list[FactorSummary] = []  # 選中因子
    selection: SelectionInfo | None = None  # 因子選擇資訊


class ModelSummary(BaseModel):
    """模型列表項目"""

    id: str
    name: str | None
    status: str
    trained_at: datetime
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    factor_count: int | None  # 選中因子數
    candidate_count: int | None  # 候選因子數
    selection_method: str | None = None  # 因子選擇方法


# 向後兼容的別名
ModelHistoryItem = ModelSummary


class ModelHistoryResponse(BaseModel):
    """歷史模型列表回應（向後兼容）"""

    items: list[ModelHistoryItem]
    total: int


class DataRange(BaseModel):
    """資料日期範圍"""

    start: date
    end: date


class WeekModel(BaseModel):
    """週模型資訊"""

    id: str  # "m001"
    name: str  # "2026W05-a1b2c3"
    model_ic: float
    factor_count: int
    factor_pool_hash: str | None
    is_outdated: bool  # 因子池已變更


class WeekSlot(BaseModel):
    """週訓練時段"""

    week_id: str  # "2026W05"
    valid_end: date
    valid_start: date
    train_end: date
    train_start: date
    status: Literal["trained", "trainable", "insufficient_data"]
    model: WeekModel | None = None


class WeeksResponse(BaseModel):
    """週列表回應"""

    slots: list[WeekSlot]
    current_factor_pool_hash: str
    data_range: DataRange


class TrainRequest(BaseModel):
    """訓練請求"""

    week_id: str  # "2026W05"


class TrainBatchRequest(BaseModel):
    """批量訓練請求"""

    year: str  # "2025"


class TrainResponse(BaseModel):
    """訓練回應"""

    job_id: str
    status: str
    message: str


class QualityMetricsItem(BaseModel):
    """訓練品質指標項目"""

    training_run_id: int
    week_id: str | None
    factor_jaccard_sim: float | None
    factor_overlap_count: int | None
    ic_moving_avg_5w: float | None
    ic_moving_std_5w: float | None
    icir_5w: float | None
    has_warning: bool
    warning_type: str | None
    warning_message: str | None
    created_at: str | None


class QualityResponse(BaseModel):
    """訓練品質回應"""

    items: list[QualityMetricsItem]
    thresholds: dict[str, float]
