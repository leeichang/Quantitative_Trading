"""
回測相關 Schema
"""

from pydantic import BaseModel


class EquityCurvePoint(BaseModel):
    """權益曲線點"""

    date: str
    equity: float
    benchmark: float | None = None
    drawdown: float | None = None


class WalkForwardRequest(BaseModel):
    """Walk-Forward 回測請求"""

    start_week_id: str  # "2024W01"
    end_week_id: str  # "2025W20"
    initial_capital: float = 1000000.0
    max_positions: int = 10
    trade_price: str = "close"
    enable_incremental: bool = False
    strategy: str = "topk"


class WalkForwardRunResponse(BaseModel):
    """觸發 Walk-Forward 回測回應"""

    backtest_id: int
    job_id: str
    status: str
    message: str


class WeekStatus(BaseModel):
    """週狀態"""

    week_id: str
    status: str  # "available" | "missing" | "not_allowed"
    model_name: str | None = None
    valid_ic: float | None = None
    fallback_week: str | None = None
    fallback_model: str | None = None
    reason: str | None = None  # 不可選的原因


class AvailableWeeksResponse(BaseModel):
    """可回測週列表"""

    weeks: list[WeekStatus]
    current_week_id: str


class IcAnalysis(BaseModel):
    """IC 分析結果"""

    avg_valid_ic: float
    avg_live_ic: float
    ic_decay: float  # (valid - live) / valid * 100
    ic_correlation: float | None = None  # 驗證 IC 與實盤 IC 的相關係數


class WalkForwardReturnMetrics(BaseModel):
    """Walk-Forward 收益指標"""

    cumulative_return: float
    market_return: float
    excess_return: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None


class WeeklyDetail(BaseModel):
    """週詳情"""

    predict_week: str  # 預測的週 (e.g., "2024W02")
    model_week: str  # 使用的模型週 (e.g., "2024W01")
    model_name: str  # 模型名稱
    valid_ic: float | None = None  # 驗證期 IC
    live_ic: float | None = None  # 實盤 IC
    ic_decay: float | None = None  # IC 衰減 %
    week_return: float | None = None  # 週收益 %
    market_return: float | None = None  # 市場收益 %
    is_fallback: bool = False  # 是否使用 fallback 模型
    incremental_days: int | None = None  # 增量學習使用的天數


class WalkForwardConfig(BaseModel):
    """Walk-Forward 回測配置"""

    initial_capital: float
    max_positions: int
    trade_price: str
    enable_incremental: bool
    strategy: str


class WalkForwardResponse(BaseModel):
    """Walk-Forward 回測回應"""

    id: int
    start_week_id: str
    end_week_id: str
    status: str
    config: WalkForwardConfig
    created_at: str
    completed_at: str | None = None


class WalkForwardDetailResponse(WalkForwardResponse):
    """Walk-Forward 回測詳情"""

    ic_analysis: IcAnalysis | None = None
    return_metrics: WalkForwardReturnMetrics | None = None
    weekly_details: list[WeeklyDetail] | None = None
    equity_curve: list[EquityCurvePoint] | None = None


class WalkForwardListResponse(BaseModel):
    """Walk-Forward 回測列表"""

    items: list[WalkForwardResponse]
    total: int


class WeeklySummaryPoint(BaseModel):
    """每週摘要資料點（用於 rolling IC 與累積報酬圖表）"""

    predict_week: str
    live_ic: float | None = None
    week_return: float | None = None
    market_return: float | None = None
    cumulative_return: float | None = None
    cumulative_market: float | None = None


class WalkForwardSummaryResponse(BaseModel):
    """Walk-Forward 回測摘要（跨所有週聚合）"""

    backtest_id: int
    start_week_id: str
    end_week_id: str
    config: WalkForwardConfig
    total_weeks: int

    # IC 摘要
    mean_ic: float
    icir: float
    ic_positive_rate: float  # IC > 0 的比例 (%)

    # 報酬摘要
    annualized_return: float | None = None
    annualized_excess: float | None = None
    cumulative_return: float
    market_return: float
    excess_return: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None

    # 週級別資料（用於圖表）
    weekly_points: list[WeeklySummaryPoint]

    # 權益曲線
    equity_curve: list[EquityCurvePoint]

    created_at: str
    completed_at: str | None = None
