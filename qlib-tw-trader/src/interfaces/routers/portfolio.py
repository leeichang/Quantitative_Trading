"""
預測 API
"""

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.portfolio import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionSignal,
    PredictionsResponse,
    TodayPredictionDetail,
    TodayPredictionStatus,
)
from src.repositories.database import get_session
from src.repositories.models import DailyPrediction, StockDaily, StockUniverse
from src.repositories.prediction import PredictionRepository
from src.repositories.training import TrainingRepository
from src.services.predictor import Predictor
from src.services.qlib_exporter import ExportConfig, QlibExporter
from src.shared.constants import LOOKBACK_DAYS, TZ_TAIPEI
from src.shared.week_utils import (
    compute_week_id,
    get_previous_week_id,
)

router = APIRouter()

QLIB_DATA_DIR = Path("data/qlib")


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------


def _find_model_with_fallback(
    session: Session,
    target_week: str,
    max_lookback: int = 10,
) -> tuple | None:
    """
    尋找可用模型（含 fallback）

    Returns:
        (training_run, model_week, is_fallback) 或 None
    """
    training_repo = TrainingRepository(session)

    # 先找目標週
    model = training_repo.get_by_week_id(target_week)
    if model:
        return model, target_week, False

    # Fallback：往前找
    current = get_previous_week_id(target_week)
    for _ in range(max_lookback):
        model = training_repo.get_by_week_id(current)
        if model:
            return model, current, True
        current = get_previous_week_id(current)

    return None


def _prediction_to_detail(prediction: DailyPrediction) -> TodayPredictionDetail:
    """DailyPrediction DB row → API response"""
    signals = json.loads(prediction.signals)
    return TodayPredictionDetail(
        trade_date=prediction.trade_date.isoformat(),
        feature_date=prediction.feature_date.isoformat(),
        model_name=prediction.model_name,
        model_week=prediction.model_week,
        is_fallback=prediction.is_fallback,
        is_incremental=prediction.is_incremental,
        incremental_days=prediction.incremental_days,
        signals=[PredictionSignal(**s) for s in signals],
        created_at=prediction.created_at.isoformat() if prediction.created_at else None,
    )


# ---------------------------------------------------------------------------
# 今日預測 API
# ---------------------------------------------------------------------------


@router.get("/predictions/today", response_model=TodayPredictionStatus)
async def get_today_prediction(session: Session = Depends(get_db)):
    """取得今日預測狀態"""
    today = datetime.now(TZ_TAIPEI).date()
    week_id = compute_week_id(today)

    # 1. 檢查 DB 是否已有今日預測
    repo = PredictionRepository(session)
    prediction = repo.get_by_date(today)

    if prediction:
        return TodayPredictionStatus(
            today=today.isoformat(),
            week_id=week_id,
            has_prediction=True,
            prediction=_prediction_to_detail(prediction),
            model_available=True,
            model_name=prediction.model_name,
            model_week=prediction.model_week,
            is_fallback=prediction.is_fallback,
            message=None,
        )

    # 2. 尋找可用模型
    # 預測本週需要「上週」的模型（本週模型要到本週結束才能訓練）
    target_model_week = get_previous_week_id(week_id)
    result = _find_model_with_fallback(session, target_model_week)

    if result is None:
        return TodayPredictionStatus(
            today=today.isoformat(),
            week_id=week_id,
            has_prediction=False,
            prediction=None,
            model_available=False,
            model_name=None,
            model_week=None,
            is_fallback=False,
            message="找不到可用模型，請先到 Training 頁面訓練模型",
        )

    model_run, model_week, is_fallback = result
    return TodayPredictionStatus(
        today=today.isoformat(),
        week_id=week_id,
        has_prediction=False,
        prediction=None,
        model_available=True,
        model_name=model_run.name,
        model_week=model_week,
        is_fallback=is_fallback,
        message=None,
    )


@router.post("/predictions/today/generate")
async def generate_today_prediction(session: Session = Depends(get_db)):
    """產生今日預測（背景任務）"""
    today = datetime.now(TZ_TAIPEI).date()
    week_id = compute_week_id(today)

    # 幂等檢查
    repo = PredictionRepository(session)
    existing = repo.get_by_date(today)
    if existing:
        raise HTTPException(status_code=400, detail="今日預測已存在")

    # 找模型
    target_model_week = get_previous_week_id(week_id)
    result = _find_model_with_fallback(session, target_model_week)
    if result is None:
        raise HTTPException(status_code=400, detail="找不到可用模型")

    model_run, model_week, is_fallback = result
    model_name = model_run.name

    # 建立背景任務
    from src.services.job_manager import job_manager

    async def prediction_task(progress_callback, **kwargs):
        task_session = get_session()
        try:
            await progress_callback(5, "Loading model...")

            # 1. 載入模型
            predictor = Predictor(QLIB_DATA_DIR)
            model, factors, config = predictor._load_model(model_name)

            await progress_callback(15, "Exporting qlib data...")

            # 2. 導出 qlib 資料
            feature_date = today - timedelta(days=1)
            export_start = feature_date - timedelta(days=LOOKBACK_DAYS)
            exporter = QlibExporter(task_session)
            await asyncio.to_thread(
                exporter.export,
                ExportConfig(
                    start_date=export_start,
                    end_date=feature_date,
                    output_dir=QLIB_DATA_DIR,
                ),
            )

            await progress_callback(40, "Incremental learning...")

            # 3. 增量學習
            is_incremental = False
            incremental_days = None
            train_end_str = config.get("train_end")

            if train_end_str:
                from src.services.incremental_learner import IncrementalLearner

                model_train_end = date.fromisoformat(train_end_str)
                learner = IncrementalLearner(task_session)
                il_result = await asyncio.to_thread(
                    learner.update_to_date,
                    model,
                    factors,
                    model_train_end,
                    feature_date,
                )
                if il_result is not None:
                    model, incremental_days = il_result
                    is_incremental = True

            await progress_callback(60, "Predicting 100 stocks...")

            # 4. 預測全部 100 支股票
            actual_feature_date, signals = await asyncio.to_thread(
                predictor.predict,
                model_name,
                today,
                100,
                preloaded_model=model,
                preloaded_factors=factors,
            )

            await progress_callback(80, "Saving results...")

            # 5. 關聯股票名稱
            stock_names = {
                s.stock_id: s.name
                for s in task_session.query(StockUniverse).all()
            }
            enriched_signals = [
                {
                    "rank": s["rank"],
                    "symbol": s["symbol"],
                    "name": stock_names.get(s["symbol"]),
                    "score": round(s["score"], 6),
                }
                for s in signals
            ]

            # 6. 存入 DB
            prediction = DailyPrediction(
                trade_date=today,
                feature_date=actual_feature_date,
                model_name=model_name,
                model_week=model_week,
                is_fallback=is_fallback,
                is_incremental=is_incremental,
                incremental_days=incremental_days,
                signals=json.dumps(enriched_signals),
            )
            task_session.add(prediction)
            task_session.commit()

            await progress_callback(100, "Done")
            return {
                "trade_date": today.isoformat(),
                "signal_count": len(enriched_signals),
            }

        finally:
            task_session.close()

    job_id = await job_manager.create_job(
        job_type="predict",
        task_fn=prediction_task,
        message=f"Generating predictions for {today}",
    )

    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# 歷史預測 API
# ---------------------------------------------------------------------------


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def list_prediction_history(
    limit: int = 30,
    session: Session = Depends(get_db),
):
    """取得歷史預測列表"""
    repo = PredictionRepository(session)
    predictions = repo.list_recent(limit)

    items = []
    for p in predictions:
        signals = json.loads(p.signals)
        top_picks = [PredictionSignal(**s) for s in signals[:10]]
        items.append(
            PredictionHistoryItem(
                trade_date=p.trade_date.isoformat(),
                feature_date=p.feature_date.isoformat(),
                model_name=p.model_name,
                model_week=p.model_week,
                is_fallback=p.is_fallback,
                is_incremental=p.is_incremental,
                incremental_days=p.incremental_days,
                signal_count=len(signals),
                top_picks=top_picks,
                created_at=p.created_at.isoformat() if p.created_at else None,
            )
        )

    return PredictionHistoryResponse(items=items, total=len(items))


# ---------------------------------------------------------------------------
# 舊的手動預測 API（保留向後兼容）
# ---------------------------------------------------------------------------


@router.post("/predictions/generate", response_model=PredictionsResponse)
async def generate_predictions(
    request: PredictionRequest,
    session: Session = Depends(get_db),
):
    """執行預測並返回 Top K 股票"""
    # 驗證模型存在
    training_repo = TrainingRepository(session)
    model = training_repo.get_by_id(request.model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.name:
        raise HTTPException(status_code=400, detail="Model has no name")

    if model.status != "completed":
        raise HTTPException(status_code=400, detail="Model is not completed")

    # 決定交易日期
    if request.trade_date:
        trade_date = request.trade_date
    else:
        # 使用資料庫中最新資料日期的下一天作為交易日
        latest_date_row = session.query(func.max(StockDaily.date)).first()
        if not latest_date_row or not latest_date_row[0]:
            raise HTTPException(status_code=400, detail="No stock data available")
        trade_date = latest_date_row[0] + timedelta(days=1)

    # 導出 qlib 資料（因子計算需要歷史資料）
    export_end = trade_date - timedelta(days=1)
    export_start = export_end - timedelta(days=LOOKBACK_DAYS)

    def do_export():
        exporter = QlibExporter(session)
        exporter.export(ExportConfig(
            start_date=export_start,
            end_date=export_end,
            output_dir=QLIB_DATA_DIR,
        ))

    await asyncio.to_thread(do_export)

    # 執行預測
    def do_predict():
        predictor = Predictor(QLIB_DATA_DIR)
        return predictor.predict(
            model_name=model.name,
            trade_date=trade_date,
            top_k=request.top_k,
        )

    feature_date, signals = await asyncio.to_thread(do_predict)

    # 關聯股票名稱
    stock_names = {s.stock_id: s.name for s in session.query(StockUniverse).all()}

    response_signals = [
        PredictionSignal(
            rank=sig["rank"],
            symbol=sig["symbol"],
            name=stock_names.get(sig["symbol"]),
            score=round(sig["score"], 6),
        )
        for sig in signals
    ]

    return PredictionsResponse(
        trade_date=trade_date.isoformat(),
        feature_date=feature_date.isoformat(),
        model_name=model.name,
        signals=response_signals,
    )
