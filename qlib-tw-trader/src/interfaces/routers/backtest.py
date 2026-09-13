"""
回測 API
"""

import json
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
import statistics

from src.interfaces.schemas.backtest import (
    AvailableWeeksResponse,
    EquityCurvePoint,
    IcAnalysis,
    WalkForwardConfig,
    WalkForwardDetailResponse,
    WalkForwardListResponse,
    WalkForwardRequest,
    WalkForwardResponse,
    WalkForwardReturnMetrics,
    WalkForwardRunResponse,
    WalkForwardSummaryResponse,
    WeeklyDetail,
    WeeklySummaryPoint,
    WeekStatus,
)
from src.services.job_manager import job_manager

router = APIRouter()


@router.get("/walk-forward/available-weeks", response_model=AvailableWeeksResponse)
async def get_available_weeks(
    session: Session = Depends(get_db),
):
    """
    取得可回測的週列表

    返回所有週的狀態（可用、缺失需 fallback、不可選）
    """
    from src.services.walk_forward_backtester import WalkForwardBacktester
    from src.shared.week_utils import get_current_week_id

    backtester = WalkForwardBacktester(session)
    weeks = backtester.get_available_weeks()

    return AvailableWeeksResponse(
        weeks=[
            WeekStatus(
                week_id=w["week_id"],
                status=w["status"],
                model_name=w.get("model_name"),
                valid_ic=w.get("valid_ic"),
                fallback_week=w.get("fallback_week"),
                fallback_model=w.get("fallback_model"),
                reason=w.get("reason"),
            )
            for w in weeks
        ],
        current_week_id=get_current_week_id(),
    )


@router.get("/walk-forward", response_model=WalkForwardListResponse)
async def list_walk_forward_backtests(
    session: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
):
    """取得 Walk-Forward 回測列表"""
    from src.repositories.walk_forward import WalkForwardBacktestRepository

    repo = WalkForwardBacktestRepository(session)
    backtests = repo.get_recent(limit)

    return WalkForwardListResponse(
        items=[
            WalkForwardResponse(
                id=bt.id,
                start_week_id=bt.start_week_id,
                end_week_id=bt.end_week_id,
                status=bt.status,
                config=WalkForwardConfig(
                    initial_capital=float(bt.initial_capital),
                    max_positions=bt.max_positions,
                    trade_price=bt.trade_price,
                    enable_incremental=bt.enable_incremental,
                    strategy=bt.strategy,
                ),
                created_at=bt.created_at.isoformat() if bt.created_at else "",
                completed_at=bt.completed_at.isoformat() if bt.completed_at else None,
            )
            for bt in backtests
        ],
        total=len(backtests),
    )


@router.get("/walk-forward/summary", response_model=WalkForwardSummaryResponse)
async def get_walk_forward_summary(
    backtest_id: int | None = Query(None, description="指定回測 ID，若無則取最新完成的"),
    session: Session = Depends(get_db),
):
    """
    取得 Walk-Forward 回測摘要

    聚合 IC、報酬等指標，提供圖表所需的週級別資料。
    """
    from src.repositories.walk_forward import WalkForwardBacktestRepository

    repo = WalkForwardBacktestRepository(session)

    if backtest_id is not None:
        bt = repo.get(backtest_id)
        if not bt or bt.status != "completed":
            raise HTTPException(status_code=404, detail="Completed backtest not found")
    else:
        bt = repo.get_latest_completed()
        if not bt:
            raise HTTPException(status_code=404, detail="No completed backtests found")

    # 解析 weekly_details
    weekly_details: list[dict] = []
    if bt.weekly_details:
        try:
            weekly_details = json.loads(bt.weekly_details)
        except json.JSONDecodeError:
            pass

    # 解析 result
    result_data: dict = {}
    if bt.result:
        try:
            result_data = json.loads(bt.result)
        except json.JSONDecodeError:
            pass

    # 解析 equity_curve
    equity_curve_raw: list[dict] = []
    if bt.equity_curve:
        try:
            equity_curve_raw = json.loads(bt.equity_curve)
        except json.JSONDecodeError:
            pass

    # 計算 IC 摘要
    live_ics = [w["live_ic"] for w in weekly_details if w.get("live_ic") is not None]
    mean_ic = statistics.mean(live_ics) if live_ics else 0.0
    ic_std = statistics.stdev(live_ics) if len(live_ics) > 1 else 0.0
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    ic_positive_rate = (sum(1 for ic in live_ics if ic > 0) / len(live_ics) * 100) if live_ics else 0.0

    # 從 result_data 取報酬指標
    ret = result_data.get("return_metrics", {})
    cumulative_return = ret.get("cumulative_return", 0.0)
    market_return = ret.get("market_return", 0.0)
    excess_return = ret.get("excess_return", 0.0)

    # 估算年化報酬（假設每週一筆，一年約 52 週）
    total_weeks = len(weekly_details)
    annualized_return = None
    annualized_excess = None
    if total_weeks > 0:
        weeks_per_year = 52
        cum_factor = 1 + cumulative_return / 100
        if cum_factor > 0:
            annualized_return = (cum_factor ** (weeks_per_year / total_weeks) - 1) * 100
        mkt_factor = 1 + market_return / 100
        if cum_factor > 0 and mkt_factor > 0:
            excess_factor = cum_factor / mkt_factor
            annualized_excess = (excess_factor ** (weeks_per_year / total_weeks) - 1) * 100

    # 建構 weekly_points（含累積報酬）
    weekly_points = []
    cum_ret = 0.0
    cum_mkt = 0.0
    for w in weekly_details:
        wr = w.get("week_return") or 0.0
        mr = w.get("market_return") or 0.0
        cum_ret = (1 + cum_ret / 100) * (1 + wr / 100) * 100 - 100
        cum_mkt = (1 + cum_mkt / 100) * (1 + mr / 100) * 100 - 100
        weekly_points.append(WeeklySummaryPoint(
            predict_week=w.get("predict_week", ""),
            live_ic=w.get("live_ic"),
            week_return=w.get("week_return"),
            market_return=w.get("market_return"),
            cumulative_return=round(cum_ret, 4),
            cumulative_market=round(cum_mkt, 4),
        ))

    equity_curve = [
        EquityCurvePoint(
            date=p.get("date", ""),
            equity=p.get("equity", 0),
            benchmark=p.get("benchmark"),
            drawdown=p.get("drawdown"),
        )
        for p in equity_curve_raw
    ]

    return WalkForwardSummaryResponse(
        backtest_id=bt.id,
        start_week_id=bt.start_week_id,
        end_week_id=bt.end_week_id,
        config=WalkForwardConfig(
            initial_capital=float(bt.initial_capital),
            max_positions=bt.max_positions,
            trade_price=bt.trade_price,
            enable_incremental=bt.enable_incremental,
            strategy=bt.strategy,
        ),
        total_weeks=total_weeks,
        mean_ic=round(mean_ic, 6),
        icir=round(icir, 4),
        ic_positive_rate=round(ic_positive_rate, 1),
        annualized_return=round(annualized_return, 2) if annualized_return is not None else None,
        annualized_excess=round(annualized_excess, 2) if annualized_excess is not None else None,
        cumulative_return=cumulative_return,
        market_return=market_return,
        excess_return=excess_return,
        sharpe_ratio=ret.get("sharpe_ratio"),
        max_drawdown=ret.get("max_drawdown"),
        win_rate=ret.get("win_rate"),
        total_trades=ret.get("total_trades"),
        weekly_points=weekly_points,
        equity_curve=equity_curve,
        created_at=bt.created_at.isoformat() if bt.created_at else "",
        completed_at=bt.completed_at.isoformat() if bt.completed_at else None,
    )


@router.get("/walk-forward/{backtest_id}", response_model=WalkForwardDetailResponse)
async def get_walk_forward_backtest(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """取得 Walk-Forward 回測詳情"""
    from src.repositories.walk_forward import WalkForwardBacktestRepository

    repo = WalkForwardBacktestRepository(session)
    bt = repo.get(backtest_id)

    if not bt:
        raise HTTPException(status_code=404, detail="Walk-forward backtest not found")

    # 解析結果
    ic_analysis = None
    return_metrics = None
    weekly_details = None
    equity_curve = None

    if bt.result:
        try:
            result_data = json.loads(bt.result)
            if "ic_analysis" in result_data:
                ic_data = result_data["ic_analysis"]
                ic_analysis = IcAnalysis(
                    avg_valid_ic=ic_data.get("avg_valid_ic", 0),
                    avg_live_ic=ic_data.get("avg_live_ic", 0),
                    ic_decay=ic_data.get("ic_decay", 0),
                    ic_correlation=ic_data.get("ic_correlation"),
                )
            if "return_metrics" in result_data:
                ret_data = result_data["return_metrics"]
                return_metrics = WalkForwardReturnMetrics(
                    cumulative_return=ret_data.get("cumulative_return", 0),
                    market_return=ret_data.get("market_return", 0),
                    excess_return=ret_data.get("excess_return", 0),
                    sharpe_ratio=ret_data.get("sharpe_ratio"),
                    max_drawdown=ret_data.get("max_drawdown"),
                    win_rate=ret_data.get("win_rate"),
                    total_trades=ret_data.get("total_trades"),
                )
        except json.JSONDecodeError:
            pass

    if bt.weekly_details:
        try:
            details_data = json.loads(bt.weekly_details)
            weekly_details = [
                WeeklyDetail(
                    predict_week=d.get("predict_week", ""),
                    model_week=d.get("model_week", ""),
                    model_name=d.get("model_name", ""),
                    valid_ic=d.get("valid_ic"),
                    live_ic=d.get("live_ic"),
                    ic_decay=d.get("ic_decay"),
                    week_return=d.get("week_return"),
                    market_return=d.get("market_return"),
                    is_fallback=d.get("is_fallback", False),
                    incremental_days=d.get("incremental_days"),
                )
                for d in details_data
            ]
        except json.JSONDecodeError:
            pass

    if bt.equity_curve:
        try:
            curve_data = json.loads(bt.equity_curve)
            equity_curve = [
                EquityCurvePoint(
                    date=p.get("date", ""),
                    equity=p.get("equity", 0),
                    benchmark=p.get("benchmark"),
                    drawdown=p.get("drawdown"),
                )
                for p in curve_data
            ]
        except json.JSONDecodeError:
            pass

    return WalkForwardDetailResponse(
        id=bt.id,
        start_week_id=bt.start_week_id,
        end_week_id=bt.end_week_id,
        status=bt.status,
        config=WalkForwardConfig(
            initial_capital=float(bt.initial_capital),
            max_positions=bt.max_positions,
            trade_price=bt.trade_price,
            enable_incremental=bt.enable_incremental,
            strategy=bt.strategy,
        ),
        ic_analysis=ic_analysis,
        return_metrics=return_metrics,
        weekly_details=weekly_details,
        equity_curve=equity_curve,
        created_at=bt.created_at.isoformat() if bt.created_at else "",
        completed_at=bt.completed_at.isoformat() if bt.completed_at else None,
    )


@router.post("/walk-forward", response_model=WalkForwardRunResponse)
async def run_walk_forward_backtest(
    request: WalkForwardRequest,
    session: Session = Depends(get_db),
):
    """執行 Walk-Forward 回測"""
    from src.repositories.walk_forward import WalkForwardBacktestRepository
    from src.shared.week_utils import compare_week_ids, get_current_week_id

    # 驗證週 ID
    current_week = get_current_week_id()
    if compare_week_ids(request.end_week_id, current_week) >= 0:
        raise HTTPException(
            status_code=400,
            detail=f"End week must be before current week ({current_week})"
        )

    if compare_week_ids(request.start_week_id, request.end_week_id) > 0:
        raise HTTPException(
            status_code=400,
            detail="Start week must be before or equal to end week"
        )

    # 建立回測記錄
    repo = WalkForwardBacktestRepository(session)
    backtest = repo.create(
        start_week_id=request.start_week_id,
        end_week_id=request.end_week_id,
        initial_capital=Decimal(str(request.initial_capital)),
        max_positions=request.max_positions,
        trade_price=request.trade_price,
        enable_incremental=request.enable_incremental,
        strategy=request.strategy,
    )

    # 建立非同步任務
    job_id = await job_manager.create_job(
        job_type="walk_forward",
        task_fn=run_walk_forward_task,
        message=f"Running walk-forward backtest {backtest.id}",
        backtest_id=backtest.id,
        start_week_id=request.start_week_id,
        end_week_id=request.end_week_id,
        initial_capital=request.initial_capital,
        max_positions=request.max_positions,
        trade_price=request.trade_price,
        enable_incremental=request.enable_incremental,
    )

    return WalkForwardRunResponse(
        backtest_id=backtest.id,
        job_id=job_id,
        status="queued",
        message="Walk-forward backtest started",
    )


@router.delete("/walk-forward/{backtest_id}")
async def delete_walk_forward_backtest(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """刪除 Walk-Forward 回測記錄"""
    from src.repositories.walk_forward import WalkForwardBacktestRepository

    repo = WalkForwardBacktestRepository(session)
    if not repo.delete(backtest_id):
        raise HTTPException(status_code=404, detail="Walk-forward backtest not found")

    return {"message": "Walk-forward backtest deleted", "id": backtest_id}


async def run_walk_forward_task(
    progress_callback,
    backtest_id: int,
    start_week_id: str,
    end_week_id: str,
    initial_capital: float,
    max_positions: int,
    trade_price: str = "open",
    enable_incremental: bool = False,
):
    """Walk-Forward 回測任務"""
    import asyncio

    from src.repositories.database import get_session
    from src.repositories.walk_forward import WalkForwardBacktestRepository
    from src.services.walk_forward_backtester import WalkForwardBacktester

    session = get_session()
    repo = WalkForwardBacktestRepository(session)

    loop = asyncio.get_event_loop()

    def sync_progress(progress: float, message: str):
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(progress_callback(progress, message))
        )

    try:
        repo.update_status(backtest_id, "running")
        await progress_callback(1, "Starting walk-forward backtest...")

        def do_backtest():
            bt_session = get_session()
            try:
                backtester = WalkForwardBacktester(bt_session)
                return backtester.run(
                    start_week_id=start_week_id,
                    end_week_id=end_week_id,
                    initial_capital=initial_capital,
                    max_positions=max_positions,
                    trade_price=trade_price,
                    enable_incremental=enable_incremental,
                    on_progress=sync_progress,
                )
            finally:
                bt_session.close()

        result = await asyncio.to_thread(do_backtest)

        await progress_callback(95, "Saving results...")

        # 轉換結果為 dict
        result_dict = {
            "ic_analysis": {
                "avg_valid_ic": result.ic_analysis.avg_valid_ic,
                "avg_live_ic": result.ic_analysis.avg_live_ic,
                "ic_decay": result.ic_analysis.ic_decay,
                "ic_correlation": result.ic_analysis.ic_correlation,
            },
            "return_metrics": {
                "cumulative_return": result.return_metrics.cumulative_return,
                "market_return": result.return_metrics.market_return,
                "excess_return": result.return_metrics.excess_return,
                "sharpe_ratio": result.return_metrics.sharpe_ratio,
                "max_drawdown": result.return_metrics.max_drawdown,
                "win_rate": result.return_metrics.win_rate,
                "total_trades": result.return_metrics.total_trades,
            },
        }

        weekly_details = [
            {
                "predict_week": w.predict_week,
                "model_week": w.model_week,
                "model_name": w.model_name,
                "valid_ic": w.valid_ic,
                "live_ic": w.live_ic,
                "ic_decay": w.ic_decay,
                "week_return": w.week_return,
                "market_return": w.market_return,
                "is_fallback": w.is_fallback,
                "incremental_days": w.incremental_days,
            }
            for w in result.weekly_details
        ]

        equity_curve = [
            {
                "date": p.date,
                "equity": p.equity,
                "benchmark": p.benchmark,
                "drawdown": p.drawdown,
            }
            for p in result.equity_curve
        ]

        repo.complete(backtest_id, result_dict, weekly_details, equity_curve)

        await progress_callback(
            100,
            f"Completed! Live IC: {result.ic_analysis.avg_live_ic:.4f}, "
            f"Decay: {result.ic_analysis.ic_decay:.1f}%"
        )

        return {
            "backtest_id": backtest_id,
            "status": "completed",
            **result_dict,
        }

    except Exception as e:
        repo.fail(backtest_id, str(e))
        raise

    finally:
        session.close()
