#!/usr/bin/env python3
"""
資料載入層測試

分兩類：
  · unit        用臨時 SQLite 建構已知資料，驗證邏輯（還原、剔除、驗證）
  · integration 打真實上游 DB，驗證接得上且資料合理

執行：
    .venv/bin/python -m pytest tests/test_loader.py -v
    .venv/bin/python -m pytest tests/test_loader.py -m unit -q        # 只跑不碰 DB 的
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from taiwan_quant.data.loader import (
    _apply_adjustment,
    DEFAULT_DB_PATH,
    DataNotAvailableError,
    coverage_report,
    load_prices,
    load_universe,
    price_quality_report,
)

# ══════════════════════════════════════════════════════════════
# 臨時 DB fixture：建構已知資料，讓斷言可以手算
# ══════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE stock_universe (
    stock_id TEXT PRIMARY KEY, name TEXT, market_cap INTEGER, rank INTEGER,
    updated_at TEXT
);
CREATE TABLE stock_daily (
    stock_id TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
    volume INTEGER, PRIMARY KEY (stock_id, date)
);
CREATE TABLE stock_daily_adj (
    stock_id TEXT, date TEXT, adj_close REAL, PRIMARY KEY (stock_id, date)
);
CREATE TABLE stock_daily_institutional (stock_id TEXT, date TEXT);
CREATE TABLE stock_daily_margin (stock_id TEXT, date TEXT);
CREATE TABLE stock_daily_per (stock_id TEXT, date TEXT);
CREATE TABLE stock_daily_securities_lending (stock_id TEXT, date TEXT);
CREATE TABLE stock_daily_shareholding (stock_id TEXT, date TEXT);
CREATE TABLE stock_monthly_revenue (stock_id TEXT, revenue_month TEXT);
"""


@pytest.fixture
def fake_db(tmp_path: Path) -> Path:
    """
    建一個臨時上游 DB。

    2330：3 天資料，其中 D2 的 adj_close 為原價的兩倍（模擬除權息還原），
          D3 的 OHLC 全為 NULL（模擬上游資料洞）
    2317：1 天正常資料
    """
    db = tmp_path / "fake.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    con.executemany(
        "INSERT INTO stock_universe VALUES (?,?,?,?,?)",
        [
            ("2330", "台積電", 600_000, 1, "2026-09-11"),
            ("2317", "鴻海", 30_000, 2, "2026-09-11"),
            ("9999", "小型股", 1_000, 60, "2026-09-11"),
        ],
    )
    con.executemany(
        "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)",
        [
            ("2330", "2026-01-05", 100.0, 110.0, 90.0, 100.0, 1000),
            ("2330", "2026-01-06", 100.0, 110.0, 90.0, 100.0, 2000),
            ("2330", "2026-01-07", None, None, None, None, 0),
            ("2317", "2026-01-05", 50.0, 52.0, 48.0, 50.0, 500),
        ],
    )
    con.executemany(
        "INSERT INTO stock_daily_adj VALUES (?,?,?)",
        [
            ("2330", "2026-01-05", 100.0),   # 因子 1.0
            ("2330", "2026-01-06", 200.0),   # 因子 2.0
            ("2317", "2026-01-05", 50.0),
        ],
    )
    con.commit()
    con.close()
    return db


# ══════════════════════════════════════════════════════════════
# unit：標的池
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_universe_orders_by_rank(fake_db: Path) -> None:
    uni = load_universe(db_path=fake_db, limit=150)
    assert uni.stock_ids == ["2330", "2317", "9999"]


@pytest.mark.unit
def test_universe_respects_limit(fake_db: Path) -> None:
    uni = load_universe(db_path=fake_db, limit=2)
    assert uni.stock_ids == ["2330", "2317"]


@pytest.mark.unit
def test_universe_tier_split_at_rank_50(fake_db: Path) -> None:
    """
    流動性分層代理規則：rank <= 50 視為 0050 級，其餘 0051 級。

    2330 rank 1  → 0050（滑價 0.3%）
    9999 rank 60 → 0051（滑價 0.4%）
    """
    uni = load_universe(db_path=fake_db)
    assert uni.tier_of("2330") == "0050"
    assert uni.tier_of("9999") == "0051"


@pytest.mark.unit
def test_universe_tier_rejects_unknown_stock(fake_db: Path) -> None:
    uni = load_universe(db_path=fake_db)
    with pytest.raises(KeyError):
        uni.tier_of("0000")


@pytest.mark.unit
def test_universe_warning_discloses_known_biases(fake_db: Path) -> None:
    """
    D2 要求 150 檔真成分股，實際只有市值代理 → 必須揭露三件事：
    代理、無歷史快照（survivorship）、檔數不足。
    """
    uni = load_universe(db_path=fake_db, limit=150, target_count=150)
    notes = uni.warning.describe()

    assert uni.warning.is_constituent_proxy is True
    assert uni.warning.has_historical_snapshot is False
    assert len(notes) == 3
    assert any("survivorship" in n for n in notes)
    assert any("代理" in n for n in notes)


@pytest.mark.unit
def test_universe_raises_when_table_empty(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    with pytest.raises(DataNotAvailableError, match="stock_universe"):
        load_universe(db_path=db)


@pytest.mark.unit
def test_missing_db_raises_with_actionable_message(tmp_path: Path) -> None:
    """找不到 DB 時錯誤訊息要告訴使用者怎麼辦，不是只丟路徑"""
    with pytest.raises(DataNotAvailableError, match="QUICKSTART"):
        load_universe(db_path=tmp_path / "nope.db")


# ══════════════════════════════════════════════════════════════
# unit：價格還原（CLAUDE.md 禁令 12）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_adjustment_scales_all_ohlc_by_same_factor(fake_db: Path) -> None:
    """
    還原因子 = adj_close / close，必須同比例套用到 OHL，K 線形狀不變。

    2330 D2：close 100 → adj_close 200，因子 = 2.0
        open 100 → 200
        high 110 → 220
        low   90 → 180
        close 100 → 200
    """
    px = load_prices(["2330"], db_path=fake_db, adjusted=True)
    row = px.loc[("2330", pd.Timestamp("2026-01-06"))]

    assert row["open"] == pytest.approx(200.0)
    assert row["high"] == pytest.approx(220.0)
    assert row["low"] == pytest.approx(180.0)
    assert row["close"] == pytest.approx(200.0)
    assert bool(row["is_adjusted"]) is True


@pytest.mark.unit
def test_adjustment_factor_one_leaves_prices_unchanged(fake_db: Path) -> None:
    """因子為 1.0 時價格不變（2330 D1：close 100、adj_close 100）"""
    px = load_prices(["2330"], db_path=fake_db, adjusted=True)
    row = px.loc[("2330", pd.Timestamp("2026-01-05"))]
    assert row["close"] == pytest.approx(100.0)
    assert row["high"] == pytest.approx(110.0)


@pytest.mark.unit
def test_unadjusted_mode_returns_raw_prices(fake_db: Path) -> None:
    """adjusted=False 時回傳原始未還原價"""
    px = load_prices(["2330"], db_path=fake_db, adjusted=False)
    row = px.loc[("2330", pd.Timestamp("2026-01-06"))]
    assert row["close"] == pytest.approx(100.0), "未還原模式不應套用因子"
    assert "is_adjusted" not in px.columns


@pytest.mark.unit
def test_prices_are_float_not_int(fake_db: Path) -> None:
    """
    SQLite 可能回 int64；還原會寫入 float。
    pandas 3.0 不允許把 float 寫進 int64 欄位，所以必須先轉型。
    """
    px = load_prices(["2330"], db_path=fake_db, adjusted=True)
    for col in ("open", "high", "low", "close"):
        assert px[col].dtype == "float64", f"{col} 應為 float64，得到 {px[col].dtype}"


# ══════════════════════════════════════════════════════════════
# unit：資料洞剔除
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_incomplete_rows_dropped_by_default(fake_db: Path) -> None:
    """
    2330 有 3 天，其中 D3 的 OHLC 全為 NULL → 預設剔除，剩 2 天。

    為什麼必須剔除：單一 NaN 會汙染 np.quantile，而 NaN 通過門檻比較時
    不會拋錯，而是讓 `NaN < 門檻` 為 False —— 靜默放行。
    """
    px = load_prices(["2330"], db_path=fake_db, drop_incomplete=True)
    assert len(px) == 2
    assert pd.Timestamp("2026-01-07") not in px.loc["2330"].index


@pytest.mark.unit
def test_incomplete_rows_kept_when_disabled(fake_db: Path) -> None:
    px = load_prices(["2330"], db_path=fake_db, drop_incomplete=False, adjusted=False)
    assert len(px) == 3


@pytest.mark.unit
def test_quality_report_lists_data_holes(fake_db: Path) -> None:
    """
    資料洞必須被揭露而非靜默丟棄——使用者有權知道哪幾天被剔除了。
    """
    rep = price_quality_report(["2330", "2317"], db_path=fake_db)
    assert len(rep) == 1
    row = rep.iloc[0]
    assert row["stock_id"] == "2330"
    assert row["incomplete_rows"] == 1
    assert row["incomplete_dates"] == ["2026-01-07"]


@pytest.mark.unit
def test_quality_report_empty_when_no_holes(fake_db: Path) -> None:
    rep = price_quality_report(["2317"], db_path=fake_db)
    assert rep.empty
    assert list(rep.columns) == [
        "stock_id", "total_rows", "incomplete_rows", "incomplete_dates"
    ]


# ══════════════════════════════════════════════════════════════
# unit：篩選與錯誤處理
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_date_range_filter_is_inclusive(fake_db: Path) -> None:
    px = load_prices(
        ["2330"], start=date(2026, 1, 6), end=date(2026, 1, 6), db_path=fake_db
    )
    assert len(px) == 1
    assert px.index[0][1] == pd.Timestamp("2026-01-06")


@pytest.mark.unit
def test_stock_filter_applies(fake_db: Path) -> None:
    px = load_prices(["2317"], db_path=fake_db)
    assert px.index.get_level_values("stock_id").unique().tolist() == ["2317"]


@pytest.mark.unit
def test_index_is_sorted_multiindex(fake_db: Path) -> None:
    """下游 label_series 要求日期升冪，索引必須排好"""
    px = load_prices(db_path=fake_db)
    assert px.index.names == ["stock_id", "date"]
    assert px.index.is_monotonic_increasing


@pytest.mark.unit
def test_empty_query_raises(fake_db: Path) -> None:
    with pytest.raises(DataNotAvailableError, match="查無日 K"):
        load_prices(["0000"], db_path=fake_db)


@pytest.mark.unit
def test_coverage_report_flags_sparse_tables(fake_db: Path) -> None:
    """
    覆蓋率 < 80% 的表標記為不可用。

    fake_db 的 stock_daily 有 4 列，其他表都是 0 列 → coverage 0，不可用。
    """
    rep = coverage_report(db_path=fake_db)
    by_table = rep.set_index("table")

    assert by_table.loc["stock_daily", "coverage"] == pytest.approx(1.0)
    assert bool(by_table.loc["stock_daily", "usable"]) is True
    assert bool(by_table.loc["stock_daily_per", "usable"]) is False


# ══════════════════════════════════════════════════════════════
# integration：接真實上游 DB
# ══════════════════════════════════════════════════════════════


requires_upstream = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(),
    reason=f"上游資料庫不存在：{DEFAULT_DB_PATH}",
)


@pytest.mark.integration
@requires_upstream
def test_real_universe_loads() -> None:
    uni = load_universe(limit=150)
    assert len(uni.stocks) > 0
    assert uni.stock_ids[0] == "2330", "市值第一應為台積電"


@pytest.mark.integration
@requires_upstream
def test_real_prices_are_adjusted_and_complete() -> None:
    """真實資料載入後不得有 NaN 或非正價格（剔除已生效）"""
    px = load_prices(["2330", "2317"], start=date(2023, 1, 1), end=date(2026, 9, 11))
    for col in ("open", "high", "low", "close"):
        assert px[col].notna().all(), f"{col} 仍有 NaN"
        assert (px[col] > 0).all(), f"{col} 仍有非正值"


@pytest.mark.integration
@requires_upstream
def test_real_ohlc_ordering_is_valid() -> None:
    """還原後 high >= max(open, close) 且 low <= min(open, close)"""
    px = load_prices(["2330"], start=date(2025, 1, 1), end=date(2026, 9, 11))
    assert (px["high"] >= px[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (px["low"] <= px[["open", "close"]].min(axis=1) + 1e-6).all()


@pytest.mark.integration
@requires_upstream
def test_real_coverage_report_matches_known_state() -> None:
    """
    已知上游狀態（2026-09-12 實測）：
        stock_daily / adj / institutional / margin 覆蓋率高 → 可用
        per / shareholding / securities_lending / monthly_revenue 低 → 不可用
        （FinMind 免費額度撞牆所致）
    """
    rep = coverage_report().set_index("table")
    assert bool(rep.loc["stock_daily_institutional", "usable"]) is True
    assert bool(rep.loc["stock_daily_margin", "usable"]) is True
    assert bool(rep.loc["stock_daily_per", "usable"]) is False


# ══════════════════════════════════════════════════════════════
# 籌碼資料載入
#
# 上游把買進 / 賣出分開存（foreign_buy / foreign_sell），
# 但特徵層要的是**買賣超淨額**。轉換在載入層做，讓
# features/chips.py 只處理一種欄位語意。
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def fake_chips_db(tmp_path: Path) -> Path:
    """含籌碼資料的臨時 DB"""
    db = tmp_path / "chips.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.executescript("""
        DROP TABLE stock_daily_institutional;
        CREATE TABLE stock_daily_institutional (
            stock_id TEXT, date TEXT,
            foreign_buy INTEGER, foreign_sell INTEGER,
            trust_buy INTEGER, trust_sell INTEGER,
            dealer_buy INTEGER, dealer_sell INTEGER,
            PRIMARY KEY (stock_id, date)
        );
        DROP TABLE stock_daily_margin;
        CREATE TABLE stock_daily_margin (
            stock_id TEXT, date TEXT,
            margin_buy INTEGER, margin_sell INTEGER, margin_balance INTEGER,
            short_buy INTEGER, short_sell INTEGER, short_balance INTEGER,
            PRIMARY KEY (stock_id, date)
        );
    """)
    con.executemany(
        "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)",
        [
            ("2330", "2026-01-05", 100.0, 110.0, 90.0, 100.0, 1000),
            ("2330", "2026-01-06", 100.0, 110.0, 90.0, 105.0, 2000),
        ],
    )
    con.executemany(
        "INSERT INTO stock_daily_adj VALUES (?,?,?)",
        [("2330", "2026-01-05", 100.0), ("2330", "2026-01-06", 105.0)],
    )
    con.executemany(
        "INSERT INTO stock_daily_institutional VALUES (?,?,?,?,?,?,?,?)",
        [
            # foreign 買 500 賣 200 → 淨 +300
            ("2330", "2026-01-05", 500, 200, 100, 150, 50, 50),
            ("2330", "2026-01-06", 100, 400, 300, 100, 20, 60),
        ],
    )
    con.executemany(
        "INSERT INTO stock_daily_margin VALUES (?,?,?,?,?,?,?,?)",
        [
            ("2330", "2026-01-05", 10, 5, 1000, 2, 1, 100),
            ("2330", "2026-01-06", 20, 8, 1012, 3, 2, 101),
        ],
    )
    con.commit()
    con.close()
    return db


@pytest.mark.unit
def test_chips_computes_net_from_buy_and_sell(fake_chips_db: Path) -> None:
    """
    手算：foreign_buy 500 − foreign_sell 200 = +300
          trust_buy 100 − trust_sell 150 = −50
          dealer_buy 50 − dealer_sell 50 = 0
    """
    from taiwan_quant.data.loader import load_chips

    chips = load_chips(["2330"], db_path=fake_chips_db)
    row = chips.loc[("2330", pd.Timestamp("2026-01-05"))]

    assert row["foreign_net"] == pytest.approx(300.0)
    assert row["trust_net"] == pytest.approx(-50.0)
    assert row["dealer_net"] == pytest.approx(0.0)


@pytest.mark.unit
def test_chips_includes_required_columns(fake_chips_db: Path) -> None:
    """
    產出的欄位必須正好覆蓋 features/chips.py 的需求，
    否則 build_chips 會在執行期才炸。
    """
    from taiwan_quant.data.loader import load_chips
    from taiwan_quant.features.chips import CHIPS_REQUIRED_COLUMNS

    chips = load_chips(["2330"], db_path=fake_chips_db)
    assert set(CHIPS_REQUIRED_COLUMNS).issubset(set(chips.columns))


@pytest.mark.unit
def test_chips_carries_price_and_volume(fake_chips_db: Path) -> None:
    """籌碼特徵需要 volume 做正規化、close 算比率，必須一併帶出"""
    from taiwan_quant.data.loader import load_chips

    chips = load_chips(["2330"], db_path=fake_chips_db)
    row = chips.loc[("2330", pd.Timestamp("2026-01-06"))]
    assert row["volume"] == pytest.approx(2000.0)
    assert row["close"] == pytest.approx(105.0)


@pytest.mark.unit
def test_chips_drops_rows_missing_institutional(fake_chips_db: Path) -> None:
    """
    籌碼資料缺漏的列必須剔除，不可補 0。

    補 0 會被誤讀為「法人當天沒買賣」，而實際是「不知道」。
    """
    from taiwan_quant.data.loader import load_chips

    con = sqlite3.connect(fake_chips_db)
    con.execute(
        "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)",
        ("2330", "2026-01-07", 105.0, 115.0, 95.0, 110.0, 3000),
    )
    con.commit()
    con.close()

    chips = load_chips(["2330"], db_path=fake_chips_db)
    assert pd.Timestamp("2026-01-07") not in chips.loc["2330"].index


@pytest.mark.integration
@requires_upstream
def test_real_chips_load() -> None:
    """真實資料載入後可直接餵給 build_chips"""
    from taiwan_quant.data.loader import load_chips
    from taiwan_quant.features.chips import build_chips

    chips = load_chips(["2330"], start=date(2023, 1, 1), end=date(2026, 9, 11))
    bars = chips.xs("2330", level="stock_id")

    assert len(bars) > 800
    frame = build_chips(bars)
    assert frame.iloc[-1].notna().all()


# ══════════════════════════════════════════════════════════════
# 還原因子的缺漏處理
#
# 實測 bug：0050 在 2025-06-18 做了 1:4 分割（188.65 → 47.57）。
# 還原價正確處理了它，但有 11 天缺還原價。原本的做法是「缺值時
# 保留原始價」，於是那些天的價格留在**分割前的尺度**：
#
#     2021-04-06  原始 137.65（保留）
#     鄰近日       還原後約 33.5
#     → 單日 4 倍尖峰 → 假回撤 84.20%
#
# 因子在兩次公司行為之間是常數，所以正確做法是**補因子**、不是補價格。
# ══════════════════════════════════════════════════════════════


def _mixed_adjustment_frame() -> pd.DataFrame:
    """
    構造：五天，中間一天缺 adj_close，且期間有一次 4:1 分割。

        日期        close    adj_close   說明
        day 1       400.0    97.0        分割前
        day 2       404.0    NaN         ← 缺值
        day 3       101.0    98.0        分割後
        day 4       102.0    99.0
        day 5       103.0    NaN         ← 末端缺值
    """
    index = pd.date_range("2025-06-16", periods=5, freq="B")
    return pd.DataFrame(
        {
            "stock_id": ["0050"] * 5,
            "date": index,
            "open": [400.0, 404.0, 101.0, 102.0, 103.0],
            "high": [400.0, 404.0, 101.0, 102.0, 103.0],
            "low": [400.0, 404.0, 101.0, 102.0, 103.0],
            "close": [400.0, 404.0, 101.0, 102.0, 103.0],
            "volume": [1000] * 5,
            "adj_close": [97.0, float("nan"), 98.0, 99.0, float("nan")],
        }
    )


@pytest.mark.unit
def test_missing_adjustment_does_not_create_price_spike() -> None:
    """
    缺還原價的那天不可保留原始價——那會在還原後的序列裡造成尖峰。

    day 2 的原始價 404 屬於分割前尺度；分割前的因子是 97/400 = 0.2425，
    所以還原後應該是 404 × 0.2425 ≈ 97.97，而不是 404。
    """
    result = _apply_adjustment(_mixed_adjustment_frame())

    assert result["close"].iloc[1] == pytest.approx(404.0 * 97.0 / 400.0, rel=1e-9)
    # 整段沒有任何一天偏離鄰居一個數量級
    ratios = result["close"] / result["close"].shift(1)
    assert ratios.dropna().max() < 2.0


@pytest.mark.unit
def test_trailing_missing_adjustment_uses_last_known_factor() -> None:
    """
    末端缺還原價時沿用最後一個已知因子。

    day 5 的原始價 103，最後已知因子是 99/102 = 0.97059，
    所以還原價應為 103 × 0.97059 ≈ 99.97。
    """
    result = _apply_adjustment(_mixed_adjustment_frame())
    assert result["close"].iloc[4] == pytest.approx(103.0 * 99.0 / 102.0, rel=1e-9)


@pytest.mark.unit
def test_is_adjusted_still_flags_interpolated_rows() -> None:
    """
    補過因子的列仍要標記為「非原生還原價」。

    報告必須分得出哪些列的還原價是查來的、哪些是推的。
    """
    result = _apply_adjustment(_mixed_adjustment_frame())
    assert result["is_adjusted"].tolist() == [True, False, True, True, False]


@pytest.mark.unit
def test_no_adjustment_data_at_all_keeps_raw() -> None:
    """
    整檔完全沒有還原價時保留原始價——沒有因子可以補。

    這是唯一該保留原值的情況，而且 is_adjusted 全為 False。
    """
    frame = _mixed_adjustment_frame()
    frame["adj_close"] = float("nan")

    result = _apply_adjustment(frame)
    assert result["close"].tolist() == frame["close"].tolist()
    assert not result["is_adjusted"].any()


@pytest.mark.unit
def test_adjustment_factor_filled_per_stock() -> None:
    """
    多檔混在同一個 DataFrame 時，因子不可跨標的外溢。

    A 的因子補到 B 上會讓 B 的價格整段偏掉。
    """
    a = _mixed_adjustment_frame()
    b = _mixed_adjustment_frame()
    b["stock_id"] = "2330"
    b[["open", "high", "low", "close"]] = 50.0
    b["adj_close"] = [25.0, float("nan"), 25.0, 25.0, 25.0]

    result = _apply_adjustment(pd.concat([a, b], ignore_index=True))
    tsmc = result[result.stock_id == "2330"]

    # 2330 的因子一直是 0.5，缺值那天也該是 0.5
    assert tsmc["close"].tolist() == pytest.approx([25.0] * 5)
