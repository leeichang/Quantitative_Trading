"""
歷史標的池載入測試

回補 2015 起的資料之後，標的池必須**逐季用當時的名單**，
不能再用今天的名單回溯——那正是 survivorship bias（禁令 2）。

實測資料顯示這個差距有多大：

    2015Q1 與 2026Q3 的標的池成員重疊只有 74 / 150
    曾入池後下市的有 34 檔，包含日月光（排名 9）、矽品（13）
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from taiwan_quant.data.loader import (
    DataNotAvailableError,
    load_universe_at,
)


@pytest.fixture
def history_db(tmp_path: Path) -> Path:
    """
    造一個有三個季度快照的小資料庫。

    2015Q1 有 2311（日月光，後來下市）、2026Q3 沒有——
    這正是要守的行為。
    """
    path = tmp_path / "history.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE stock_universe_history (
            as_of_date TEXT NOT NULL, stock_id TEXT NOT NULL,
            turnover REAL NOT NULL, rank INTEGER NOT NULL,
            PRIMARY KEY (as_of_date, stock_id));
        CREATE TABLE stock_master (
            stock_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '', market TEXT NOT NULL DEFAULT '',
            delisted_date TEXT);
    """)
    con.executemany(
        "INSERT INTO stock_universe_history VALUES (?, ?, ?, ?)",
        [
            ("2015-01-01", "2330", 9e9, 1),
            ("2015-01-01", "2311", 5e9, 2),   # 日月光，2018 下市
            ("2015-01-01", "9999", 1e9, 51),  # 排名 > 50 → 0051 級
            ("2018-04-01", "2330", 9e9, 1),
            ("2018-04-01", "2317", 4e9, 2),
            ("2026-07-01", "2330", 9e9, 1),
            ("2026-07-01", "2454", 6e9, 2),
        ],
    )
    con.executemany(
        "INSERT INTO stock_master VALUES (?, ?, ?, ?, ?)",
        [
            ("2330", "台積電", "半導體", "twse", None),
            ("2311", "日月光", "半導體", "delisted", "2018-04-30"),
            ("2317", "鴻海", "電子", "twse", None),
            ("2454", "聯發科", "半導體", "twse", None),
            ("9999", "小型股", "其他", "twse", None),
        ],
    )
    con.commit()
    con.close()
    return path


@pytest.mark.unit
def test_returns_snapshot_members(history_db: Path) -> None:
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert universe.stock_ids == ["2330", "2311", "9999"]


@pytest.mark.unit
def test_uses_latest_snapshot_on_or_before(history_db: Path) -> None:
    """2018-06 應取 2018-04 的快照，不是 2015-01 的"""
    universe = load_universe_at(date(2018, 6, 1), db_path=history_db)
    assert universe.stock_ids == ["2330", "2317"]


@pytest.mark.unit
def test_never_uses_future_snapshot(history_db: Path) -> None:
    """
    反 look-ahead（禁令 1）：2018-03-31 只能看到 2015-01 的快照。

    2018-04-01 的快照當天還沒產生——用它等於在 3 月就知道 4 月的名單。
    """
    universe = load_universe_at(date(2018, 3, 31), db_path=history_db)
    assert universe.stock_ids == ["2330", "2311", "9999"]


@pytest.mark.unit
def test_includes_delisted_members(history_db: Path) -> None:
    """
    2015 的名單必須含 2311（日月光），即使它 2018 就下市了。

    漏掉它就是 survivorship bias——2015 年它是真實可交易的第 2 大標的。
    """
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert "2311" in universe.stock_ids


@pytest.mark.unit
def test_snapshot_composition_changes_over_time(history_db: Path) -> None:
    early = set(load_universe_at(date(2015, 3, 15), db_path=history_db).stock_ids)
    late = set(load_universe_at(date(2026, 8, 1), db_path=history_db).stock_ids)
    assert early != late
    assert "2311" in early and "2311" not in late


@pytest.mark.unit
def test_tier_follows_rank(history_db: Path) -> None:
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert universe.tier_of("2330") == "0050"
    assert universe.tier_of("9999") == "0051"


@pytest.mark.unit
def test_warning_reports_real_snapshot(history_db: Path) -> None:
    """有歷史快照時 survivorship 警告必須消失——否則報告會謊報缺陷"""
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert universe.warning.has_historical_snapshot is True
    assert not any("survivorship" in n for n in universe.warning.describe())


@pytest.mark.unit
def test_warning_discloses_turnover_ranking(history_db: Path) -> None:
    """
    排名依據是成交金額，不是市值。

    市值需要股數，FinMind 免費層沒有。這是代理，必須講清楚，
    否則讀者會以為那是真的 0050+0051 成分股。
    """
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert any("成交金額" in note for note in universe.warning.describe())


@pytest.mark.unit
def test_respects_limit(history_db: Path) -> None:
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db, limit=2)
    assert universe.stock_ids == ["2330", "2311"]


@pytest.mark.unit
def test_raises_when_no_snapshot_before_date(history_db: Path) -> None:
    """
    早於第一個快照時**拋錯**，不可回傳最早的快照。

    回最早的等於用 2015 的名單去跑 2014——那是 look-ahead。
    """
    with pytest.raises(DataNotAvailableError, match="快照"):
        load_universe_at(date(2014, 1, 1), db_path=history_db)


@pytest.mark.unit
def test_raises_when_db_missing(tmp_path: Path) -> None:
    with pytest.raises(DataNotAvailableError):
        load_universe_at(date(2015, 3, 15), db_path=tmp_path / "nope.db")


@pytest.mark.unit
def test_carries_names_from_master(history_db: Path) -> None:
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    names = dict(zip(universe.stocks["stock_id"], universe.stocks["name"]))
    assert names["2311"] == "日月光"


# ══════════════════════════════════════════════════════════════
# 市值排名（取代成交金額）
#
# 用今日真實的 0050 + 0051 名單驗證兩種代理：
#
#     市值排名      142 / 150 = 94.7%
#     成交金額排名  109 / 150 = 72.7%
#
# 成交金額系統性漏掉大市值低週轉的傳產（中鋼、亞泥、和泰車），
# 而 0050 / 0051 是市值加權指數。
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def dual_basis_db(tmp_path: Path) -> Path:
    """同時含兩種排名的資料庫"""
    path = tmp_path / "dual.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE universe_history (
            as_of_date TEXT NOT NULL, basis TEXT NOT NULL, stock_id TEXT NOT NULL,
            metric REAL NOT NULL, rank INTEGER NOT NULL,
            PRIMARY KEY (as_of_date, basis, stock_id));
        CREATE TABLE stock_master (
            stock_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '', market TEXT NOT NULL DEFAULT '',
            delisted_date TEXT);
    """)
    con.executemany(
        "INSERT INTO universe_history VALUES (?, ?, ?, ?, ?)",
        [
            # 市值版：含低週轉的權值股 2002 中鋼
            ("2015-01-01", "market_cap", "2330", 9e12, 1),
            ("2015-01-01", "market_cap", "2002", 3e11, 2),
            ("2015-01-01", "market_cap", "9999", 1e10, 51),
            # 成交金額版：漏掉 2002，換成熱門小股
            ("2015-01-01", "turnover", "2330", 9e9, 1),
            ("2015-01-01", "turnover", "8888", 5e9, 2),
        ],
    )
    con.executemany(
        "INSERT INTO stock_master VALUES (?, ?, ?, ?, ?)",
        [("2330", "台積電", "半導體", "twse", None),
         ("2002", "中鋼", "鋼鐵", "twse", None),
         ("8888", "熱門小股", "其他", "twse", None),
         ("9999", "小型股", "其他", "twse", None)],
    )
    con.commit()
    con.close()
    return path


@pytest.mark.unit
def test_defaults_to_market_cap_basis(dual_basis_db: Path) -> None:
    """
    預設用市值排名。

    成交金額版只抓到真實成分股的 72.7%，市值版 94.7%——
    預設值選錯會讓每一次回測都用比較差的標的池。
    """
    universe = load_universe_at(date(2015, 3, 15), db_path=dual_basis_db)
    assert "2002" in universe.stock_ids


@pytest.mark.unit
def test_turnover_basis_still_reachable(dual_basis_db: Path) -> None:
    """
    舊的成交金額版要保留。

    先前的驗證結果是用它跑的，刪掉就無法重現。
    """
    universe = load_universe_at(
        date(2015, 3, 15), db_path=dual_basis_db, basis="turnover"
    )
    assert universe.stock_ids == ["2330", "8888"]


@pytest.mark.unit
def test_warning_reports_market_cap_basis(dual_basis_db: Path) -> None:
    """市值版不該再說「依成交金額排名」——那是上一版的警告"""
    universe = load_universe_at(date(2015, 3, 15), db_path=dual_basis_db)
    notes = universe.warning.describe()
    assert not any("成交金額" in n for n in notes)
    assert any("市值" in n for n in notes)


@pytest.mark.unit
def test_raises_on_unknown_basis(dual_basis_db: Path) -> None:
    with pytest.raises(DataNotAvailableError, match="basis"):
        load_universe_at(date(2015, 3, 15), db_path=dual_basis_db, basis="亂寫")


@pytest.mark.unit
def test_falls_back_to_legacy_table(history_db: Path) -> None:
    """
    只有舊表（`stock_universe_history`）的資料庫仍要能讀。

    否則 `build_universe_marketcap.py` 還沒跑過的環境會整個壞掉。
    """
    universe = load_universe_at(date(2015, 3, 15), db_path=history_db)
    assert universe.stock_ids == ["2330", "2311", "9999"]
