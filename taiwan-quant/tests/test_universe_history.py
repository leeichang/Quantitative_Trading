#!/usr/bin/env python3
"""
標的池歷史快照測試

核心問題（D2 / CLAUDE.md 禁令 2）：回測需要知道「在 2024-06-28 那天，
0050 的成分股是哪 50 檔」。用今日名單回溯歷史會產生 survivorship bias
——被剔除的弱股不在樣本內，績效被高估。

本模組的三條規則：

1. **絕不使用晚於決策日的快照。**用 2026 的名單回測 2024 就是 look-ahead
   兼 survivorship，是最嚴重的作弊形式之一。
2. **每個標的池都要宣告來源（provenance）。**REAL（真實快照）或
   PROXY（市值排名代理）必須明示，報告要據此標註可信度。
3. **代理重建必須納入退市股。**只用今日還活著的股票重建歷史，
   等於把已倒的公司從樣本裡刪掉。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from taiwan_quant.data.universe_history import (
    Provenance,
    SnapshotStore,
    UniverseResolution,
    build_proxy_universe,
    resolve_universe,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "universe_history")


def make_members(prefix: str, count: int) -> dict[str, list[str]]:
    """產生假成分股：{etf_id: [stock_id, ...]}"""
    return {
        "0050": [f"{prefix}{i:03d}" for i in range(count)],
        "0051": [f"{prefix}{i:03d}" for i in range(count, count * 3)],
    }


# ══════════════════════════════════════════════════════════════
# 快照存取
# ══════════════════════════════════════════════════════════════


def test_save_and_load_roundtrip(store: SnapshotStore) -> None:
    members = {"0050": ["2330", "2454"], "0051": ["6446", "3481"]}
    store.save(date(2026, 9, 12), members)

    loaded = store.load(date(2026, 9, 12))
    assert loaded == members


def test_load_missing_snapshot_returns_none(store: SnapshotStore) -> None:
    assert store.load(date(2020, 1, 1)) is None


def test_available_dates_sorted(store: SnapshotStore) -> None:
    for d in (date(2026, 6, 30), date(2026, 3, 31), date(2026, 9, 12)):
        store.save(d, {"0050": ["2330"], "0051": ["6446"]})

    assert store.available_dates() == [
        date(2026, 3, 31),
        date(2026, 6, 30),
        date(2026, 9, 12),
    ]


def test_save_rejects_empty_members(store: SnapshotStore) -> None:
    """空快照沒有意義，且會讓下游誤以為那天沒有成分股"""
    with pytest.raises(ValueError, match="不可為空"):
        store.save(date(2026, 9, 12), {"0050": [], "0051": []})


def test_save_overwrites_same_date(store: SnapshotStore) -> None:
    store.save(date(2026, 9, 12), {"0050": ["2330"], "0051": ["6446"]})
    store.save(date(2026, 9, 12), {"0050": ["2317"], "0051": ["3481"]})
    assert store.load(date(2026, 9, 12)) == {"0050": ["2317"], "0051": ["3481"]}


# ══════════════════════════════════════════════════════════════
# 時序安全：絕不使用未來快照
# ══════════════════════════════════════════════════════════════


def test_uses_most_recent_snapshot_at_or_before_date(store: SnapshotStore) -> None:
    """
    決策日 2026-08-15 應使用 2026-06-30 的快照（最近的、不晚於決策日者），
    不可用 2026-09-12 的。
    """
    store.save(date(2026, 3, 31), {"0050": ["A"], "0051": ["a"]})
    store.save(date(2026, 6, 30), {"0050": ["B"], "0051": ["b"]})
    store.save(date(2026, 9, 12), {"0050": ["C"], "0051": ["c"]})

    result = resolve_universe(date(2026, 8, 15), store=store)

    assert result.provenance is Provenance.REAL
    assert result.snapshot_date == date(2026, 6, 30)
    assert result.stock_ids == ["B", "b"]


def test_snapshot_on_exact_date_is_usable(store: SnapshotStore) -> None:
    """快照日期正好等於決策日 → 可用（當天盤後就知道成分股）"""
    store.save(date(2026, 6, 30), {"0050": ["B"], "0051": ["b"]})
    result = resolve_universe(date(2026, 6, 30), store=store)
    assert result.snapshot_date == date(2026, 6, 30)


def test_never_uses_future_snapshot(store: SnapshotStore) -> None:
    """
    決策日早於所有快照時，**不可**退而使用最早的那個未來快照。

    這是最容易寫錯的一行：`min(available_dates)` 看起來很合理，
    但那等於用未來的成分股名單回測過去。必須改走代理重建。
    """
    store.save(date(2026, 9, 12), {"0050": ["C"], "0051": ["c"]})

    result = resolve_universe(
        date(2024, 6, 28),
        store=store,
        market_caps={"1111": 900.0, "2222": 800.0},
    )

    assert result.provenance is Provenance.PROXY, "不得使用未來快照"
    assert result.snapshot_date is None
    assert "C" not in result.stock_ids


def test_falls_back_to_proxy_when_store_empty(store: SnapshotStore) -> None:
    result = resolve_universe(
        date(2024, 6, 28),
        store=store,
        market_caps={"1111": 900.0, "2222": 800.0},
    )
    assert result.provenance is Provenance.PROXY
    assert result.stock_ids == ["1111", "2222"]


# ══════════════════════════════════════════════════════════════
# Provenance 必須被明示
# ══════════════════════════════════════════════════════════════


def test_real_resolution_has_no_survivorship_warning(store: SnapshotStore) -> None:
    store.save(date(2026, 6, 30), {"0050": ["B"], "0051": ["b"]})
    result = resolve_universe(date(2026, 6, 30), store=store)
    assert result.warnings == []


def test_proxy_resolution_warns_about_survivorship(store: SnapshotStore) -> None:
    """代理重建必須揭露 survivorship bias 與「非真實成分股」兩件事"""
    result = resolve_universe(
        date(2024, 6, 28), store=store, market_caps={"1111": 900.0}
    )
    joined = " ".join(result.warnings)
    assert "survivorship" in joined
    assert "代理" in joined


def test_tier_comes_from_real_membership_not_rank(store: SnapshotStore) -> None:
    """
    有真實快照時，流動性分層要用「實際屬於哪個 ETF」判定，
    不是用市值排名猜。

    構造：'zzz' 在 0050 名單裡但排在第 3 個。若用 rank<=50 的代理規則
    也會對，所以這裡讓 0051 的成員排在前面來製造差異。
    """
    store.save(
        date(2026, 6, 30),
        {"0050": ["big1", "big2"], "0051": ["mid1", "mid2"]},
    )
    result = resolve_universe(date(2026, 6, 30), store=store)

    assert result.tier_of("big1") == "0050"
    assert result.tier_of("mid1") == "0051"


def test_proxy_tier_splits_at_rank_50(store: SnapshotStore) -> None:
    """代理模式下，前 50 名視為 0050 級、其餘 0051 級"""
    caps = {f"s{i:03d}": float(1000 - i) for i in range(60)}
    result = resolve_universe(date(2024, 6, 28), store=store, market_caps=caps)

    assert result.tier_of("s000") == "0050"
    assert result.tier_of("s049") == "0050"
    assert result.tier_of("s050") == "0051"


def test_tier_of_unknown_stock_raises(store: SnapshotStore) -> None:
    store.save(date(2026, 6, 30), {"0050": ["B"], "0051": ["b"]})
    result = resolve_universe(date(2026, 6, 30), store=store)
    with pytest.raises(KeyError):
        result.tier_of("nope")


# ══════════════════════════════════════════════════════════════
# 代理重建
# ══════════════════════════════════════════════════════════════


def test_proxy_takes_top_n_by_market_cap() -> None:
    caps = {"A": 100.0, "B": 300.0, "C": 200.0, "D": 50.0}
    proxy = build_proxy_universe(caps, top_n=3)
    assert proxy == ["B", "C", "A"]


def test_proxy_is_deterministic_on_ties() -> None:
    """
    市值相同時要有穩定排序，否則回測不可重現
    （CLAUDE.md 禁令 7、8 要求可稽核）。
    """
    caps = {"B": 100.0, "A": 100.0, "C": 100.0}
    assert build_proxy_universe(caps, top_n=3) == ["A", "B", "C"]


def test_proxy_excludes_nonpositive_market_cap() -> None:
    """市值為 0 或負（資料異常）必須排除，不可佔用名額"""
    caps = {"A": 100.0, "B": 0.0, "C": -5.0, "D": 50.0}
    assert build_proxy_universe(caps, top_n=10) == ["A", "D"]


def test_proxy_handles_fewer_stocks_than_top_n() -> None:
    caps = {"A": 100.0, "B": 50.0}
    assert build_proxy_universe(caps, top_n=150) == ["A", "B"]


def test_proxy_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="市值資料為空"):
        build_proxy_universe({}, top_n=150)


def test_proxy_includes_delisted_stocks() -> None:
    """
    退市股必須留在歷史樣本內（禁令 2）。

    只用今日還活著的股票重建歷史，等於把已倒的公司從樣本刪掉，
    回測會系統性高估績效。
    """
    caps = {"ALIVE": 100.0, "DELISTED": 200.0}
    proxy = build_proxy_universe(caps, top_n=10)
    assert "DELISTED" in proxy, "退市股被排除 → survivorship bias"
    assert proxy[0] == "DELISTED"


# ══════════════════════════════════════════════════════════════
# UniverseResolution 結構
# ══════════════════════════════════════════════════════════════


def test_resolution_is_immutable(store: SnapshotStore) -> None:
    store.save(date(2026, 6, 30), {"0050": ["B"], "0051": ["b"]})
    result = resolve_universe(date(2026, 6, 30), store=store)
    with pytest.raises(Exception):
        result.provenance = Provenance.PROXY  # type: ignore[misc]


def test_resolution_requires_market_caps_for_proxy(store: SnapshotStore) -> None:
    """沒有快照又沒給市值資料 → 明確報錯，不可回傳空池"""
    with pytest.raises(ValueError, match="market_caps"):
        resolve_universe(date(2024, 6, 28), store=store)


def test_real_resolution_preserves_etf_order(store: SnapshotStore) -> None:
    """0050 成員排在 0051 之前，方便報告與分層閱讀"""
    store.save(
        date(2026, 6, 30),
        {"0050": ["big1", "big2"], "0051": ["mid1"]},
    )
    result = resolve_universe(date(2026, 6, 30), store=store)
    assert result.stock_ids == ["big1", "big2", "mid1"]
