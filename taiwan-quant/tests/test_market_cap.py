#!/usr/bin/env python3
"""
歷史市值重建測試

預期值全部手算在註解裡。
"""

from __future__ import annotations

from datetime import date

import pytest

from taiwan_quant.data.market_cap import (
    DelistedStock,
    MarketCapError,
    compute_market_caps,
    fetch_delisted,
    fetch_issued_shares,
    parse_delisted,
    parse_issued_shares,
)


def qfiis_payload(rows: list[list[str]]) -> dict:
    """組一份 TWSE MI_QFIIS 格式的回應"""
    return {
        "stat": "OK",
        "date": "20240628",
        "fields": [
            "證券代號",
            "證券名稱",
            "國際證券編碼",
            "發行股數",
            "外資及陸資尚可投資股數",
        ],
        "data": rows,
    }


# ══════════════════════════════════════════════════════════════
# 發行股數解析
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parses_thousand_separated_shares() -> None:
    """
    手算：「2,224,500,000」→ 2224500000
    TWSE 的數字帶千分位，不處理會拋 ValueError。
    """
    payload = qfiis_payload([
        ["2330", "台積電", "TW0000023308", "25,930,380,458", "x"],
    ])
    assert parse_issued_shares(payload) == {"2330": 25_930_380_458}


@pytest.mark.unit
def test_excludes_non_four_digit_codes() -> None:
    """
    ETF（0050、0051）與權證等非 4 位數代號必須排除。

    實測：MI_QFIIS 的前兩列就是 0050 與 0051 本身。
    """
    payload = qfiis_payload([
        ["0050", "元大台灣50", "TW0000050004", "2,224,500,000", "x"],
        ["0051", "元大中型100", "TW0000051002", "20,000,000", "x"],
        ["2330", "台積電", "TW0000023308", "25,930,380,458", "x"],
        ["03001", "某權證", "TW00003001", "1,000", "x"],
    ])
    assert list(parse_issued_shares(payload)) == ["2330"]


@pytest.mark.unit
def test_excludes_invalid_share_values() -> None:
    """
    無效股數（`--`、空值、0、負值）必須排除。

    關鍵：`_parse_int` 對無效值回 None 而不是 0。
    回 0 會讓這些標的帶著「市值 0」進入排名，佔用名額又永遠排最後。
    """
    payload = qfiis_payload([
        ["2330", "台積電", "x", "1,000", "x"],
        ["2317", "鴻海", "x", "--", "x"],
        ["2454", "聯發科", "x", "", "x"],
        ["2308", "台達電", "x", "0", "x"],
    ])
    assert list(parse_issued_shares(payload)) == ["2330"]


@pytest.mark.unit
def test_rejects_non_ok_response() -> None:
    with pytest.raises(MarketCapError, match="非 OK"):
        parse_issued_shares({"stat": "很抱歉，沒有符合條件的資料"})


@pytest.mark.unit
def test_rejects_missing_fields() -> None:
    """
    TWSE 改欄位名時必須明確報錯。

    qlib-tw-trader 的教訓：TWSE 把 STOCK_DAY_ALL 從 JSON 改成 CSV 且
    欄位索引位移，原程式用位置索引直接壞掉。這裡用欄位名查 index，
    找不到就報錯，不猜。
    """
    bad = {"stat": "OK", "fields": ["代號", "名稱"], "data": [["2330", "台積電"]]}
    with pytest.raises(MarketCapError, match="缺少必要欄位"):
        parse_issued_shares(bad)


@pytest.mark.unit
def test_rejects_when_nothing_parsed() -> None:
    """全部被過濾掉 → 報錯而非回空字典（空字典會讓下游誤以為那天沒資料）"""
    payload = qfiis_payload([["0050", "元大台灣50", "x", "1,000", "x"]])
    with pytest.raises(MarketCapError, match="解析不到任何發行股數"):
        parse_issued_shares(payload)


@pytest.mark.unit
def test_tolerates_short_rows() -> None:
    """列長度不足時略過該列，不可 IndexError"""
    payload = qfiis_payload([
        ["2330", "台積電"],
        ["2317", "鴻海", "x", "5,000", "x"],
    ])
    assert parse_issued_shares(payload) == {"2317": 5_000}


# ══════════════════════════════════════════════════════════════
# 市值計算
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_market_cap_is_shares_times_close() -> None:
    """
    手算：
        2330  25,930,380,458 股 × 1,000 元 = 25,930,380,458,000
        2317   1,000,000 股     ×   200 元 =     200,000,000
    """
    caps = compute_market_caps(
        issued_shares={"2330": 25_930_380_458, "2317": 1_000_000},
        close_prices={"2330": 1000.0, "2317": 200.0},
    )
    assert caps["2330"] == pytest.approx(25_930_380_458_000.0)
    assert caps["2317"] == pytest.approx(200_000_000.0)


@pytest.mark.unit
def test_market_cap_requires_both_sides() -> None:
    """只有股數或只有價格的標的必須排除，不可用 0 補"""
    caps = compute_market_caps(
        issued_shares={"A": 100, "B": 100},
        close_prices={"A": 10.0, "C": 10.0},
    )
    assert list(caps) == ["A"]


@pytest.mark.unit
def test_market_cap_excludes_nonpositive_price() -> None:
    caps = compute_market_caps(
        issued_shares={"A": 100, "B": 100},
        close_prices={"A": 10.0, "B": 0.0},
    )
    assert list(caps) == ["A"]


@pytest.mark.unit
def test_market_cap_empty_input_gives_empty_output() -> None:
    """空輸入回空字典（由 build_proxy_universe 負責拒絕空池）"""
    assert compute_market_caps({}, {}) == {}


# ══════════════════════════════════════════════════════════════
# 退市清單（CLAUDE.md 禁令 2）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parses_delisted_payload() -> None:
    payload = {
        "status": 200,
        "msg": "success",
        "data": [
            {"date": "2026-09-03", "stock_id": "5371", "stock_name": "中光電"},
            {"date": "2026-09-01", "stock_id": "2867", "stock_name": "三商壽"},
        ],
    }
    result = parse_delisted(payload)
    assert result == [
        DelistedStock("5371", "中光電", date(2026, 9, 3)),
        DelistedStock("2867", "三商壽", date(2026, 9, 1)),
    ]


@pytest.mark.unit
def test_parses_delisted_skips_malformed_rows() -> None:
    """單列格式異常不可讓整批失敗"""
    payload = {
        "status": 200,
        "data": [
            {"date": "bad-date", "stock_id": "1111", "stock_name": "X"},
            {"stock_id": "2222"},
            {"date": "2026-09-01", "stock_id": "2867", "stock_name": "三商壽"},
        ],
    }
    result = parse_delisted(payload)
    assert [d.stock_id for d in result] == ["2867"]


@pytest.mark.unit
def test_parses_delisted_rejects_error_response() -> None:
    """
    FinMind 額度用盡時回 status 400 / 402，必須報錯。

    實測訊息：「Your level is free. Please update your user level.」
    以及「Requests reach the upper limit.」
    """
    payload = {"status": 400, "msg": "Your level is free. Please update your user level."}
    with pytest.raises(MarketCapError, match="FinMind 回應異常"):
        parse_delisted(payload)


# ══════════════════════════════════════════════════════════════
# integration：打真實來源
# ══════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_real_issued_shares_historical_date() -> None:
    """
    TWSE MI_QFIIS 支援歷史日期——這是能重建歷史市值的關鍵。

    實測 2024-06-28 可取得 1,223 列（含 ETF 等），過濾後剩台股個股。
    """
    shares = fetch_issued_shares(date(2024, 6, 28))
    assert len(shares) > 800, f"只取得 {len(shares)} 檔，疑似 TWSE 回應異常"
    assert "2330" in shares
    assert shares["2330"] > 20_000_000_000, "台積電發行股數應超過 200 億股"
    assert all(v > 0 for v in shares.values())
    assert all(len(k) == 4 and k.isdigit() for k in shares)


@pytest.mark.integration
def test_real_delisted_list() -> None:
    """FinMind TaiwanStockDelisting 免費方案可用"""
    result = fetch_delisted(date(2026, 1, 1), date(2026, 9, 11))
    assert all(date(2026, 1, 1) <= d.delisted_on <= date(2026, 9, 11) for d in result)
    assert all(d.stock_id for d in result)
