#!/usr/bin/env python3
"""
0050 / 0051 成分股解析測試

用最小化的假 Nuxt payload 驗證解析邏輯，不打網路（unit）；
另有 integration 測試打真實官網。

實測 bug（本檔 test_parses_js_leading_dot_decimal 覆蓋）：
    元大官網的權重是 JS 數字字面值 `.69`、`.65`（前導點）。
    JSON 規範不允許前導點，`json.loads(".69")` 會拋 JSONDecodeError，
    原本的 fallback 回傳字串 ".69" → 不是數值 → 權重被當成 0.0。
    結果 0050 有 37/50 檔權重為 0、合計只有 81.37%（應接近 99.7%）。
"""

from __future__ import annotations

import pytest

from taiwan_quant.data.constituents import (
    ConstituentFetchError,
    fetch_universe_constituents,
    parse_constituents,
)


def build_payload(entries: str, params: str, args: str, trade_date: str = "2026/09/11") -> str:
    """
    組一個最小可解析的 Nuxt payload。

    Nuxt 的格式是 `window.__NUXT__=(function(參數列){...}(引數列));`
    欄位值以參數名引用，真值在引數列。
    """
    return (
        f'<html><body><script>window.__NUXT__=(function({params})'
        f'{{return {{tradeDate:"{trade_date}",'
        f"FundWeights:{{StockWeights:[{entries}]}}}}}}({args}));"
        "</script></body></html>"
    )


# ══════════════════════════════════════════════════════════════
# 數字字面值解析
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parses_js_leading_dot_decimal() -> None:
    """
    JS 的前導點小數 `.69` 必須正確解析為 0.69。

    這是實測 bug：json.loads(".69") 會失敗，導致權重歸零。
    """
    payload = build_payload(
        entries=(
            "{code:a,ym:z,name:b,ename:z,weights:57.01,qty:560288622},"
            "{code:c,ym:z,name:d,ename:z,weights:.69,qty:7110990},"
            "{code:e,ym:z,name:f,ename:z,weights:.05,qty:1000}"
        ),
        params="a,b,c,d,e,f,z",
        args='"2330","台積電","6669","緯穎","2603","長榮",null',
    )
    snapshot = parse_constituents(payload, "TEST")
    by_id = {c.stock_id: c for c in snapshot.constituents}

    assert by_id["2330"].weight == pytest.approx(57.01)
    assert by_id["6669"].weight == pytest.approx(0.69), "前導點小數解析失敗"
    assert by_id["2603"].weight == pytest.approx(0.05)


@pytest.mark.unit
def test_parses_weight_from_variable_reference() -> None:
    """
    權重也可能是變數引用（`weights:nV`），必須從變數表解出。

    實測：`{code:iB,...,weights:nV,qty:406069000}` 其中 nV = 1.15
    """
    payload = build_payload(
        entries="{code:a,ym:z,name:b,ename:z,weights:nV,qty:406069000}",
        params="a,b,nV,z",
        args='"2882","國泰金",1.15,null',
    )
    snapshot = parse_constituents(payload, "TEST")
    assert snapshot.constituents[0].weight == pytest.approx(1.15)


@pytest.mark.unit
def test_parses_integer_weight() -> None:
    """整數權重（無小數點）也要能解析"""
    payload = build_payload(
        entries="{code:a,ym:z,name:b,ename:z,weights:3,qty:100}",
        params="a,b,z",
        args='"2330","台積電",null',
    )
    assert parse_constituents(payload, "TEST").constituents[0].weight == pytest.approx(3.0)


# ══════════════════════════════════════════════════════════════
# 變數表與字串切分
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_splits_args_containing_commas_inside_strings() -> None:
    """
    引數列不能用 split(",") 切——公司英文名本身含逗號。

    構造：第一個引數是 `"Yuanta Securities Co., Ltd"`，內含逗號。
    若切分錯誤，後面所有變數都會錯位，代號會解析成公司名。
    """
    payload = build_payload(
        entries="{code:b,ym:z,name:c,ename:z,weights:1.5,qty:100}",
        params="a,b,c,z",
        args='"Yuanta Securities Co., Ltd","2330","台積電",null',
    )
    snapshot = parse_constituents(payload, "TEST")
    assert snapshot.constituents[0].stock_id == "2330", "引數切分錯位"
    assert snapshot.constituents[0].name == "台積電"


@pytest.mark.unit
def test_handles_escaped_quotes_in_args() -> None:
    """引數含轉義引號時切分不可斷裂"""
    payload = build_payload(
        entries="{code:b,ym:z,name:c,ename:z,weights:1.0,qty:1}",
        params="a,b,c,z",
        args='"say \\"hi\\", ok","2317","鴻海",null',
    )
    assert parse_constituents(payload, "TEST").constituents[0].stock_id == "2317"


# ══════════════════════════════════════════════════════════════
# 過濾：只收 4 位數台股代號
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_excludes_futures_and_non_stock_codes() -> None:
    """
    期貨（TX、NYF）與非 4 位數代號必須排除。

    元大頁面同時列出「基金權重-股票」與「基金權重-期貨」。
    """
    payload = build_payload(
        entries=(
            "{code:a,ym:z,name:b,ename:z,weights:57.01,qty:100},"
            "{code:c,ym:z,name:d,ename:z,weights:.23,qty:588},"
            "{code:e,ym:z,name:f,ename:z,weights:.03,qty:738},"
            "{code:g,ym:z,name:h,ename:z,weights:1.0,qty:50}"
        ),
        params="a,b,c,d,e,f,g,h,z",
        args='"2330","台積電","TX","臺股期貨","NYF","台灣50ETF股票期貨","00631L","正2",null',
    )
    snapshot = parse_constituents(payload, "TEST")
    assert snapshot.stock_ids == ["2330"], f"未正確過濾：{snapshot.stock_ids}"


@pytest.mark.unit
def test_deduplicates_repeated_codes() -> None:
    """同一代號重複出現（頁面多處列表）只保留一筆"""
    payload = build_payload(
        entries=(
            "{code:a,ym:z,name:b,ename:z,weights:57.01,qty:100},"
            "{code:a,ym:z,name:b,ename:z,weights:57.01,qty:100}"
        ),
        params="a,b,z",
        args='"2330","台積電",null',
    )
    assert len(parse_constituents(payload, "TEST").constituents) == 1


@pytest.mark.unit
def test_sorted_by_weight_descending() -> None:
    payload = build_payload(
        entries=(
            "{code:a,ym:z,name:b,ename:z,weights:1.0,qty:1},"
            "{code:c,ym:z,name:d,ename:z,weights:5.0,qty:1},"
            "{code:e,ym:z,name:f,ename:z,weights:3.0,qty:1}"
        ),
        params="a,b,c,d,e,f,z",
        args='"1111","A","2222","B","3333","C",null',
    )
    snapshot = parse_constituents(payload, "TEST")
    assert snapshot.stock_ids == ["2222", "3333", "1111"]


# ══════════════════════════════════════════════════════════════
# 錯誤處理：拒絕半套資料
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rejects_count_far_from_expected() -> None:
    """
    0050 應為 50 檔。只解析出 1 檔時必須報錯，**不可靜默回傳半套資料**。

    官網改版會讓 regex 只命中部分項目，若不檢查筆數，
    下游會拿到一個「看起來正常但少了 49 檔」的標的池。
    """
    payload = build_payload(
        entries="{code:a,ym:z,name:b,ename:z,weights:57.01,qty:100}",
        params="a,b,z",
        args='"2330","台積電",null',
    )
    with pytest.raises(ConstituentFetchError, match="偏離超過"):
        parse_constituents(payload, "0050")


@pytest.mark.unit
def test_raises_when_no_entries_found() -> None:
    payload = build_payload(
        entries="", params="a,z", args='"x",null'
    )
    with pytest.raises(ConstituentFetchError, match="StockWeights"):
        parse_constituents(payload, "TEST")


@pytest.mark.unit
def test_raises_when_nuxt_payload_missing() -> None:
    """官網格式變更（沒有 __NUXT__）必須明確報錯"""
    with pytest.raises(ConstituentFetchError, match="__NUXT__"):
        parse_constituents("<html><body>nothing here</body></html>", "0050")


@pytest.mark.unit
def test_parses_trade_date() -> None:
    payload = build_payload(
        entries="{code:a,ym:z,name:b,ename:z,weights:1.0,qty:1}",
        params="a,b,z",
        args='"2330","台積電",null',
        trade_date="2026/09/11",
    )
    snapshot = parse_constituents(payload, "TEST")
    assert snapshot.as_of.isoformat() == "2026-09-11"


# ══════════════════════════════════════════════════════════════
# integration：打真實官網
# ══════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_real_universe_is_exactly_150_with_no_overlap() -> None:
    """
    D2 的核心假設：0050（50 檔）+ 0051（100 檔）不重疊，合計 150 檔。

    這條也驗證了使用者當初的糾正是對的——原建議文件寫「約 80 檔」有誤。
    """
    snapshots = fetch_universe_constituents()

    ids_0050 = set(snapshots["0050"].stock_ids)
    ids_0051 = set(snapshots["0051"].stock_ids)

    assert len(ids_0050) == 50, f"0050 應為 50 檔，得到 {len(ids_0050)}"
    assert len(ids_0051) == 100, f"0051 應為 100 檔，得到 {len(ids_0051)}"
    assert not (ids_0050 & ids_0051), f"0050/0051 不應重疊：{sorted(ids_0050 & ids_0051)}"
    assert len(ids_0050 | ids_0051) == 150


@pytest.mark.integration
def test_real_weights_sum_close_to_full() -> None:
    """
    權重合計應接近 100%（扣掉現金與期貨部位）。

    這條是前導點小數 bug 的迴歸測試：修正前 0050 合計只有 81.37%。
    """
    snapshots = fetch_universe_constituents()
    for etf_id, snapshot in snapshots.items():
        assert snapshot.total_weight > 90.0, (
            f"{etf_id} 權重合計僅 {snapshot.total_weight:.2f}%，疑似有權重解析失敗"
        )
        zeros = [c.stock_id for c in snapshot.constituents if c.weight == 0.0]
        assert not zeros, f"{etf_id} 有 {len(zeros)} 檔權重為 0：{zeros[:8]}"


@pytest.mark.integration
def test_real_top_holding_is_tsmc() -> None:
    """0050 權重第一應為台積電，且權重明顯偏高（指數極度集中）"""
    snapshot = fetch_universe_constituents(("0050",))["0050"]
    top = snapshot.constituents[0]
    assert top.stock_id == "2330"
    assert top.weight > 30.0, f"台積電權重 {top.weight}% 異常偏低"
