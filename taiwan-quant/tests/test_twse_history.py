"""
TWSE 歷史日報解析測試

解析錯誤不會拋例外，只會產生看起來正常的壞資料：
`"--"` 被當成 0、千分位逗號被截斷、權證混進股票池。

所以每個欄位都比對手算值，並用真實回應的結構當樣本。
"""

from __future__ import annotations

import pytest

from taiwan_quant.data.twse_history import (
    TwseParseError,
    is_etf,
    is_ordinary_security,
    parse_mi_index,
    parse_mi_margn,
    parse_number,
    parse_t86,
    roc_date_to_iso,
)

# ── 真實回應的縮樣（欄位順序與 2015-01-05 完全相同） ──

MI_INDEX_FIELDS = [
    "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
    "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差",
    "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比",
]

MI_INDEX_PAYLOAD = {
    "stat": "OK",
    "tables": [
        {"title": "價格指數", "fields": ["指數", "收盤"], "data": [["發行量加權股價指數", "9274.11"]]},
        {"title": "104年01月05日 每日收盤行情(全部(不含權證、牛熊證、可展延牛熊證))",
         "fields": MI_INDEX_FIELDS,
         "data": [
             ["0050", "元大台灣50", "6,295,612", "1,528", "417,637,764",
              "66.40", "66.75", "66.00", "66.55",
              "<p style= color:green>-</p>", "0.30", "66.55", "20", "66.60", "17", "0.00"],
             ["2330", "台積電", "32,046,000", "12,345", "4,500,000,000",
              "140.50", "140.50", "137.50", "139.50",
              "<p style= color:green>-</p>", "1.00", "139.50", "100", "140.00", "50", "13.50"],
             ["0081", "恒香港", "0", "0", "0",
              "--", "--", "--", "--", "", "0.00", "--", "0", "--", "0", "0.00"],
             ["00631L", "元大台灣50正2", "1,000", "5", "25,000",
              "25.00", "25.10", "24.90", "25.00",
              "<p style= color:red>+</p>", "0.10", "25.00", "1", "25.05", "1", "0.00"],
         ]},
    ],
}

T86_PAYLOAD = {
    "stat": "OK",
    "fields": [
        "證券代號", "證券名稱", "外資買進股數", "外資賣出股數", "外資買賣超股數",
        "投信買進股數", "投信賣出股數", "投信買賣超股數",
        "自營商買賣超股數", "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
        "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)", "自營商賣出股數(避險)",
        "自營商買賣超股數(避險)", "三大法人買賣超股數",
    ],
    "data": [
        ["2330", "台積電          ", "26,561,027", "15,956,027", "10,605,000",
         "10,588,000", "0", "10,588,000", "30,944,000",
         "4,200,000", "335,000", "3,865,000",
         "28,714,000", "1,635,000", "27,079,000", "52,137,000"],
        ["031234", "某某權證        ", "1,000", "0", "1,000",
         "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1,000"],
    ],
}

MI_MARGN_PAYLOAD = {
    "stat": "OK",
    "tables": [
        {"title": "104年01月05日 信用交易統計", "fields": ["項目"], "data": [["融資(交易單位)"]]},
        {"title": "104年01月05日 融資融券彙總 (全部)",
         "fields": [
             "代號", "名稱",
             "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
             "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額",
             "資券互抵", "註記",
         ],
         "data": [
             ["2330", "台積電", "1,200", "800", "50", "10,000", "10,350", "999,999",
              "300", "500", "100", "2,000", "2,100", "999,999", "20", ""],
             ["0081", "恒香港", "0", "0", "0", "0", "0", "0",
              "0", "0", "0", "0", "0", "0", "0", ""],
         ]},
    ],
}


# ══════════════════════════════════════════════════════════════
# 數字解析
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parse_number_strips_thousands_separator() -> None:
    assert parse_number("6,295,612") == 6_295_612.0
    assert parse_number("417,637,764") == 417_637_764.0


@pytest.mark.unit
def test_parse_number_handles_plain_decimal() -> None:
    assert parse_number("66.40") == pytest.approx(66.40)


@pytest.mark.unit
def test_parse_number_returns_none_for_no_trade_markers() -> None:
    """
    `--` 代表當天沒有成交，**不是 0**。

    當成 0 會讓那天變成「跌到零」，報酬率算出 −100%。
    """
    for marker in ("--", "---", "", " ", "N/A"):
        assert parse_number(marker) is None


@pytest.mark.unit
def test_parse_number_returns_none_for_garbage() -> None:
    assert parse_number("<p>-</p>") is None
    assert parse_number("停止買賣") is None


@pytest.mark.unit
def test_parse_number_handles_leading_plus() -> None:
    assert parse_number("+1.05") == pytest.approx(1.05)


# ══════════════════════════════════════════════════════════════
# 代號分類
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_ordinary_security_accepts_four_digit_codes() -> None:
    assert is_ordinary_security("2330")
    assert is_ordinary_security("0050")


@pytest.mark.unit
def test_ordinary_security_rejects_warrants() -> None:
    """
    權證是 6 位數。T86 一天有 6,880 個代號，其中絕大多數是權證——
    混進股票池會讓標的池膨脹數倍，而且權證的籌碼資料沒有可比性。
    """
    assert not is_ordinary_security("031234")
    assert not is_ordinary_security("01001T")


@pytest.mark.unit
def test_etf_detection() -> None:
    """台股 ETF 代號以 00 開頭"""
    assert is_etf("0050")
    assert is_etf("0056")
    assert not is_etf("2330")
    assert not is_etf("1101")


# ══════════════════════════════════════════════════════════════
# 民國日期
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_roc_date_to_iso() -> None:
    assert roc_date_to_iso("104年01月05日") == "2015-01-05"
    assert roc_date_to_iso("115年09月11日") == "2026-09-11"


@pytest.mark.unit
def test_roc_date_to_iso_rejects_bad_format() -> None:
    assert roc_date_to_iso("2015-01-05") is None
    assert roc_date_to_iso("") is None


# ══════════════════════════════════════════════════════════════
# MI_INDEX（每日收盤行情）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parse_mi_index_matches_hand_values() -> None:
    quotes = parse_mi_index(MI_INDEX_PAYLOAD)
    by_id = {q.stock_id: q for q in quotes}

    tsmc = by_id["2330"]
    assert tsmc.open == pytest.approx(140.50)
    assert tsmc.high == pytest.approx(140.50)
    assert tsmc.low == pytest.approx(137.50)
    assert tsmc.close == pytest.approx(139.50)
    assert tsmc.volume == 32_046_000
    assert tsmc.turnover == pytest.approx(4_500_000_000)


@pytest.mark.unit
def test_parse_mi_index_keeps_etfs() -> None:
    """
    0050 必須留下——CLAUDE.md 要求的「0050 買進持有」對照組
    就靠它，上游資料庫至今沒有 ETF。
    """
    ids = {q.stock_id for q in parse_mi_index(MI_INDEX_PAYLOAD)}
    assert "0050" in ids


@pytest.mark.unit
def test_parse_mi_index_drops_no_trade_rows() -> None:
    """
    OHLC 為 `--` 的列必須剔除，不可補 0。

    0081 當天成交股數為 0、四個價格都是 `--`。
    """
    ids = {q.stock_id for q in parse_mi_index(MI_INDEX_PAYLOAD)}
    assert "0081" not in ids


@pytest.mark.unit
def test_parse_mi_index_drops_leveraged_etfs() -> None:
    """00631L 是 6 位數槓桿型，不納入一般標的池"""
    ids = {q.stock_id for q in parse_mi_index(MI_INDEX_PAYLOAD)}
    assert "00631L" not in ids


@pytest.mark.unit
def test_parse_mi_index_extracts_date() -> None:
    quotes = parse_mi_index(MI_INDEX_PAYLOAD)
    assert all(q.date == "2015-01-05" for q in quotes)


@pytest.mark.unit
def test_parse_mi_index_rejects_missing_quote_table() -> None:
    """
    找不到行情表時**拋錯**，不可回空清單。

    回空清單會讓 backfill 把那天標記成「已完成、沒有資料」，
    之後不會重試——資料就永久缺一天且無人察覺。
    """
    with pytest.raises(TwseParseError, match="每日收盤行情"):
        parse_mi_index({"stat": "OK", "tables": [{"title": "價格指數", "data": []}]})


@pytest.mark.unit
def test_parse_mi_index_rejects_non_ok_stat() -> None:
    with pytest.raises(TwseParseError, match="stat"):
        parse_mi_index({"stat": "很抱歉，沒有符合條件的資料!", "tables": []})


# ══════════════════════════════════════════════════════════════
# T86（三大法人）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parse_t86_matches_hand_values() -> None:
    """
    自營商買賣超要合併自行買賣與避險兩欄——
    既有 schema 的 dealer_buy/dealer_sell 是合併值。

    手算：買 4,200,000 + 28,714,000 = 32,914,000
          賣   335,000 +  1,635,000 =  1,970,000
    """
    rows = parse_t86(T86_PAYLOAD, trade_date="2015-01-05")
    tsmc = next(r for r in rows if r.stock_id == "2330")

    assert tsmc.foreign_buy == 26_561_027
    assert tsmc.foreign_sell == 15_956_027
    assert tsmc.trust_buy == 10_588_000
    assert tsmc.trust_sell == 0
    assert tsmc.dealer_buy == 32_914_000
    assert tsmc.dealer_sell == 1_970_000


@pytest.mark.unit
def test_parse_t86_drops_warrants() -> None:
    ids = {r.stock_id for r in parse_t86(T86_PAYLOAD, trade_date="2015-01-05")}
    assert "031234" not in ids
    assert ids == {"2330"}


@pytest.mark.unit
def test_parse_t86_rejects_non_ok_stat() -> None:
    with pytest.raises(TwseParseError, match="stat"):
        parse_t86({"stat": "查無資料", "data": []}, trade_date="2015-01-05")


T86_PAYLOAD_2018 = {
    "stat": "OK",
    "fields": [
        "證券代號", "證券名稱",
        "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
        "外陸資買賣超股數(不含外資自營商)",
        "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
        "投信買進股數", "投信賣出股數", "投信買賣超股數",
        "自營商買賣超股數",
        "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
        "自營商買賣超股數(自行買賣)",
        "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
        "三大法人買賣超股數",
    ],
    "data": [
        ["2330", "台積電          ",
         "13,430,460", "21,022,009", "-7,591,549",      # 外陸資
         "0", "0", "0",                                   # 外資自營商
         "676,140", "84,010", "592,130",                  # 投信
         "-249,720",                                      # 自營商合計
         "419,010", "1,192,130", "-773,120",              # 自行買賣
         "133,011", "-390,389", "523,400",                # 避險
         "-7,249,139"],
    ],
}


@pytest.mark.unit
def test_parse_t86_handles_2018_nineteen_field_layout() -> None:
    """
    2018 起 T86 從 16 欄變成 19 欄（外資被拆成外陸資 + 外資自營商）。

    **這是實測抓到的 regression**：原本用 2015 的固定索引去解析 2024 的
    回應，外資碰巧對、投信與自營商全錯——97 檔裡只有 1 檔吻合，
    而且不會拋任何錯。所以必須依欄位名稱取值。

    手算（2024-01-03 台積電，比對資料庫既有值）：
        foreign_buy  13,430,460    外陸資，不含外資自營商
        trust_buy       676,140
        dealer_buy      419,010 + 133,011 = 552,021   自行買賣 + 避險
    """
    rows = parse_t86(T86_PAYLOAD_2018, trade_date="2024-01-03")
    tsmc = next(r for r in rows if r.stock_id == "2330")

    assert tsmc.foreign_buy == 13_430_460
    assert tsmc.foreign_sell == 21_022_009
    assert tsmc.trust_buy == 676_140
    assert tsmc.trust_sell == 84_010
    assert tsmc.dealer_buy == 552_021
    assert tsmc.dealer_sell == 1_192_130 + (-390_389)


@pytest.mark.unit
def test_parse_t86_rejects_unknown_field_layout() -> None:
    """
    欄位結構不認得時**拋錯**，不可猜位置。

    猜錯不會拋例外，只會產生看起來正常的壞資料。
    """
    with pytest.raises(TwseParseError, match="找不到欄位"):
        parse_t86(
            {"stat": "OK", "fields": ["證券代號", "證券名稱", "某個新欄位"],
             "data": [["2330", "台積電", "1"]]},
            trade_date="2024-01-03",
        )


@pytest.mark.unit
def test_parse_t86_rejects_missing_fields_key() -> None:
    with pytest.raises(TwseParseError, match="fields"):
        parse_t86({"stat": "OK", "data": []}, trade_date="2024-01-03")


# ══════════════════════════════════════════════════════════════
# MI_MARGN（融資融券）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_parse_mi_margn_matches_hand_values() -> None:
    rows = parse_mi_margn(MI_MARGN_PAYLOAD, trade_date="2015-01-05")
    tsmc = next(r for r in rows if r.stock_id == "2330")

    assert tsmc.margin_buy == 1_200
    assert tsmc.margin_sell == 800
    assert tsmc.margin_balance == 10_350
    assert tsmc.short_buy == 300
    assert tsmc.short_sell == 500
    assert tsmc.short_balance == 2_100


@pytest.mark.unit
def test_parse_mi_margn_keeps_zero_balance_rows() -> None:
    """
    融資餘額 0 是真實資訊（沒人融資），與「沒有資料」不同——
    不可比照 MI_INDEX 的 `--` 剔除。
    """
    ids = {r.stock_id for r in parse_mi_margn(MI_MARGN_PAYLOAD, trade_date="2015-01-05")}
    assert "0081" in ids


@pytest.mark.unit
def test_parse_mi_margn_rejects_changed_field_layout() -> None:
    """
    MI_MARGN 的欄位名稱有重複（`買進` 出現兩次），只能用位置索引。

    代價是結構一變就會取到錯的區塊，所以解析前先比對欄位表。
    """
    bad = {
        "stat": "OK",
        "tables": [{
            "title": "融資融券彙總 (全部)",
            "fields": ["代號", "名稱", "買進"],
            "data": [["2330", "台積電", "1"]],
        }],
    }
    with pytest.raises(TwseParseError, match="欄位結構"):
        parse_mi_margn(bad, trade_date="2015-01-05")


@pytest.mark.unit
def test_parse_mi_margn_rejects_missing_table() -> None:
    with pytest.raises(TwseParseError, match="融資融券"):
        parse_mi_margn(
            {"stat": "OK", "tables": [{"title": "信用交易統計", "data": []}]},
            trade_date="2015-01-05",
        )
