"""
單檔月報回補的解析測試

兩個容易靜默出錯的地方：

1. **民國年**——`112/10/02` 是 2023 年。用字串前四碼當西元年會得到 1120 年。
2. **`--` 代表當天沒成交**，不是 0。補 0 會讓報酬算出 −100%。
"""

from __future__ import annotations

import pytest

from scripts.backfill_symbol_history import BackfillError, months, parse_month


def _payload(data: list[list[str]], stat: str = "OK") -> dict:
    return {"stat": stat, "data": data}


def test_roc_year_converts_to_western():
    """112/10/02 → 2023-10-02。前四碼當西元年會得到 1120"""
    rows = parse_month(
        _payload([["112/10/02", "6,376,884", "62,615,821",
                   "9.88", "9.88", "9.80", "9.82", "-0.07", "3,370"]]),
        "00712",
    )

    assert len(rows) == 1
    assert rows[0][0] == "2023-10-02"
    assert rows[0][1] == pytest.approx(9.88)   # open
    assert rows[0][4] == pytest.approx(9.82)   # close
    assert rows[0][5] == 6_376_884             # volume


def test_no_trade_rows_are_dropped_not_zeroed():
    """OHLC 任一為 `--` 就剔除。補 0 會讓報酬率算出 −100%"""
    rows = parse_month(
        _payload([
            ["112/10/02", "1,000", "9,000", "9.88", "9.88", "9.80", "9.82", "0", "5"],
            ["112/10/03", "0", "0", "--", "--", "--", "--", "0", "0"],
        ]),
        "00712",
    )

    assert [r[0] for r in rows] == ["2023-10-02"]


def test_non_ok_stat_raises():
    with pytest.raises(BackfillError, match="stat"):
        parse_month(_payload([], stat="很抱歉，沒有符合條件的資料!"), "00712")


def test_malformed_date_is_skipped_without_raising():
    """壞掉的一列不該讓整個月失敗——其餘列仍要收下"""
    rows = parse_month(
        _payload([
            ["不是日期", "1,000", "9,000", "9.88", "9.88", "9.80", "9.82", "0", "5"],
            ["112/10/02", "1,000", "9,000", "9.88", "9.88", "9.80", "9.82", "0", "5"],
        ]),
        "00712",
    )

    assert len(rows) == 1


def test_months_spans_the_inclusive_range():
    assert months("2023-11", "2024-02") == ["202311", "202312", "202401", "202402"]
    assert months("2024-01", "2024-01") == ["202401"]


def test_months_rejects_a_reversed_range():
    with pytest.raises(BackfillError, match="不可晚於"):
        months("2024-05", "2024-01")


def test_months_rejects_a_bad_format():
    with pytest.raises(BackfillError, match="YYYY-MM"):
        months("2024/01", "2024-05")
