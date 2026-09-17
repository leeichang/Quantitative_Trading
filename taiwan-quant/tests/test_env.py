"""
`.env` 載入的測試

## 為什麼這個模組存在

實測 2026-09-17：整個 repo 裡只有 `qlib-tw-trader/src/interfaces/app.py`
呼叫 `load_dotenv()`。taiwan-quant 側直接讀 `os.environ`，所以 `.env`
**完全沒有作用**——設定看起來填好了，程式卻讀不到，而且不報錯。

`backfill_finmind_history.api_token()` 讀不到就回 `None` 走匿名層，
額度被砍但沒有任何訊息。那是最糟的失敗形態。

## 測試裡絕對不放真實值

所有測試用的值都是明顯的假值。`EnvStatus` 在型別上就不攜帶值，
這件事本身有測試釘住。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from taiwan_quant.config.env import (
    FINMIND_KEYS,
    PLACEHOLDER_VALUES,
    TELEGRAM_KEYS,
    EnvError,
    EnvStatus,
    load_env,
    require,
)

WATCHED = FINMIND_KEYS + TELEGRAM_KEYS


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每個測試都從乾淨的環境開始，不受開發機的真實設定影響"""
    for key in WATCHED:
        monkeypatch.delenv(key, raising=False)


def write_env(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════
# 載入
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_loads_keys_from_file() -> None:
    """基本行為：檔案裡的鍵要進到 os.environ"""


@pytest.mark.unit
def test_load_env_sets_missing_keys(tmp_path: Path) -> None:
    path = write_env(tmp_path, "FINMIND_API_TOKEN=fake-token-aaa\n")

    status = load_env(path)

    assert os.environ["FINMIND_API_TOKEN"] == "fake-token-aaa"
    assert "FINMIND_API_TOKEN" in status.loaded_keys
    assert status.env_file_found is True


@pytest.mark.unit
def test_shell_environment_wins_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    `override=False`：已匯出的值不被 `.env` 蓋掉。

    CI 與 `FINMIND_API_TOKEN=xxx python scripts/...` 這種一次性覆寫
    必須有效，否則那個用法會靜默失效——又是一個「看起來設好了」的陷阱。
    """
    monkeypatch.setenv("FINMIND_API_TOKEN", "from-shell")
    path = write_env(tmp_path, "FINMIND_API_TOKEN=from-file\n")

    status = load_env(path)

    assert os.environ["FINMIND_API_TOKEN"] == "from-shell"
    assert "FINMIND_API_TOKEN" not in status.loaded_keys


@pytest.mark.unit
def test_override_true_lets_the_file_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINMIND_API_TOKEN", "from-shell")
    path = write_env(tmp_path, "FINMIND_API_TOKEN=from-file\n")

    load_env(path, override=True)

    assert os.environ["FINMIND_API_TOKEN"] == "from-file"


@pytest.mark.unit
def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """
    `.env` 不存在不該拋錯——CI 走純環境變數，沒有檔案是正常狀態。
    """
    status = load_env(tmp_path / "nope.env")

    assert status.env_file_found is False
    assert status.loaded_keys == ()
    assert set(status.missing) == set(WATCHED)


@pytest.mark.unit
def test_load_env_is_repeatable(tmp_path: Path) -> None:
    """腳本各自在啟動時呼叫一次，不可互相干擾"""
    path = write_env(tmp_path, "FINMIND_API_TOKEN=fake-token-bbb\n")

    first = load_env(path)
    second = load_env(path)

    assert "FINMIND_API_TOKEN" in first.loaded_keys
    # 第二次已經在環境裡，所以不再「載入」，但狀態仍是已設定
    assert "FINMIND_API_TOKEN" not in second.loaded_keys
    assert "FINMIND_API_TOKEN" in second.configured


# ══════════════════════════════════════════════════════════════
# 解析：擋掉常見寫法
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize(
    "line, expected",
    [
        ("FINMIND_API_TOKEN=plain", "plain"),
        ('FINMIND_API_TOKEN="double"', "double"),
        ("FINMIND_API_TOKEN='single'", "single"),
        ("export FINMIND_API_TOKEN=exported", "exported"),
        ("  FINMIND_API_TOKEN = spaced  ", "spaced"),
    ],
)
def test_parses_common_shell_styles(
    tmp_path: Path, line: str, expected: str
) -> None:
    """
    `export ` 前綴與引號是最常見的兩種寫法。不處理的話使用者會得到
    一個值為 `"token"`（含引號）的變數，然後在 HTTP 401 才發現。
    """
    load_env(write_env(tmp_path, line + "\n"))

    assert os.environ["FINMIND_API_TOKEN"] == expected


@pytest.mark.unit
def test_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        "# 這是註解\n\nFINMIND_API_TOKEN=fake-ccc\n# FINMIND_KEY=should-not-load\n",
    )

    load_env(path)

    assert os.environ["FINMIND_API_TOKEN"] == "fake-ccc"
    assert "FINMIND_KEY" not in os.environ


# ══════════════════════════════════════════════════════════════
# 佔位字串等於沒設定
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_placeholder_counts_as_unset(tmp_path: Path) -> None:
    """
    填了佔位字串等於沒填。讓 `your_token_here` 一路傳到 API 才失敗，
    錯誤訊息會指向 HTTP 401 而不是「你沒填 token」。
    """
    path = write_env(
        tmp_path,
        "FINMIND_API_TOKEN=paste_your_finmind_token_here\n"
        "TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN\n",
    )

    status = load_env(path)

    assert "FINMIND_API_TOKEN" in status.placeholders
    assert "TELEGRAM_BOT_TOKEN" in status.placeholders
    assert "FINMIND_API_TOKEN" not in status.configured


@pytest.mark.unit
def test_require_rejects_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMIND_API_TOKEN", "your_token_here")

    with pytest.raises(EnvError, match="佔位字串"):
        require("FINMIND_API_TOKEN")


@pytest.mark.unit
def test_require_rejects_missing() -> None:
    with pytest.raises(EnvError, match="FINMIND_API_TOKEN"):
        require("FINMIND_API_TOKEN")


@pytest.mark.unit
def test_require_returns_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMIND_API_TOKEN", "  fake-real-value  ")

    assert require("FINMIND_API_TOKEN") == "fake-real-value"


@pytest.mark.unit
def test_notifier_shares_the_same_placeholder_set() -> None:
    """
    `notify/telegram.py` 原本有自己的一份，而兩份不一致：它擋
    `YOUR_CHAT_ID` 與 `xxx`，env 那邊沒有。**同一個佔位字串在一處
    被擋、在另一處放行。**

    現在斷言是**同一個物件**，不只是子集——子集關係允許兩邊再次分岔。
    """
    from taiwan_quant.notify.telegram import PLACEHOLDER_VALUES as NOTIFY

    assert NOTIFY is PLACEHOLDER_VALUES


@pytest.mark.unit
@pytest.mark.parametrize(
    "value", ["YOUR_CHAT_ID", "xxx", "your_token_here", "YOUR_TELEGRAM_BOT_TOKEN"]
)
def test_previously_divergent_placeholders_are_all_covered(value: str) -> None:
    """收斂前兩份各有遺漏，逐一釘住"""
    assert value in PLACEHOLDER_VALUES


# ══════════════════════════════════════════════════════════════
# 秘密不可外洩到狀態物件
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_status_never_carries_values(tmp_path: Path) -> None:
    """
    `EnvStatus` 會被印在終端機與報告裡，所以它在型別上就不能攜帶值。

    這個測試檢查整個物件的字串表示裡不含秘密。
    """
    secret = "super-secret-fake-token-9f8e7d"
    path = write_env(tmp_path, f"FINMIND_API_TOKEN={secret}\n")

    status = load_env(path)

    assert secret not in repr(status)
    assert secret not in "\n".join(status.describe())
    assert secret not in str(status)


@pytest.mark.unit
def test_describe_flags_placeholders_loudly(tmp_path: Path) -> None:
    """摘要要讓「填了佔位字串」看得出來，不可與「已設定」混在一起"""
    path = write_env(tmp_path, "FINMIND_API_TOKEN=changeme\n")

    lines = load_env(path).describe()

    assert any("佔位字串" in line for line in lines)


# ══════════════════════════════════════════════════════════════
# 兩個 FinMind 變數名都要被監看
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_both_finmind_names_are_watched() -> None:
    """
    同一個 token 有兩個消費者、兩個變數名：

        FINMIND_KEY          qlib-tw-trader/src/services/sync_service.py
        FINMIND_API_TOKEN    taiwan-quant/scripts/backfill_finmind_history.py

    **兩個都是真的。** 只設一個會讓另一側靜默走匿名層。
    """
    assert set(FINMIND_KEYS) == {"FINMIND_KEY", "FINMIND_API_TOKEN"}


@pytest.mark.unit
def test_backfill_script_reads_the_watched_name() -> None:
    """
    回補腳本讀的變數名必須在監看清單內。改了腳本卻沒改這裡，
    狀態摘要就會漏報。
    """
    import scripts.backfill_finmind_history as backfill

    assert backfill.TOKEN_ENV in FINMIND_KEYS
