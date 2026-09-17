"""
`.env` 載入：讓設定檔真的生效

## 為什麼需要這個

實測 2026-09-17：整個 repo 裡**只有** `qlib-tw-trader/src/interfaces/app.py`
呼叫 `load_dotenv()`。taiwan-quant 的腳本與 `notify/telegram.py` 都直接讀
`os.environ`，所以 `.env` 對這一側**完全沒有作用**——設定看起來填好了，
程式卻讀不到。

那是最糟的一種失敗：不報錯，只是安靜地走沒有 token 的路徑。
`backfill_finmind_history.api_token()` 讀不到就回 `None` 並走匿名層，
額度被砍但不會有任何訊息。

## ⚠️ 同一個 FinMind token 有兩個變數名

```
FINMIND_KEY          qlib-tw-trader/src/services/sync_service.py
                     qlib-tw-trader/src/interfaces/routers/datasets.py
FINMIND_API_TOKEN    taiwan-quant/scripts/backfill_finmind_history.py
                     taiwan-quant/scripts/backfill_adj_from_dividends.py
```

**兩個都是真的、都在用。** `.env` 必須同時設定兩個（同一個值）。

統一成一個名字會更乾淨，但 `FINMIND_KEY` 那兩處在
`qlib-tw-trader/src/services/` 底下，而 CLAUDE.md 寫明「不重寫資料同步層
——那部分已驗證可用」。所以這裡選擇容忍重複並把它寫清楚，而不是動那一層。

## 環境變數優先於 `.env`

`override=False`：已經在 shell 匯出的值不會被 `.env` 蓋掉。

理由是 CI 與一次性覆寫要能贏——`FINMIND_API_TOKEN=xxx python scripts/...`
必須有效，否則那個用法會靜默失效，又是一個「看起來設好了」的陷阱。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""`taiwan-quant/` 目錄"""

DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

FINMIND_KEYS: tuple[str, ...] = ("FINMIND_API_TOKEN", "FINMIND_KEY")
"""
同一個 FinMind token 的兩個變數名。見模組說明。

順序有意義：`FINMIND_API_TOKEN` 是 taiwan-quant 自己的腳本在讀的，
排前面。
"""

TELEGRAM_KEYS: tuple[str, ...] = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

PLACEHOLDER_VALUES = frozenset({
    "",
    # 通用
    "your", "your_token", "your_token_here", "changeme", "none", "xxx",
    # FinMind
    "paste_your_finmind_token_here",
    # Telegram（原本散在 notify/telegram.py，已收斂到這裡）
    "YOUR_TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_CHAT_ID", "YOUR_CHAT_ID",
})
"""
佔位字串的**唯一來源**。填了佔位字串等於沒填，不可讓它一路走到 API
呼叫才失敗——那時錯誤訊息會指向 HTTP 401，而不是「你沒填 token」。

## 為什麼集中在這裡

`notify/telegram.py` 原本有自己的一份，而兩份不一致：它擋
`YOUR_CHAT_ID` 與 `xxx`，這裡沒有。**同一個佔位字串在一處被擋、
在另一處放行**——這是測試抓出來的。

現在 `telegram.py` 從這裡匯入。分層方向也對：`config` 在下層，
`notify` 在上層。
"""


class EnvError(RuntimeError):
    """環境設定不合法"""


@dataclass(frozen=True)
class EnvStatus:
    """
    設定狀態。**只帶鍵名與布林值，永遠不帶值。**

    這個類別會被印在終端機與報告裡，所以它在型別上就不能攜帶秘密。
    """

    env_path: Path
    env_file_found: bool
    loaded_keys: tuple[str, ...]
    """從 `.env` 實際載入（且原本不在環境中）的鍵名"""

    configured: tuple[str, ...]
    """有真實值（非佔位字串）的鍵名"""

    placeholders: tuple[str, ...]
    """仍是佔位字串的鍵名"""

    missing: tuple[str, ...]
    """完全沒設定的鍵名"""

    def describe(self) -> list[str]:
        """人可讀的摘要，供腳本啟動時印出"""
        notes = [
            f".env {'已載入' if self.env_file_found else '不存在'}：{self.env_path}"
        ]
        if self.configured:
            notes.append(f"已設定：{', '.join(self.configured)}")
        if self.placeholders:
            notes.append(
                f"⚠️ 仍是佔位字串（等於沒設定）：{', '.join(self.placeholders)}"
            )
        if self.missing:
            notes.append(f"未設定：{', '.join(self.missing)}")
        return notes


def _parse(text: str) -> dict[str, str]:
    """
    最小的 `.env` 解析：`KEY=VALUE`，`#` 開頭為註解。

    不引進 python-dotenv 的理由：它只裝在 `qlib-tw-trader/.venv`，
    taiwan-quant 的 venv 沒有。為了讀一個 `KEY=VALUE` 檔案加一個依賴
    不划算，而且自己解析可以順便擋掉 `export ` 前綴與引號這兩個常見寫法。
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(
    env_path: Path = DEFAULT_ENV_PATH, *, override: bool = False
) -> EnvStatus:
    """
    載入 `.env` 並回報狀態。

    Args:
        env_path: `.env` 路徑
        override: `True` 時 `.env` 蓋掉既有環境變數。**預設 False**

    Returns:
        `EnvStatus`（只含鍵名，不含值）

    Raises:
        EnvError: `.env` 存在但無法讀取

    `override=False` 讓 shell 匯出的值優先——CI 與
    `FINMIND_API_TOKEN=xxx python scripts/...` 這種一次性覆寫必須有效，
    否則會靜默失效。

    **可重複呼叫。** 腳本各自在啟動時呼叫一次不會互相干擾。
    """
    found = env_path.is_file()
    loaded: list[str] = []
    if found:
        try:
            text = env_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EnvError(f"無法讀取 {env_path}：{exc}") from exc
        for key, value in _parse(text).items():
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)

    watched = FINMIND_KEYS + TELEGRAM_KEYS
    configured, placeholders, missing = [], [], []
    for key in watched:
        if key not in os.environ:
            missing.append(key)
        elif os.environ[key].strip() in PLACEHOLDER_VALUES:
            placeholders.append(key)
        else:
            configured.append(key)

    return EnvStatus(
        env_path=env_path,
        env_file_found=found,
        loaded_keys=tuple(loaded),
        configured=tuple(configured),
        placeholders=tuple(placeholders),
        missing=tuple(missing),
    )


def require(key: str) -> str:
    """
    取一個必填的環境變數。

    Args:
        key: 變數名

    Returns:
        值

    Raises:
        EnvError: 未設定或仍是佔位字串

    **佔位字串視為未設定。** 讓 `your_token_here` 一路傳到 API 呼叫才
    失敗，錯誤訊息會指向 HTTP 401 而不是「你沒填 token」。
    """
    value = os.environ.get(key, "").strip()
    if value in PLACEHOLDER_VALUES:
        raise EnvError(
            f"{key} 未設定或仍是佔位字串。請在 "
            f"{DEFAULT_ENV_PATH} 填入真實值——"
            "秘密只走環境變數或 .env，不可硬編碼進程式"
        )
    return value
