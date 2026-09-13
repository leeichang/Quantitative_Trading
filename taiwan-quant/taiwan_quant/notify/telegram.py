"""
Telegram 推播

系統定位：**只推播，不下單。**7/7 個 AI 來源一致同意這條，也是使用者
的原始需求（「傳送訊息到 telegram 給使用者就可以！不用直接下單」）。

三條硬規定：

1. **Token / Chat ID 走環境變數**，程式碼中無明文（CLAUDE.md 秘密管理）
2. **`--dry-run` 完全不發網路請求**——開發時誤發到真實頻道是不可逆的
3. **推播失敗必須拋錯**，不可靜默吞掉——使用者以為收到建議、實際沒發出去，
   比明確報錯危險得多

訊息內容要求（規格 13、15）：

    · 交易計畫（進場區間 / 目標價 / 失效價 / R:R / 部位）
    · 換手率與年化成本拖累
    · 對照組績效（買進持有 / 0050）
    · 版本號（可稽核性）
    · 不下單聲明
"""

from __future__ import annotations

from typing import Protocol

import os
from dataclasses import dataclass

import httpx

class PlanResult(Protocol):
    """
    推播格式化只需要 `describe()`。

    刻意用 Protocol 而非具體型別：triple-barrier 的 `PortfolioResult`
    與移動停損的 `TrailingPortfolioResult` 欄位完全不同（一個有目標價、
    一個有移動停損幅度），但推播的外框、版本號、免責聲明是共用的。
    綁死其中一個會逼另一個複製整套格式化邏輯。
    """

    def describe(self) -> str: ...

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

MAX_MESSAGE_LENGTH = 4096
"""Telegram 單則訊息字元上限。超長不分段會被 API 拒絕，整份建議發不出去"""

REQUEST_TIMEOUT = 30.0

PLACEHOLDER_VALUES = frozenset({
    "",
    "your_token_here",
    "YOUR_TELEGRAM_BOT_TOKEN",
    "YOUR_CHAT_ID",
    "xxx",
})
"""
`.env.example` 的佔位字串。

不擋掉的話，使用者會以為設定好了，實際推播到不存在的 bot。
"""


class TelegramError(RuntimeError):
    """推播失敗"""


@dataclass(frozen=True)
class TelegramConfig:
    """
    推播設定。

    `__repr__` 會遮蔽 token——例外訊息、traceback、除錯輸出都可能外洩。
    """

    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> TelegramConfig:
        """從環境變數讀取（唯一合法來源）"""
        return cls(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
        )

    @property
    def is_configured(self) -> bool:
        """設定是否可用（排除佔位字串）"""
        return (
            self.bot_token not in PLACEHOLDER_VALUES
            and self.chat_id not in PLACEHOLDER_VALUES
        )

    def __repr__(self) -> str:
        masked = "***" if self.bot_token else "(未設定)"
        return f"TelegramConfig(bot_token={masked}, chat_id={self.chat_id!r})"


class TelegramNotifier:
    """Telegram 推播器"""

    def __init__(self, config: TelegramConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run

    @property
    def will_send(self) -> bool:
        """是否會真的發出請求"""
        return self.config.is_configured and not self.dry_run

    def send(self, message: str) -> bool:
        """
        推播訊息。

        Args:
            message: 訊息內容（超過 4096 字元會自動分段）

        Returns:
            True 表示流程完成（dry-run 與未設定時也回 True）

        Raises:
            ValueError: 訊息為空
            TelegramError: 推播失敗

        未設定 token 時**優雅降級為只印出**，不拋錯——使用者可能只想
        看看訊息長怎樣。但真的嘗試發送而失敗時一定拋錯。
        """
        if not message.strip():
            raise ValueError("訊息不可為空")

        if not self.will_send:
            reason = "dry-run" if self.dry_run else "未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
            print(f"\n{'=' * 60}\n[{reason}，以下訊息未實際發送]\n{'=' * 60}")
            print(message)
            print("=" * 60)
            return True

        url = TELEGRAM_API.format(token=self.config.bot_token)
        for chunk in _split_message(message):
            try:
                response = httpx.post(
                    url,
                    json={"chat_id": self.config.chat_id, "text": chunk},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                # 不可把 exc 直接串進訊息——httpx 的錯誤會含完整 URL（內含 token）
                raise TelegramError(
                    f"推播失敗：{type(exc).__name__}。"
                    "請確認網路與 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
                ) from None

        return True


def _split_message(message: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """
    把長訊息切成符合 Telegram 上限的段落。

    盡量在換行處切，避免把一筆交易計畫從中間斷開。
    """
    if len(message) <= limit:
        return [message]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in message.split("\n"):
        # 單行就超長（極少見）：硬切
        while len(line) > limit:
            if current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            chunks.append(line[:limit])
            line = line[limit:]

        addition = len(line) + (1 if current else 0)
        if current_len + addition > limit:
            chunks.append("\n".join(current))
            current, current_len = [line], len(line)
        else:
            current.append(line)
            current_len += addition

    if current:
        chunks.append("\n".join(current))

    return chunks


# ══════════════════════════════════════════════════════════════
# 訊息格式
# ══════════════════════════════════════════════════════════════


def format_weekly_plan(
    result: PlanResult,
    week_id: str,
    strategy_version: str = "",
    model_version: str = "",
    benchmark_lines: list[str] | None = None,
    turnover: float | None = None,
    annual_cost_drag: float | None = None,
    data_as_of: str = "",
    universe_note: str = "",
) -> str:
    """
    組出週頻交易計畫的推播訊息。

    Args:
        result: 選股結果
        week_id: 計畫週別
        strategy_version / model_version: 版本號（可稽核性，禁令 7、8）
        benchmark_lines: 對照組績效（規格 15）
        turnover / annual_cost_drag: 換手率與成本拖累（規格 13）
        data_as_of: 資料截止日
        universe_note: 標的池來源說明（REAL / PROXY）

    Returns:
        可直接推播的文字
    """
    lines: list[str] = [f"📊 下週交易計畫 ｜ {week_id}"]

    version_bits = [v for v in (strategy_version, model_version) if v]
    if version_bits:
        lines.append("　".join(version_bits))
    if data_as_of:
        lines.append(f"資料截至 {data_as_of}")
    if universe_note:
        lines.append(universe_note)
    lines.append("")

    lines.append(result.describe())
    lines.append("")

    if turnover is not None or annual_cost_drag is not None:
        lines.append("─" * 32)
        lines.append("🔄 換手率與成本")
        if turnover is not None:
            lines.append(f"　週換手率　　{turnover * 100:.1f}%")
        if annual_cost_drag is not None:
            lines.append(f"　年化成本拖累　{annual_cost_drag * 100:.2f}%")
        lines.append("")

    if benchmark_lines:
        lines.append("─" * 32)
        lines.append("🔎 對照組（樣本外）")
        lines.extend(f"　{line}" for line in benchmark_lines)
        lines.append("")

    lines.append("─" * 32)
    lines.append("⚠️ 僅供研究，非投資建議。系統不下單。")
    lines.append("　回測績效不代表未來表現。")

    return "\n".join(lines)
