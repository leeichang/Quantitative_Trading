#!/usr/bin/env python3
"""
Telegram 推播測試

D4 的推播格式 + CLAUDE.md 的三條硬規定：

    · 只推播，不下單（7/7 來源一致同意）
    · Token / Chat ID 走環境變數，程式碼中無明文
    · 報告必須含換手率、成本情境、對照組（規格 13、15）

推播訊息本身也是可稽核的一環：帶版本號才能反查
「為什麼 2026-09-12 系統推薦這支股票」。
"""

from __future__ import annotations

import httpx
import pytest

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.notify.telegram import (
    TelegramConfig,
    TelegramError,
    TelegramNotifier,
    format_weekly_plan,
)
from taiwan_quant.ranking.portfolio import Candidate, select_portfolio

pytestmark = pytest.mark.unit

CAPITAL = 400_000.0


def cand(
    stock_id: str = "2330",
    prob_up: float = 0.62,
    target_pct: float = 0.062,
    stop_pct: float = 0.0238,
    entry_price: float = 2430.0,
    industry: str = "半導體",
    beta: float = 0.9,
    tier: str = "0050",
) -> Candidate:
    return Candidate(
        stock_id=stock_id,
        prob_up=prob_up,
        target_pct=target_pct,
        stop_pct=stop_pct,
        entry_price=entry_price,
        tier=Tier(tier),
        industry=industry,
        volatility_pct=0.5,
        beta=beta,
    )


def sample_result():
    candidates = [
        cand("2330", prob_up=0.62, industry="半導體", beta=0.95),
        cand("2882", prob_up=0.58, entry_price=95.0, industry="金融", beta=0.7),
    ]
    return select_portfolio(
        candidates, CAPITAL, correlations={("2330", "2882"): 0.25}
    )


# ══════════════════════════════════════════════════════════════
# 設定：秘密管理
# ══════════════════════════════════════════════════════════════


def test_config_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token 與 Chat ID 只能來自環境變數（CLAUDE.md 秘密管理）"""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    config = TelegramConfig.from_env()
    assert config.bot_token == "test-token"
    assert config.chat_id == "12345"
    assert config.is_configured


def test_config_not_configured_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    config = TelegramConfig.from_env()
    assert not config.is_configured


def test_config_rejects_placeholder_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `.env.example` 的佔位字串不可被當成真 token。

    否則使用者會以為設定好了，實際上推播到不存在的 bot。
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "your_token_here")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

    assert not TelegramConfig.from_env().is_configured


def test_config_repr_masks_token() -> None:
    """
    token 不可出現在 repr / log。

    例外訊息、traceback、除錯輸出都可能外洩。
    """
    config = TelegramConfig(bot_token="secret-token-12345", chat_id="999")
    assert "secret-token-12345" not in repr(config)
    assert "***" in repr(config)


# ══════════════════════════════════════════════════════════════
# 訊息格式（D4）
# ══════════════════════════════════════════════════════════════


def test_message_contains_trade_plan_fields() -> None:
    """進場區間 / 目標價 / 失效價 / R:R 缺一不可"""
    message = format_weekly_plan(sample_result(), week_id="2026W38")
    for field in ("進場區間", "目標價", "失效價", "R:R"):
        assert field in message, f"缺少 {field}"


def test_message_contains_position_sizing() -> None:
    """D4：部位用「佔總資金 %」+ 零股股數表達"""
    message = format_weekly_plan(sample_result(), week_id="2026W38")
    assert "資金" in message
    assert "股" in message
    assert "零股" in message


def test_message_contains_version_for_audit() -> None:
    """
    帶版本號才能反查「為什麼當時推薦這支」（禁令 7、8）。
    """
    message = format_weekly_plan(
        sample_result(),
        week_id="2026W38",
        strategy_version="momentum_tb_5d@a3f9c1",
        model_version="v0.3.1",
    )
    assert "momentum_tb_5d@a3f9c1" in message
    assert "v0.3.1" in message


def test_message_contains_no_trading_disclaimer() -> None:
    """
    必須聲明「不下單、僅供研究」。

    7/7 來源一致同意這條，是整個系統的定位。
    """
    message = format_weekly_plan(sample_result(), week_id="2026W38")
    assert "不下單" in message or "不執行交易" in message
    assert "研究" in message


def test_message_includes_benchmark_comparison() -> None:
    """
    規格 13、15：報告必須並列對照組與成本情境。

    Kimi 的意見：「若三策略近五年都跑輸買進持有，就代表短打策略沒有優勢」。
    看不到對照組就無法做這個判斷。
    """
    message = format_weekly_plan(
        sample_result(),
        week_id="2026W38",
        benchmark_lines=[
            "本策略    OOS CAGR 14.2%｜Sharpe 0.91｜MaxDD −18.3%",
            "買進持有  OOS CAGR 21.5%｜Sharpe 1.12｜MaxDD −24.1%",
            "0050      OOS CAGR 12.8%｜Sharpe 0.95｜MaxDD −19.7%",
        ],
        turnover=0.20,
        annual_cost_drag=0.111,
    )
    assert "買進持有" in message
    assert "0050" in message
    assert "換手率" in message
    assert "成本拖累" in message


def test_message_surfaces_warnings() -> None:
    """
    投組警告（例如沒有 defensive）必須出現在推播裡，不可只留在 log。
    """
    candidates = [
        cand("2330", prob_up=0.62, industry="半導體", beta=1.5),
        cand("3231", prob_up=0.58, entry_price=95.0, industry="電腦", beta=1.4),
    ]
    result = select_portfolio(
        candidates, CAPITAL, correlations={("2330", "3231"): 0.25}
    )
    message = format_weekly_plan(result, week_id="2026W38")
    assert "defensive" in message


def test_message_when_no_positions() -> None:
    """
    「這週沒有值得買的」是合法且重要的結論，必須明確推播。

    靜默不發會讓使用者以為系統壞了。
    """
    result = select_portfolio([], CAPITAL, correlations={})
    message = format_weekly_plan(result, week_id="2026W38")
    assert "無符合條件" in message or "不推播" in message


def test_message_includes_week_id() -> None:
    assert "2026W38" in format_weekly_plan(sample_result(), week_id="2026W38")


def test_message_never_contains_order_instructions() -> None:
    """
    訊息不可含任何下單指令字樣。

    系統定位是研究工具，措辭上也不該暗示執行。
    """
    message = format_weekly_plan(sample_result(), week_id="2026W38")
    for forbidden in ("立即下單", "馬上買進", "掛單", "委託"):
        assert forbidden not in message


# ══════════════════════════════════════════════════════════════
# 推播行為
# ══════════════════════════════════════════════════════════════


def test_dry_run_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `--dry-run` 必須完全不發出網路請求。

    開發時誤發訊息到真實頻道是不可逆的。
    """
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run 不應發出任何 HTTP 請求")

    monkeypatch.setattr(httpx, "post", explode)

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="t", chat_id="c"), dry_run=True
    )
    assert notifier.send("測試訊息") is True


def test_unconfigured_falls_back_to_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    未設定 token 時優雅降級為只印出，不可拋錯。

    使用者可能只想看看訊息長怎樣。
    """
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("未設定時不應發出 HTTP 請求")

    monkeypatch.setattr(httpx, "post", explode)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    notifier = TelegramNotifier(TelegramConfig.from_env())
    assert notifier.send("測試訊息") is True


def test_send_posts_to_telegram_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    notifier = TelegramNotifier(TelegramConfig(bot_token="tok", chat_id="chat"))
    assert notifier.send("hello") is True

    assert "tok" in str(captured["url"])
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["chat_id"] == "chat"
    assert payload["text"] == "hello"


def test_send_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    推播失敗必須拋錯，不可靜默吞掉。

    使用者以為收到了建議、實際沒發出去，比明確報錯危險得多。
    """
    def fake_post(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "post", fake_post)

    notifier = TelegramNotifier(TelegramConfig(bot_token="tok", chat_id="chat"))
    with pytest.raises(TelegramError, match="推播失敗"):
        notifier.send("hello")


def test_error_message_does_not_leak_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """例外訊息不可含 token"""
    def fake_post(url: str, **kwargs: object) -> None:
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "post", fake_post)

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="super-secret-token", chat_id="chat")
    )
    with pytest.raises(TelegramError) as exc:
        notifier.send("hello")
    assert "super-secret-token" not in str(exc.value)


def test_send_rejects_empty_message() -> None:
    notifier = TelegramNotifier(TelegramConfig(bot_token="t", chat_id="c"), dry_run=True)
    with pytest.raises(ValueError, match="訊息不可為空"):
        notifier.send("")


def test_long_message_is_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Telegram 單則訊息上限 4096 字元，超長必須分段。

    不分段會被 API 拒絕，整份建議都發不出去。
    """
    sent: list[str] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        sent.append(payload["text"])
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    notifier = TelegramNotifier(TelegramConfig(bot_token="t", chat_id="c"))
    notifier.send("行" * 9000)

    assert len(sent) >= 3
    assert all(len(chunk) <= 4096 for chunk in sent)


@pytest.mark.unit
def test_format_weekly_plan_accepts_trailing_result() -> None:
    """
    推播格式化對兩種選股結果都要能用。

    triple-barrier 與移動停損的欄位完全不同，但外框、版本號、
    免責聲明是共用的——綁死其中一個會逼另一個複製整套邏輯。
    """
    from taiwan_quant.config.costs import Tier
    from taiwan_quant.ranking.trailing_portfolio import (
        TrailingCandidate,
        select_trailing_portfolio,
    )

    candidate = TrailingCandidate(
        stock_id="2330",
        expected_gross_return=0.18,
        return_std=0.20,
        n_samples=400,
        trail_pct=0.12,
        entry_price=200.0,
        tier=Tier.LARGE,
        industry="半導體",
        volatility_pct=0.5,
        beta=0.9,
        max_horizon=60,
    )
    result = select_trailing_portfolio([candidate], 400_000.0, {})

    text = format_weekly_plan(result, week_id="2026W37", strategy_version="trail@v0.1")

    assert "移動停損" in text
    assert "2026W37" in text
    assert "系統不下單" in text
