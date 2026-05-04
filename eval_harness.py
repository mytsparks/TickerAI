"""
eval_harness.py — Evaluation metrics and grid runner for v1 vs v2 vs buy-and-hold.

Metrics: Sharpe ratio (annualised), max drawdown (%), win rate (round-trips),
         turnover (trades/bars), total return (%), estimated LLM cost ($).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import yfinance as yf

from backtest import BacktestResult, TradeRecord, run_backtest

# -----------------------------------------------------------------------
# LLM cost rates ($/1M tokens, as of April 2026)
# -----------------------------------------------------------------------
_COST_RATES = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output":  0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


# -----------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------

def compute_sharpe(portfolio_values: list[float], rf: float = 0.04) -> float:
    """Annualised Sharpe ratio from a daily portfolio value series."""
    if len(portfolio_values) < 2:
        return float("nan")
    returns = [
        (portfolio_values[i] / portfolio_values[i - 1]) - 1
        for i in range(1, len(portfolio_values))
    ]
    n = len(returns)
    if n < 2:
        return float("nan")
    rf_daily = (1 + rf) ** (1 / 252) - 1
    mean_r = sum(returns) / n
    excess_mean = mean_r - rf_daily
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (excess_mean / std) * math.sqrt(252)


def compute_max_drawdown(portfolio_values: list[float]) -> float:
    """Max drawdown as a percentage (0–100)."""
    if len(portfolio_values) < 2:
        return 0.0
    peak = portfolio_values[0]
    max_dd = 0.0
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100


def compute_win_rate(trades: list[TradeRecord]) -> float:
    """FIFO round-trip win rate: profitable close / total closed pairs."""
    BUY_TYPES = {"BUY", "STRONG_BUY"}
    SELL_TYPES = {"SELL", "STRONG_SELL", "STOP-LOSS", "TAKE-PROFIT"}
    open_buys: list[float] = []
    wins = 0
    total_pairs = 0
    for t in trades:
        if t.trade_type in BUY_TYPES:
            open_buys.append(t.price)
        elif t.trade_type in SELL_TYPES and open_buys:
            buy_price = open_buys.pop(0)
            total_pairs += 1
            if t.price > buy_price:
                wins += 1
    return wins / total_pairs if total_pairs else 0.0


def compute_buy_hold_return(
    ticker: str,
    test_start: date,
    test_end: date,
    budget: float,
) -> float:
    """Buy-and-hold return % over the period."""
    try:
        raw = yf.download(
            ticker,
            start=test_start.strftime("%Y-%m-%d"),
            end=(test_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return 0.0
        if hasattr(raw.columns, "get_level_values"):
            raw.columns = raw.columns.get_level_values(0)
        start_price = float(raw["Close"].iloc[0])
        end_price = float(raw["Close"].iloc[-1])
        if start_price <= 0:
            return 0.0
        return (end_price / start_price - 1) * 100
    except Exception:
        return 0.0


def estimate_llm_cost(token_log: list[dict]) -> float:
    """Estimate USD cost from a list of {model, prompt, completion} dicts."""
    total = 0.0
    for entry in token_log:
        model = entry.get("model", "gpt-4o-mini")
        prompt_tokens = entry.get("prompt", 0)
        completion_tokens = entry.get("completion", 0)
        rates = _COST_RATES.get(model, _COST_RATES["gpt-4o-mini"])
        total += (prompt_tokens / 1_000_000) * rates["input"]
        total += (completion_tokens / 1_000_000) * rates["output"]
    return round(total, 6)


def _extract_portfolio_values(result: BacktestResult) -> list[float]:
    """Build daily portfolio value series. Uses daily_values if populated."""
    if result.daily_values:
        return [v for _, v in result.daily_values]
    # Fallback: reconstruct from trade records
    if not result.trades:
        return [result.initial_budget, result.final_value]
    values = [result.initial_budget]
    for t in result.trades:
        values.append(t.portfolio_value)
    values.append(result.final_value)
    return values


# -----------------------------------------------------------------------
# EvalMetrics dataclass
# -----------------------------------------------------------------------

@dataclass
class EvalMetrics:
    sharpe_ratio: float = float("nan")
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0       # trades / bars
    total_return_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    llm_cost_usd: float = 0.0
    total_trades: int = 0
    bars: int = 0

    def as_dict(self) -> dict:
        def _fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "N/A"
            return v
        return {k: _fmt(v) for k, v in self.__dict__.items()}


def compute_metrics(
    result: BacktestResult,
    ticker: str,
    test_start: date,
    test_end: date,
    token_log: list[dict] | None = None,
) -> EvalMetrics:
    values = _extract_portfolio_values(result)
    bars = len(result.daily_values) if result.daily_values else max(len(values) - 1, 1)
    buy_hold_ret = compute_buy_hold_return(ticker, test_start, test_end, result.initial_budget)
    cost = estimate_llm_cost(token_log) if token_log else 0.0

    return EvalMetrics(
        sharpe_ratio=compute_sharpe(values),
        max_drawdown_pct=compute_max_drawdown(values),
        win_rate=compute_win_rate(result.trades),
        turnover=result.total_trades / bars if bars > 0 else 0.0,
        total_return_pct=result.pnl_pct,
        buy_hold_return_pct=buy_hold_ret,
        llm_cost_usd=cost,
        total_trades=result.total_trades,
        bars=bars,
    )


# -----------------------------------------------------------------------
# Grid runner
# -----------------------------------------------------------------------

@dataclass
class GridCell:
    ticker: str
    regime_label: str
    provider_label: str
    metrics: EvalMetrics | None = None
    error: str = ""


def run_eval_grid(
    tickers: list[str],
    regimes: list[dict],
    budget: float,
    settings: dict,
    v1_provider,
    v2_provider,
    progress_callback=None,
) -> list[GridCell]:
    """
    Run v1, v2, and buy-and-hold across all (ticker, regime) combinations.

    regimes: list of {"label": str, "start": date, "end": date}
    Returns flat list of GridCell for table rendering.
    progress_callback(done: int, total: int) fired after each cell.
    """
    cells: list[GridCell] = []
    total = len(tickers) * len(regimes) * 2  # v1 + v2 (buy-hold is free)
    done = 0

    for ticker in tickers:
        for regime in regimes:
            label = regime["label"]
            test_start: date = regime["start"]
            test_end: date = regime["end"]

            # Buy-and-hold (no LLM calls)
            bh_return = compute_buy_hold_return(ticker, test_start, test_end, budget)
            bh_metrics = EvalMetrics(
                total_return_pct=bh_return,
                buy_hold_return_pct=bh_return,
                bars=0,
            )
            cells.append(GridCell(ticker, label, "buy_hold", bh_metrics))

            for provider_label, provider in [("v1_openai", v1_provider), ("v2_committee", v2_provider)]:
                if hasattr(provider, "clear_cache"):
                    provider.clear_cache()
                try:
                    result = run_backtest(
                        ticker=ticker,
                        test_start=test_start,
                        test_end=test_end,
                        budget=budget,
                        settings=settings,
                        provider=provider,
                    )
                    token_log = getattr(provider, "last_token_log", None)
                    if result.error:
                        cells.append(GridCell(ticker, label, provider_label, error=result.error))
                    else:
                        m = compute_metrics(result, ticker, test_start, test_end, token_log)
                        cells.append(GridCell(ticker, label, provider_label, metrics=m))
                except Exception as e:
                    cells.append(GridCell(ticker, label, provider_label, error=str(e)))

                done += 1
                if progress_callback:
                    progress_callback(done, total)

    return cells
