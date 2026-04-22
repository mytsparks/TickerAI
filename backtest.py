"""
backtest.py — Bar-by-bar historical backtesting engine.

Entry point: run_backtest(ticker, test_start, test_end, budget, settings, provider)

For each bar in the test window the AI provider is queried with current
technical indicators and portfolio state. No ML training phase is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from engine import TradingEngine

# Bars of lookback carried forward into each test-window step so that
# rolling indicators (SMA_200, ATR_14, etc.) are warm from bar 1.
_LOOKBACK = 250


@dataclass
class TradeRecord:
    date: str           # ISO date string of execution bar
    trade_type: str     # BUY | STRONG_BUY | SELL | STRONG_SELL | STOP-LOSS | TAKE-PROFIT
    qty: int
    price: float
    reason: str
    portfolio_value: float  # total portfolio value snapshot after this trade


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)          # list[TradeRecord]
    candle_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_value: float = 0.0
    initial_budget: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    total_trades: int = 0
    error: str = ""     # non-empty string signals failure; UI shows this as alert


def run_backtest(
    ticker: str,
    test_start: date,
    test_end: date,
    budget: float,
    settings: dict,
    provider,
    progress_callback=None,
) -> BacktestResult:
    """
    Run a bar-by-bar backtest over the given date range.

    Parameters
    ----------
    ticker      : e.g. "NVDA"
    test_start  : first bar of the test window (inclusive)
    test_end    : last bar of the test window (inclusive)
    budget      : starting cash in dollars
    settings    : dict with keys —
                    buy_thresh       float  (default 0.65)
                    sell_thresh      float  (default 0.35)
                    shares_per_trade int    (default 1)
                    max_position     int    (default 10)
                    stop_loss_pct    float  (default 0)
                    take_profit_pct  float  (default 0)
    provider    : BaseProvider instance (Ollama, Claude, or Gemini)

    Returns
    -------
    BacktestResult — always returned; check .error for failure details.
    """

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    today = date.today()

    if test_start >= test_end:
        return BacktestResult(error="Test start date must be before test end date.")

    if test_end > today:
        return BacktestResult(error="Test end date cannot be in the future.")

    if (test_end - test_start).days < 5:
        return BacktestResult(error="Test window must span at least 5 calendar days.")

    if provider is None:
        return BacktestResult(error="No AI provider configured. Apply provider config first.")

    # ------------------------------------------------------------------
    # 2. Fetch lookback + test data in a single yfinance call
    #    ~400 calendar days gives ≥250 trading days for indicator warm-up.
    # ------------------------------------------------------------------
    lookback_start = test_start - timedelta(days=400)

    try:
        raw = yf.download(
            ticker,
            start=lookback_start.strftime("%Y-%m-%d"),
            end=(test_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        return BacktestResult(error=f"Data download failed: {exc}")

    if raw.empty:
        return BacktestResult(error=f"No data returned for '{ticker}'. Check the ticker symbol.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ------------------------------------------------------------------
    # 3. Split into pre-test lookback and test window
    # ------------------------------------------------------------------
    pre_mask  = raw.index.date < test_start
    test_mask = raw.index.date >= test_start

    lookback_full = raw[pre_mask]
    test_df       = raw[test_mask]

    if len(lookback_full) < 10:
        return BacktestResult(
            error=(
                f"Only {len(lookback_full)} bars found before {test_start}. "
                "Choose an earlier test start date or a ticker with longer history."
            )
        )

    if test_df.empty:
        return BacktestResult(
            error="No trading bars found in the test window. "
                  "The market may have been closed across the entire period."
        )

    # ------------------------------------------------------------------
    # 4. Extract settings
    # ------------------------------------------------------------------
    buy_thresh       = float(settings.get("buy_thresh",       0.65))
    sell_thresh      = float(settings.get("sell_thresh",      0.35))
    shares_per_trade = int(settings.get("shares_per_trade",   1))
    max_position     = int(settings.get("max_position",       10))
    stop_loss_pct    = float(settings.get("stop_loss_pct",    0))
    take_profit_pct  = float(settings.get("take_profit_pct",  0))

    # ------------------------------------------------------------------
    # 5. Bar-by-bar simulation
    # ------------------------------------------------------------------
    _engine = TradingEngine()  # stateless — used only for feature computation

    lookback_df = lookback_full.tail(_LOOKBACK).copy()

    portfolio = {
        "cash":         float(budget),
        "shares":       0,
        "initial_cash": float(budget),
    }
    trades: list[TradeRecord] = []

    for i in range(len(test_df)):
        bar           = test_df.iloc[[i]]
        bar_date_str  = test_df.index[i].strftime("%Y-%m-%d")
        current_price = float(bar["Close"].iloc[0])

        if pd.isna(current_price) or current_price <= 0:
            lookback_df = pd.concat([lookback_df, bar]).tail(_LOOKBACK)
            continue

        window = pd.concat([lookback_df, bar]).tail(_LOOKBACK)

        trades_as_dicts = [
            {"date": t.date, "type": t.trade_type, "qty": t.qty, "price": t.price}
            for t in trades[-5:]
        ]

        try:
            context = _engine.build_context(
                window, ticker, portfolio, trades_as_dicts,
                buy_thresh=buy_thresh, sell_thresh=sell_thresh,
                personality=settings.get("personality", "balanced"),
            )
            decision = provider.decide(context)
            action      = decision.action
            confidence  = decision.confidence
            rule_reason = decision.reasoning
        except Exception:
            lookback_df = window
            continue

        p         = portfolio
        total_val = p["cash"] + p["shares"] * current_price
        pnl_pct   = (total_val - p["initial_cash"]) / p["initial_cash"]

        # --- Risk controls ---
        if (stop_loss_pct > 0
                and pnl_pct <= -(stop_loss_pct / 100)
                and p["shares"] > 0):
            p["cash"] += p["shares"] * current_price
            trades.append(TradeRecord(
                date=bar_date_str, trade_type="STOP-LOSS",
                qty=p["shares"], price=round(current_price, 2),
                reason="stop-loss triggered",
                portfolio_value=round(p["cash"], 2),
            ))
            p["shares"] = 0

        elif (take_profit_pct > 0
                and pnl_pct >= (take_profit_pct / 100)
                and p["shares"] > 0):
            p["cash"] += p["shares"] * current_price
            trades.append(TradeRecord(
                date=bar_date_str, trade_type="TAKE-PROFIT",
                qty=p["shares"], price=round(current_price, 2),
                reason="take-profit triggered",
                portfolio_value=round(p["cash"], 2),
            ))
            p["shares"] = 0

        elif action == "STRONG_BUY":
            scale    = 1 + round(confidence)
            headroom = max_position - p["shares"]
            can_buy  = min(shares_per_trade * scale, headroom)
            cost     = can_buy * current_price
            if can_buy > 0 and p["cash"] >= cost:
                p["cash"]   -= cost
                p["shares"] += can_buy
                total_val = p["cash"] + p["shares"] * current_price
                trades.append(TradeRecord(
                    date=bar_date_str, trade_type="STRONG_BUY",
                    qty=can_buy, price=round(current_price, 2),
                    reason=rule_reason,
                    portfolio_value=round(total_val, 2),
                ))

        elif action == "BUY":
            headroom = max_position - p["shares"]
            can_buy  = min(shares_per_trade, headroom)
            cost     = can_buy * current_price
            if can_buy > 0 and p["cash"] >= cost:
                p["cash"]   -= cost
                p["shares"] += can_buy
                total_val = p["cash"] + p["shares"] * current_price
                trades.append(TradeRecord(
                    date=bar_date_str, trade_type="BUY",
                    qty=can_buy, price=round(current_price, 2),
                    reason=rule_reason,
                    portfolio_value=round(total_val, 2),
                ))

        elif action == "STRONG_SELL" and p["shares"] > 0:
            qty = p["shares"]
            p["cash"] += qty * current_price
            p["shares"] = 0
            trades.append(TradeRecord(
                date=bar_date_str, trade_type="STRONG_SELL",
                qty=qty, price=round(current_price, 2),
                reason=rule_reason,
                portfolio_value=round(p["cash"], 2),
            ))

        elif action == "SELL" and p["shares"] > 0:
            qty = min(shares_per_trade, p["shares"])
            p["cash"]   += qty * current_price
            p["shares"] -= qty
            total_val = p["cash"] + p["shares"] * current_price
            trades.append(TradeRecord(
                date=bar_date_str, trade_type="SELL",
                qty=qty, price=round(current_price, 2),
                reason=rule_reason,
                portfolio_value=round(total_val, 2),
            ))

        lookback_df = window

        if progress_callback:
            progress_callback((i + 1) / len(test_df))

    # ------------------------------------------------------------------
    # 6. Mark-to-market any remaining open position at last test bar close
    # ------------------------------------------------------------------
    last_price  = float(test_df["Close"].iloc[-1])
    final_value = round(portfolio["cash"] + portfolio["shares"] * last_price, 2)
    net_pnl     = round(final_value - budget, 2)
    pnl_pct_val = round((net_pnl / budget) * 100, 2) if budget else 0.0

    return BacktestResult(
        trades=trades,
        candle_df=test_df,
        final_value=final_value,
        initial_budget=float(budget),
        net_pnl=net_pnl,
        pnl_pct=pnl_pct_val,
        total_trades=len(trades),
        error="",
    )
