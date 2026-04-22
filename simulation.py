import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from state import state

_PERIOD_MAP = {"1m": "7d", "5m": "60d", "15m": "60d"}


def _log_trade(trade_type, qty, price, reason=""):
    state.trade_log.append({
        "time":   datetime.now().strftime("%H:%M:%S"),
        "type":   trade_type,
        "qty":    qty,
        "price":  round(price, 2),
        "reason": reason,
    })


def run_simulation(ticker, settings):
    """
    Background thread target. Runs until state.running is set to False.

    settings keys:
        buy_thresh       float  Confidence threshold hint for BUY   (0–1, default 0.65)
        sell_thresh      float  Confidence threshold hint for SELL  (0–1, default 0.35)
        shares_per_trade int    Base shares per BUY/SELL signal
        poll_interval    int    Seconds between ticks
        stop_loss_pct    float  Auto-liquidate if portfolio down X% from start
        take_profit_pct  float  Auto-liquidate if portfolio up X% from start
        max_position     int    Maximum shares held simultaneously
        candle_interval  str    yfinance interval: "1m", "5m", or "15m"
    """
    fetch_period = _PERIOD_MAP.get(settings["candle_interval"], "7d")

    while state.running:
        if state.provider is None:
            with state.lock:
                state.status_msg = "No AI provider configured — stopping"
                state.running = False
            break

        try:
            data = yf.download(
                ticker,
                period=fetch_period,
                interval=settings["candle_interval"],
                auto_adjust=True,
                progress=False,
            ).tail(250)

            if data.empty:
                with state.lock:
                    state.status_msg = "No data — market may be closed"
                time.sleep(settings["poll_interval"])
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            current_price = float(data["Close"].iloc[-1])

            # Build context and ask AI provider for a decision
            with state.lock:
                trade_log_snapshot = list(state.trade_log)
                portfolio_snapshot = dict(state.portfolio)

            context = state.engine.build_context(
                data, ticker, portfolio_snapshot, trade_log_snapshot,
                buy_thresh=settings["buy_thresh"],
                sell_thresh=settings["sell_thresh"],
                personality=settings.get("personality", "balanced"),
            )
            decision = state.provider.decide(context)
            action     = decision.action
            confidence = decision.confidence
            reasoning  = decision.reasoning

            with state.lock:
                p = state.portfolio
                total_val = p["cash"] + p["shares"] * current_price
                pnl_pct   = (total_val - p["initial_cash"]) / p["initial_cash"]

                sl = settings["stop_loss_pct"]
                tp = settings["take_profit_pct"]

                # Risk controls (checked before AI signal)
                if sl and sl > 0 and pnl_pct <= -(sl / 100) and p["shares"] > 0:
                    p["cash"] += p["shares"] * current_price
                    _log_trade("STOP-LOSS", p["shares"], current_price, "stop-loss triggered")
                    p["shares"] = 0
                    action = "STOP-LOSS"

                elif tp and tp > 0 and pnl_pct >= (tp / 100) and p["shares"] > 0:
                    p["cash"] += p["shares"] * current_price
                    _log_trade("TAKE-PROFIT", p["shares"], current_price, "take-profit triggered")
                    p["shares"] = 0
                    action = "TAKE-PROFIT"

                elif action == "STRONG_BUY":
                    scale    = 1 + round(confidence)
                    headroom = settings["max_position"] - p["shares"]
                    can_buy  = min(settings["shares_per_trade"] * scale, headroom)
                    cost     = can_buy * current_price
                    if can_buy > 0 and p["cash"] >= cost:
                        p["cash"]   -= cost
                        p["shares"] += can_buy
                        _log_trade("STRONG_BUY", can_buy, current_price, reasoning)

                elif action == "BUY":
                    headroom = settings["max_position"] - p["shares"]
                    can_buy  = min(settings["shares_per_trade"], headroom)
                    cost     = can_buy * current_price
                    if can_buy > 0 and p["cash"] >= cost:
                        p["cash"]   -= cost
                        p["shares"] += can_buy
                        _log_trade("BUY", can_buy, current_price, reasoning)

                elif action == "STRONG_SELL" and p["shares"] > 0:
                    p["cash"] += p["shares"] * current_price
                    _log_trade("STRONG_SELL", p["shares"], current_price, reasoning)
                    p["shares"] = 0

                elif action == "SELL" and p["shares"] > 0:
                    sell_qty    = min(settings["shares_per_trade"], p["shares"])
                    p["cash"]  += sell_qty * current_price
                    p["shares"] -= sell_qty
                    _log_trade("SELL", sell_qty, current_price, reasoning)

                state.candle_df    = data
                state.last_prob    = confidence
                state.last_action  = action
                state.last_signals = context.get("signals", {})
                state.last_reasoning = reasoning
                state.status_msg = (
                    f"Running — last tick {datetime.now().strftime('%H:%M:%S')}"
                )

        except Exception as exc:
            with state.lock:
                state.status_msg = f"Error: {exc}"

        time.sleep(settings["poll_interval"])

    with state.lock:
        state.status_msg = "Stopped"
