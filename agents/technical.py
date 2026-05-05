from __future__ import annotations

from agents.base import Agent, AgentVote, extract_action

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are a quantitative TechnicalAnalyst. Analyze the provided price and indicator data "
    "and write a concise 2-3 sentence assessment. End your response by stating your recommended "
    "action: BUY, SELL, HOLD, STRONG_BUY, or STRONG_SELL."
)


class TechnicalAnalyst(Agent):
    name = "TechnicalAnalyst"
    model = "gpt-4o-mini"

    def __init__(self, llm) -> None:
        from llm_client import LLMClient
        self._llm: LLMClient = llm
        self.model = llm.model

    def vote(self, context: dict) -> AgentVote:
        prompt = self._build_prompt(context)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            text, usage = self._llm.chat_prose(messages, max_tokens=200, temperature=0.2)
            return AgentVote(
                agent_name=self.name,
                action=extract_action(text, _VALID_ACTIONS),
                confidence=0.5,
                reasoning=text.strip(),
                token_usage=usage,
            )
        except Exception as e:
            return AgentVote(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                reasoning=f"Error: {self._llm.endpoint} — {e}",
            )

    def _build_prompt(self, context: dict) -> str:
        s = context.get("signals", {})
        h = context.get("historical", {})
        ticker = context.get("ticker", "UNKNOWN")
        price = context.get("current_price", 0)

        def _pct(v):
            if v is None:
                return "N/A"
            return f"{'+' if v >= 0 else ''}{v:.2f}%"

        hist_block = ""
        if h:
            hist_block = (
                f"  Return 5-bar:   {_pct(h.get('return_5bar'))}\n"
                f"  Return 20-bar:  {_pct(h.get('return_20bar'))}\n"
                f"  Return 60-bar:  {_pct(h.get('return_60bar'))}\n"
                f"  From high:      {_pct(h.get('pct_from_high'))}\n"
                f"  From low:       {_pct(h.get('pct_from_low'))}\n"
                f"  Trend:          {h.get('trend_alignment', 'unknown')}"
            )

        return f"""TICKER: {ticker}  PRICE: ${price:.2f}

TECHNICAL INDICATORS:
  RSI (14):          {s.get('rsi', 50):.1f}  [overbought >70, oversold <30]
  MACD Histogram:    {s.get('macd_hist', 0):.4f}  [+ = bullish momentum]
  Bollinger %B:      {s.get('bb_pct', 0.5):.3f}  [0=lower band, 1=upper band]
  Price vs SMA20:    {s.get('price_vs_sma20', 0):+.2%}
  Price vs SMA50:    {s.get('price_vs_sma50', 0):+.2%}
  ATR (normalized):  {s.get('atr_norm', 0):.4f}

CANDLESTICK PATTERNS:
  Hammer:             {s.get('is_hammer', False)}
  Bullish Engulfing:  {s.get('is_bull_engulf', False)}
  Bearish Engulfing:  {s.get('is_bear_engulf', False)}
  Doji:               {s.get('is_doji', False)}

HISTORICAL CONTEXT:
{hist_block}

Write a 2-3 sentence technical analysis of these indicators and conclude with your recommended action."""
