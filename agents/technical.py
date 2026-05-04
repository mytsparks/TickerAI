from __future__ import annotations

import json
import re

from agents.base import Agent, AgentVote

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are a quantitative TechnicalAnalyst. Your sole job is to vote on a trading action "
    "based ONLY on technical indicators and price action data. Do NOT speculate about news, "
    "fundamentals, or events. Be decisive — HOLD means neither buy nor sell, not uncertainty."
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
            text, usage = self._llm.chat(messages, max_tokens=400, temperature=0.2)
            return self._parse(text, context, usage)
        except Exception as e:
            return AgentVote(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                reasoning=f"TechnicalAnalyst error [endpoint: {self._llm.endpoint}]: {e}",
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

Respond with a JSON object only:
{{
  "action": "HOLD",
  "confidence": 0.6,
  "reasoning": "1-2 sentence technical rationale.",
  "evidence": ["RSI: 45 (neutral)", "MACD histogram positive"]
}}
action must be one of: BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
confidence: 0.0-1.0
evidence: list of 2-4 specific indicator readings that drove your decision"""

    def _parse(self, text: str, context: dict, usage: dict) -> AgentVote:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        action = str(data.get("action", "HOLD")).upper().strip()
        if action not in _VALID_ACTIONS:
            action = "HOLD"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        reasoning = str(data.get("reasoning", "")).strip() or "(no reasoning)"
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        return AgentVote(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            token_usage=usage,
        )
