from __future__ import annotations

import json
import re

from agents.base import Agent, AgentVote

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are a RiskManager. Your job is to assess portfolio and market risk, not predict direction. "
    "You vote HOLD or SELL when risk is too high regardless of other bullish signals. "
    "You vote BUY only when risk is low and portfolio is under-allocated. "
    "Never use STRONG_BUY or STRONG_SELL — that is not your role. Be conservative."
)


class RiskManager(Agent):
    name = "RiskManager"
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
            text, usage = self._llm.chat(messages, max_tokens=350, temperature=0.1)
            return self._parse(text, usage)
        except Exception as e:
            return AgentVote(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                reasoning=f"RiskManager error [endpoint: {self._llm.endpoint}]: {e}",
            )

    def _build_prompt(self, context: dict) -> str:
        portfolio = context.get("portfolio", {})
        signals = context.get("signals", {})
        historical = context.get("historical", {})
        portfolio_value_series: list[float] = context.get("portfolio_value_series", [])

        cash = portfolio.get("cash", 0)
        shares = portfolio.get("shares", 0)
        total = portfolio.get("total_value", cash)
        price = context.get("current_price", 0)
        ticker = context.get("ticker", "UNKNOWN")

        equity_value = shares * price
        concentration_pct = (equity_value / total * 100) if total > 0 else 0.0
        cash_pct = (cash / total * 100) if total > 0 else 100.0
        atr_norm = signals.get("atr_norm", 0)

        drawdown_pct = 0.0
        if len(portfolio_value_series) >= 2:
            peak = max(portfolio_value_series)
            current_val = portfolio_value_series[-1]
            if peak > 0:
                drawdown_pct = (peak - current_val) / peak * 100

        return f"""TICKER: {ticker}  PRICE: ${price:.2f}

PORTFOLIO STATE:
  Cash:              ${cash:.2f}  ({cash_pct:.1f}% of portfolio)
  Shares held:       {shares}
  Equity value:      ${equity_value:.2f}  ({concentration_pct:.1f}% of portfolio)
  Total value:       ${total:.2f}
  Drawdown from peak: {drawdown_pct:.1f}%

MARKET VOLATILITY:
  ATR (normalized):  {atr_norm:.4f}  [>0.03 = elevated volatility]
  Trend alignment:   {historical.get('trend_alignment', 'unknown')}
  Return 5-bar:      {historical.get('return_5bar', 0) or 0:.2f}%

Assess risk and vote. Respond with JSON only:
{{
  "action": "HOLD",
  "confidence": 0.65,
  "reasoning": "1-2 sentence risk rationale.",
  "evidence": ["Concentration: 45% in single position", "ATR elevated at 0.035"]
}}
action: BUY | SELL | HOLD only (no STRONG_BUY/STRONG_SELL)
confidence: 0.0-1.0
evidence: 2-3 specific risk metrics"""

    def _parse(self, text: str, usage: dict) -> AgentVote:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        action = str(data.get("action", "HOLD")).upper().strip()
        if action not in _VALID_ACTIONS:
            action = "HOLD"
        if action in ("STRONG_BUY", "STRONG_SELL"):
            action = "BUY" if action == "STRONG_BUY" else "SELL"
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
