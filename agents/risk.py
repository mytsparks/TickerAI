from __future__ import annotations

from agents.base import Agent, AgentVote, extract_action

_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

_SYSTEM = (
    "You are a RiskManager. Assess portfolio and market risk based on the provided data. "
    "Write a concise 2-3 sentence risk assessment. End your response by stating your "
    "recommended action: BUY, SELL, or HOLD only — never STRONG_BUY or STRONG_SELL."
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
            text, usage = self._llm.chat_prose(messages, max_tokens=200, temperature=0.1)
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
  Cash:               ${cash:.2f}  ({cash_pct:.1f}% of portfolio)
  Shares held:        {shares}
  Equity value:       ${equity_value:.2f}  ({concentration_pct:.1f}% of portfolio)
  Total value:        ${total:.2f}
  Drawdown from peak: {drawdown_pct:.1f}%

MARKET VOLATILITY:
  ATR (normalized):   {atr_norm:.4f}  [>0.03 = elevated]
  Trend alignment:    {historical.get('trend_alignment', 'unknown')}
  Return 5-bar:       {historical.get('return_5bar', 0) or 0:.2f}%

Write a 2-3 sentence risk assessment and conclude with your recommended action (BUY, SELL, or HOLD only)."""
