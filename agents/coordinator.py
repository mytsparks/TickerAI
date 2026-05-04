from __future__ import annotations

import json
import re

from agents.base import AgentVote

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are the Coordinator of a multi-agent trading committee. "
    "You receive votes from specialist analysts and must synthesize them into a single, committed decision. "
    "Do NOT hedge by defaulting to HOLD when the committee disagrees — you must weigh the evidence and commit. "
    "Use STRONG_BUY or STRONG_SELL when multiple high-confidence analysts agree. "
    "Factor in past lessons from memory when relevant. "
    "Your response must be decisive and grounded in the analysts' evidence."
)


class Coordinator:
    model = "gpt-4o"

    def __init__(self, llm) -> None:
        from llm_client import LLMClient
        self._llm: LLMClient = llm
        self.model = llm.model

    def synthesize(
        self,
        votes: list[AgentVote],
        context: dict,
        memory_lessons: list[str],
    ):
        """Returns a TradingDecision (imported lazily to avoid circular imports)."""
        from providers import TradingDecision

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": self._build_prompt(votes, context, memory_lessons)},
        ]
        try:
            text, usage = self._llm.chat(messages, max_tokens=500, temperature=0.2)
            return self._parse(text, usage)
        except Exception as e:
            return TradingDecision(
                action="HOLD",
                confidence=0.5,
                reasoning=f"Coordinator error [endpoint: {self._llm.endpoint}]: {e}",
            )

    def _build_prompt(
        self,
        votes: list[AgentVote],
        context: dict,
        lessons: list[str],
    ) -> str:
        ticker = context.get("ticker", "UNKNOWN")
        price = context.get("current_price", 0)
        portfolio = context.get("portfolio", {})

        vote_rows = []
        for v in votes:
            ev = "; ".join(v.evidence[:2]) if v.evidence else "—"
            vote_rows.append(
                f"  {v.agent_name:<22} | {v.action:<12} | {v.confidence:.2f} | "
                f"{v.reasoning[:180]}\n    Evidence: {ev}"
            )
        vote_block = "\n".join(vote_rows) if vote_rows else "  (no votes received)"

        lessons_block = ""
        if lessons:
            lessons_block = "\nPAST EXPERIENCE (from memory — apply if relevant):\n"
            for i, lesson in enumerate(lessons, 1):
                lessons_block += f"  {i}. {lesson[:200]}\n"

        action_counts: dict[str, int] = {}
        for v in votes:
            action_counts[v.action] = action_counts.get(v.action, 0) + 1
        vote_summary = ", ".join(f"{k}:{n}" for k, n in action_counts.items())

        return f"""TICKER: {ticker}  PRICE: ${price:.2f}
Portfolio: ${portfolio.get('total_value', 0):.2f} (cash ${portfolio.get('cash', 0):.2f}, {portfolio.get('shares', 0)} shares)

COMMITTEE VOTES:
  Agent                  | Action       | Conf | Reasoning / Evidence
  {'─'*80}
{vote_block}

Vote tally: {vote_summary}
{lessons_block}
Synthesize these votes into a single trading decision. You MUST commit to an action.
Respond with JSON only:
{{
  "action": "HOLD",
  "confidence": 0.72,
  "reasoning": "2-3 sentence synthesis citing the key deciding factors across analysts."
}}
action: BUY | SELL | HOLD | STRONG_BUY | STRONG_SELL
confidence: 0.0-1.0"""

    def _parse(self, text: str, usage: dict):
        from providers import TradingDecision

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
        reasoning = str(data.get("reasoning", "")).strip() or "(no synthesis reasoning)"

        td = TradingDecision(action=action, confidence=confidence, reasoning=reasoning)
        td.coordinator_token_usage = usage
        return td
