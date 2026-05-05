from __future__ import annotations

from agents.base import AgentVote, _COORDINATOR_FORMAT, _COORDINATOR_FORMAT_JSON, parse_coordinator

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are the Coordinator of a multi-agent trading committee. "
    "You receive votes from specialist analysts and must synthesize them into a single, committed decision. "
    "Do NOT hedge by defaulting to HOLD when the committee disagrees — you must weigh the evidence and commit. "
    "Use STRONG_BUY or STRONG_SELL when multiple high-confidence analysts agree. "
    "Factor in past lessons from memory when relevant."
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
        from providers import TradingDecision

        use_json = (self._llm.provider == "openai")
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": self._build_prompt(votes, context, memory_lessons, use_json=use_json)},
        ]
        try:
            if use_json:
                import json as _json
                text, usage = self._llm.chat(messages, max_tokens=250, temperature=0.2, json_mode=True)
                try:
                    data = _json.loads(text)
                    action = data.get("action", "HOLD").upper().replace(" ", "_")
                    if action not in _VALID_ACTIONS:
                        action = "HOLD"
                    td = TradingDecision(
                        action=action,
                        confidence=float(data.get("confidence", 0.5)),
                        reasoning=str(data.get("reasoning", "(no synthesis reasoning)")),
                    )
                except Exception:
                    td = TradingDecision(action="HOLD", confidence=0.5, reasoning=text or "(parse error)")
            else:
                text, usage = self._llm.chat_prose(messages, max_tokens=250, temperature=0.2)
                td = self._parse(text, usage)
            td.coordinator_token_usage = usage
            return td
        except Exception as e:
            return TradingDecision(
                action="HOLD",
                confidence=0.5,
                reasoning=f"Coordinator error [{self._llm.endpoint}]: {e}",
            )

    def _build_prompt(self, votes: list[AgentVote], context: dict, lessons: list[str], use_json: bool = False) -> str:
        ticker = context.get("ticker", "UNKNOWN")
        price = context.get("current_price", 0)
        portfolio = context.get("portfolio", {})

        vote_rows = []
        for v in votes:
            vote_rows.append(
                f"  {v.agent_name:<22} | {v.action:<12} | {v.reasoning[:200]}"
            )
        vote_block = "\n".join(vote_rows) if vote_rows else "  (no votes received)"

        action_counts: dict[str, int] = {}
        for v in votes:
            action_counts[v.action] = action_counts.get(v.action, 0) + 1
        vote_summary = ", ".join(f"{k}:{n}" for k, n in action_counts.items())

        lessons_block = ""
        if lessons:
            lessons_block = "\nPAST EXPERIENCE (from memory — apply if relevant):\n"
            for i, lesson in enumerate(lessons, 1):
                lessons_block += f"  {i}. {lesson[:200]}\n"

        format_block = _COORDINATOR_FORMAT_JSON if use_json else _COORDINATOR_FORMAT
        return f"""TICKER: {ticker}  PRICE: ${price:.2f}
Portfolio: ${portfolio.get('total_value', 0):.2f} (cash ${portfolio.get('cash', 0):.2f}, {portfolio.get('shares', 0)} shares)

COMMITTEE VOTES:
  Agent                  | Action       | Analysis
  {'─'*80}
{vote_block}

Vote tally: {vote_summary}
{lessons_block}
{format_block}"""

    def _parse(self, text: str, usage: dict):
        from providers import TradingDecision
        d = parse_coordinator(text, _VALID_ACTIONS)
        td = TradingDecision(
            action=d["action"],
            confidence=d["confidence"],
            reasoning=d["reasoning"] or "(no synthesis reasoning)",
        )
        td.coordinator_token_usage = usage
        return td
