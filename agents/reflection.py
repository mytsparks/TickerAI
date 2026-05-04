from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import AgentVote
    from memory import MemoryStore

_REFLECTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="reflection"
)

_SYSTEM = (
    "You are a ReflectionAgent. After each trading decision, you extract a concise, "
    "actionable lesson the committee should remember for similar situations in the future. "
    "Focus on WHAT was decided, WHY, and whether the signals were clear or conflicting. "
    "Write a single sentence: specific, factual, and forward-looking."
)


class ReflectionAgent:
    model = "gpt-4o-mini"

    def __init__(self, llm, memory_store: "MemoryStore") -> None:
        from llm_client import LLMClient
        self._llm: LLMClient = llm
        self.model = llm.model
        self._memory = memory_store

    def reflect(
        self,
        ticker: str,
        bar_date: str,
        context: dict,
        votes: list["AgentVote"],
        decision,
        outcome_pnl: float | None = None,
    ) -> concurrent.futures.Future:
        """Submit reflection to background executor. Non-blocking."""
        return _REFLECTION_EXECUTOR.submit(
            self._do_reflect, ticker, bar_date, context, votes, decision, outcome_pnl
        )

    def _do_reflect(
        self,
        ticker: str,
        bar_date: str,
        context: dict,
        votes: list["AgentVote"],
        decision,
        outcome_pnl: float | None,
    ) -> None:
        try:
            prompt = self._build_prompt(ticker, bar_date, context, votes, decision, outcome_pnl)
            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ]
            text, _ = self._llm.chat(messages, max_tokens=120, temperature=0.3, json_mode=False)
            lesson = text.strip()
            if lesson:
                regime = self._derive_regime(context)
                trigger = f"{decision.action}_{bar_date}"
                self._memory.add_lesson(ticker, regime, trigger, lesson)
        except Exception as e:
            print(f"[ReflectionAgent] Reflection failed for {ticker} {bar_date} [endpoint: {self._llm.endpoint}]: {e}")

    def _build_prompt(
        self,
        ticker: str,
        bar_date: str,
        context: dict,
        votes: list["AgentVote"],
        decision,
        outcome_pnl: float | None,
    ) -> str:
        vote_lines = "\n".join(
            f"  {v.agent_name}: {v.action} ({v.confidence:.2f}) — {v.reasoning[:100]}"
            for v in votes
        )
        pnl_str = f"{outcome_pnl:+.2f}%" if outcome_pnl is not None else "unknown"
        signals = context.get("signals", {})

        return (
            f"TICKER: {ticker}  DATE: {bar_date}\n"
            f"RSI: {signals.get('rsi', 50):.1f}  MACD: {signals.get('macd_hist', 0):.4f}\n\n"
            f"COMMITTEE VOTES:\n{vote_lines}\n\n"
            f"FINAL DECISION: {decision.action} (confidence {decision.confidence:.2f})\n"
            f"OUTCOME P&L: {pnl_str}\n\n"
            f"Write a single actionable lesson (1 sentence) for the committee to remember:"
        )

    @staticmethod
    def _derive_regime(context: dict) -> str:
        trend = context.get("historical", {}).get("trend_alignment", "")
        if "bullish" in trend.lower():
            return "bull"
        if "bearish" in trend.lower():
            return "bear"
        return "sideways"
