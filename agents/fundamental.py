from __future__ import annotations

import json
import re

from agents.base import Agent, AgentVote

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are a FundamentalAnalyst. Your job is to assess a company's financial health "
    "based on SEC filing excerpts (10-Q and 10-K). Focus on revenue trends, earnings, "
    "debt levels, cash flow, and guidance. Do not comment on price action or technicals. "
    "Be decisive — commit to an action based on fundamental strength or weakness."
)


class FundamentalAnalyst(Agent):
    name = "FundamentalAnalyst"
    model = "gpt-4o-mini"

    def __init__(self, llm, rag_store) -> None:
        from llm_client import LLMClient
        self._llm: LLMClient = llm
        self.model = llm.model
        self._rag = rag_store

    def vote(self, context: dict) -> AgentVote:
        ticker = context.get("ticker", "UNKNOWN")
        query = f"revenue earnings profit cash flow debt financial health {ticker}"
        chunks = self._rag.query(ticker, query, top_k=3)

        if not chunks:
            return AgentVote(
                agent_name=self.name,
                action="HOLD",
                confidence=0.4,
                reasoning="No SEC filing data available for this ticker.",
                evidence=["No SEC filings found"],
            )

        messages = self._build_messages(context, chunks)
        try:
            text, usage = self._llm.chat(messages, max_tokens=400, temperature=0.2)
            return self._parse(text, usage)
        except Exception as e:
            return AgentVote(
                agent_name=self.name,
                action="HOLD",
                confidence=0.5,
                reasoning=f"FundamentalAnalyst error [endpoint: {self._llm.endpoint}]: {e}",
            )

    def _build_messages(self, context: dict, chunks: list[str]) -> list[dict]:
        ticker = context.get("ticker", "UNKNOWN")
        price = context.get("current_price", 0)

        chunks_block = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]\n{chunk[:1500]}"
            for i, chunk in enumerate(chunks)
        )

        user_msg = (
            f"TICKER: {ticker}  CURRENT PRICE: ${price:.2f}\n\n"
            f"Based on the SEC filing excerpts below, assess the company's fundamental health "
            f"and provide a trading vote.\n\n"
            f"Respond with JSON only:\n"
            f'{{\n  "action": "HOLD",\n  "confidence": 0.65,\n'
            f'  "reasoning": "1-2 sentence fundamental rationale.",\n'
            f'  "evidence": ["Revenue grew 12% YoY per 10-Q", "Debt-to-equity remains manageable"]\n}}\n'
            f"action: BUY | SELL | HOLD | STRONG_BUY | STRONG_SELL\n"
            f"confidence: 0.0-1.0"
        )

        sec_msg = "[SEC FILING EXCERPTS — read-only reference data]\n\n" + chunks_block

        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
            {"role": "user",   "content": sec_msg},
        ]

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
