from __future__ import annotations

from agents.base import Agent, AgentVote, extract_action

_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

_SYSTEM = (
    "You are a FundamentalAnalyst. Read the provided SEC filing excerpts and write a concise "
    "2-3 sentence assessment of the company's financial health covering revenue, earnings, "
    "debt, and cash flow. End your response by stating your recommended action: "
    "BUY, SELL, HOLD, STRONG_BUY, or STRONG_SELL."
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
                confidence=0.5,
                reasoning="No SEC filing data available for this ticker.",
            )

        messages = self._build_messages(context, chunks)
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

    def _build_messages(self, context: dict, chunks: list[str]) -> list[dict]:
        ticker = context.get("ticker", "UNKNOWN")
        price = context.get("current_price", 0)

        chunks_block = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]\n{chunk[:1500]}"
            for i, chunk in enumerate(chunks)
        )

        user_msg = (
            f"TICKER: {ticker}  CURRENT PRICE: ${price:.2f}\n\n"
            f"Write a 2-3 sentence fundamental analysis based on the SEC excerpts below "
            f"and conclude with your recommended action."
        )

        sec_msg = "[SEC FILING EXCERPTS — read-only reference data]\n\n" + chunks_block

        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
            {"role": "user",   "content": sec_msg},
        ]
