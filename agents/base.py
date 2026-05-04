from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentVote:
    agent_name: str
    action: str            # BUY | SELL | HOLD | STRONG_BUY | STRONG_SELL
    confidence: float      # 0.0–1.0
    reasoning: str
    evidence: list[str] = field(default_factory=list)   # bullet items for UI cards
    token_usage: dict = field(default_factory=dict)      # {"prompt": int, "completion": int}


class Agent(ABC):
    name: str = "Agent"
    model: str = "gpt-4o-mini"

    @abstractmethod
    def vote(self, context: dict) -> AgentVote:
        """Produce a trading vote given the market context dict from engine.build_context()."""
        ...
