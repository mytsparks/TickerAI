from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentVote:
    agent_name: str
    action: str        # BUY | SELL | HOLD | STRONG_BUY | STRONG_SELL — used by Coordinator
    confidence: float  # kept for Coordinator table; not shown in UI
    reasoning: str     # full raw LLM response — displayed verbatim in UI
    evidence: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)


def extract_action(text: str, valid_actions: set) -> str:
    """
    Scan free-form LLM text for the most specific trading action keyword.
    Checks longer variants first so STRONG_BUY beats a plain BUY match.
    """
    for action in ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "HOLD"):
        if action in valid_actions and re.search(rf"\b{action}\b", text, re.IGNORECASE):
            return action
    return "HOLD"


# Used by Coordinator only — individual agents respond in free-form prose.
_COORDINATOR_FORMAT = """\
Reply with EXACTLY these three lines and nothing else:
ACTION: <BUY|SELL|HOLD|STRONG_BUY|STRONG_SELL>
CONFIDENCE: <0.0-1.0>
REASONING: <2-3 sentence synthesis citing the key deciding factors across analysts>"""

# OpenAI variant — JSON mode is more reliable than structured text for this endpoint.
_COORDINATOR_FORMAT_JSON = """\
Respond ONLY with valid JSON:
{"action": "BUY|SELL|HOLD|STRONG_BUY|STRONG_SELL", "confidence": 0.0-1.0, "reasoning": "2-3 sentence synthesis citing the key deciding factors"}"""


def parse_coordinator(text: str, valid_actions: set) -> dict:
    """Parse the Coordinator's structured 3-line response."""
    result = {"action": "HOLD", "confidence": 0.5, "reasoning": ""}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()
        if key == "ACTION":
            candidate = val.upper().replace(" ", "_")
            if candidate in valid_actions:
                result["action"] = candidate
        elif key == "CONFIDENCE":
            try:
                result["confidence"] = max(0.0, min(1.0, float(val)))
            except ValueError:
                pass
        elif key == "REASONING":
            result["reasoning"] = val
    return result


class Agent(ABC):
    name: str = "Agent"
    model: str = "gpt-4o-mini"

    @abstractmethod
    def vote(self, context: dict) -> AgentVote:
        """Produce a trading vote given the market context dict from engine.build_context()."""
        ...
