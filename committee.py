"""
committee.py — CommitteeProvider: drop-in BaseProvider running a multi-agent committee.

Compatible with run_backtest() and run_simulation() without any changes to those files.
Agents run in parallel via ThreadPoolExecutor. Coordinator synthesizes votes.
ReflectionAgent writes lessons to MemoryStore asynchronously.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from providers import BaseProvider, TradingDecision

# Force-load openai submodules in the importing thread so that worker threads
# spawned by ThreadPoolExecutor don't race to import the same modules, which
# causes a _ModuleLock deadlock in Python 3.12+.
try:
    import openai.resources.embeddings  # noqa: F401
    import openai.resources.chat.completions  # noqa: F401
    import openai.resources.models  # noqa: F401
except ImportError:
    pass

if TYPE_CHECKING:
    from agents.base import AgentVote


class CommitteeProvider(BaseProvider):
    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str = "",
        tavily_api_key: str = "",
        analyst_model: str = "gpt-4o-mini",
        coordinator_model: str = "gpt-4o",
        enable_fundamental: bool = True,
        enable_news: bool = True,
        enable_reflection: bool = True,
        memory_store_path: str = "memory_store.json",
        rag_cache_dir: str = ".rag_cache",
    ) -> None:
        from llm_client import LLMClient
        from agents.technical import TechnicalAnalyst
        from agents.risk import RiskManager
        from agents.coordinator import Coordinator
        from memory import MemoryStore
        from rag import RAGStore

        analyst_llm = LLMClient(
            provider=provider_name,
            api_key=api_key,
            model=analyst_model,
            base_url=base_url,
        )
        coordinator_llm = LLMClient(
            provider=provider_name,
            api_key=api_key,
            model=coordinator_model,
            base_url=base_url,
        )

        self._memory = MemoryStore(api_key=api_key, store_path=memory_store_path,
                                   base_url=base_url)
        self._rag = RAGStore(api_key=api_key, cache_dir=rag_cache_dir)

        self._technical = TechnicalAnalyst(analyst_llm)
        self._risk = RiskManager(analyst_llm)
        self._coordinator = Coordinator(coordinator_llm)

        self._fundamental = None
        if enable_fundamental:
            from agents.fundamental import FundamentalAnalyst
            self._fundamental = FundamentalAnalyst(analyst_llm, self._rag)

        self._news = None
        if enable_news and tavily_api_key:
            from agents.news import NewsAnalyst
            self._news = NewsAnalyst(analyst_llm, tavily_api_key)

        self._reflection = None
        if enable_reflection:
            from agents.reflection import ReflectionAgent
            self._reflection = ReflectionAgent(analyst_llm, self._memory)

        self._portfolio_value_series: list[float] = []
        self._cache: dict[str, TradingDecision] = {}
        self._lock = threading.Lock()

        # Exposed for UI rendering (thread-safe via _lock)
        self._last_votes: list[AgentVote] = []
        self._last_lessons: list[str] = []
        self._last_token_log: list[dict] = []

    # ------------------------------------------------------------------
    # BaseProvider contract
    # ------------------------------------------------------------------

    def decide(self, context: dict) -> TradingDecision:
        cache_key = self._make_cache_key(context)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        ticker = context.get("ticker", "UNKNOWN")
        bar_date = context.get("bar_date", "")

        # Ensure RAG is loaded for this ticker (no-op if already cached)
        if self._fundamental and bar_date:
            try:
                self._rag.ensure_loaded(ticker, bar_date)
            except Exception:
                pass

        # Enrich context with portfolio value series for RiskManager
        total_val = context.get("portfolio", {}).get("total_value", 0)
        self._portfolio_value_series.append(float(total_val))
        enriched = dict(context)
        enriched["portfolio_value_series"] = list(self._portfolio_value_series)

        votes = self._run_agents_parallel(enriched)
        token_log = [v.token_usage for v in votes if v.token_usage]

        lessons = []
        try:
            lessons = self._memory.retrieve(ticker, query=f"{ticker} trading lesson", top_k=3)
        except Exception:
            pass

        decision = self._coordinator.synthesize(votes, enriched, lessons)

        # Collect coordinator token usage
        coord_usage = getattr(decision, "coordinator_token_usage", None)
        if coord_usage:
            token_log.append({**coord_usage, "model": "gpt-4o"})

        if self._reflection and bar_date:
            self._reflection.reflect(ticker, bar_date, enriched, votes, decision)

        with self._lock:
            self._last_votes = list(votes)
            self._last_lessons = list(lessons)
            self._last_token_log = token_log
            self._cache[cache_key] = decision

        return decision

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._portfolio_value_series.clear()

    # ------------------------------------------------------------------
    # UI-facing properties
    # ------------------------------------------------------------------

    @property
    def last_votes(self) -> list[AgentVote]:
        with self._lock:
            return list(self._last_votes)

    @property
    def last_lessons(self) -> list[str]:
        with self._lock:
            return list(self._last_lessons)

    @property
    def last_token_log(self) -> list[dict]:
        with self._lock:
            return list(self._last_token_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, context: dict) -> str:
        ticker = context.get("ticker", "")
        bar_date = context.get("bar_date", "")
        cash = context.get("portfolio", {}).get("cash", 0)
        shares = context.get("portfolio", {}).get("shares", 0)
        return f"{ticker}_{bar_date}_{cash:.0f}_{shares}"

    def _run_agents_parallel(self, context: dict) -> list[AgentVote]:
        from agents.base import AgentVote as AV
        # Belt-and-suspenders: ensure openai submodules are in sys.modules before
        # the executor spawns threads, preventing _ModuleLock deadlocks.
        try:
            import openai.resources.embeddings  # noqa: F401
            import openai.resources.chat.completions  # noqa: F401
        except ImportError:
            pass

        agents = [("technical", self._technical)]
        if self._fundamental:
            agents.append(("fundamental", self._fundamental))
        if self._news:
            agents.append(("news", self._news))
        agents.append(("risk", self._risk))

        results: list[AV] = []
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            future_to_name = {
                pool.submit(agent.vote, context): name
                for name, agent in agents
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(AV(
                        agent_name=name,
                        action="HOLD",
                        confidence=0.5,
                        reasoning=f"Agent failed: {e}",
                        evidence=[f"Error: {e}"],
                    ))

        # Preserve a stable order: Technical, Fundamental, News, Risk
        order = ["TechnicalAnalyst", "FundamentalAnalyst", "NewsAnalyst", "RiskManager"]
        results.sort(key=lambda v: order.index(v.agent_name) if v.agent_name in order else 99)
        return results
