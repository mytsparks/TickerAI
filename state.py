import threading
import pandas as pd
from dataclasses import dataclass, field
from engine import TradingEngine


@dataclass
class AppState:
    engine: TradingEngine = field(default_factory=TradingEngine)
    provider: object = field(default=None)   # BaseProvider instance | None
    provider_name: str = ""                  # "ollama" | "claude" | "gemini"
    api_key: str = ""                        # raw API key for committee/eval/adversarial reuse
    api_base_url: str = "https://hub.kelley.iu.edu/llmapi/v1"  # custom base URL for OpenAI-compatible endpoints
    ai_model: str = ""
    personality: str = "balanced"
    portfolio: dict = field(default_factory=lambda: {
        "cash": 1000.0,
        "shares": 0,
        "initial_cash": 1000.0,
    })
    trade_log: list = field(default_factory=list)
    candle_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_prob: float = 0.5
    last_action: str = "—"
    last_signals: dict = field(default_factory=dict)
    last_reasoning: str = ""
    pull_status: str = ""
    pull_active: bool = False
    backtest_progress: float = 0.0
    backtest_running: bool = False
    backtest_result: object = field(default=None)
    ticker: str = ""
    running: bool = False
    status_msg: str = "Idle — configure an AI provider to begin"
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: object = field(default=None)  # threading.Thread | None

    # Committee tab
    committee_provider: object = field(default=None)   # CommitteeProvider | None
    committee_ticker: str = ""
    committee_running: bool = False
    committee_progress: int = 0                        # 0-100
    committee_votes: list = field(default_factory=list)
    committee_decision: object = field(default=None)
    committee_lessons: list = field(default_factory=list)
    committee_status: str = ""

    # Evaluation tab
    eval_running: bool = False
    eval_progress_done: int = 0
    eval_progress_total: int = 0
    eval_grid_result: list = field(default_factory=list)
    eval_thread: object = field(default=None)

    # Adversarial tab
    adversarial_running: bool = False
    adversarial_progress: int = 0                      # 0-20 (attacks completed)
    adversarial_results: list = field(default_factory=list)
    adversarial_thread: object = field(default=None)


# Module-level singleton shared between app.py and simulation.py
state = AppState()
