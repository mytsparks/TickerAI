# TickerAI — Agentic AI for Paper Trading

TickerAI is a paper-trading research platform that demonstrates how a **multi-agent AI committee** can make more robust, transparent, and explainable financial decisions than a single LLM call. Built as a BUS-S 579 group project at the Kelley School of Business, it shows the practical application of agentic AI orchestration — specialist agents, memory, retrieval-augmented generation (RAG), and adversarial robustness — all within a domain where decision quality is objectively measurable.

---

## Why Agentic AI for Trading Decisions?

A single LLM call produces one answer with one perspective. Real investment committees deliberately use specialists who challenge each other: a technician reads price action, a fundamentals analyst reads the balance sheet, a risk manager asks "what could go wrong," and a news analyst tracks sentiment. Each brings a distinct information source and a different failure mode.

TickerAI replicates this structure in software. Each agent is an independent LLM with its own system prompt, its own data source, and its own vote. A Coordinator agent then synthesizes the votes — weighing agreement, disagreement, and past lessons stored in memory — into a single committed trading decision. No single agent can hijack the outcome.

This architecture delivers concrete business value:

- **Reduced false signals.** Technical momentum alone generates many false positives; cross-checking with fundamentals and risk metrics filters them out.
- **Explainability.** Every decision surfaces the vote, confidence score, and key evidence from each agent, satisfying audit and compliance requirements that black-box systems cannot.
- **Specialization without cost.** Each agent only sees the data relevant to its role. The technician never touches news; the risk manager never touches SEC filings. This reduces token consumption and keeps each agent's reasoning focused.
- **Institutional memory.** A ReflectionAgent writes lessons to persistent storage after each decision. Future decisions for the same ticker and market regime retrieve those lessons, allowing the system to avoid repeating past mistakes.
- **Adversarial robustness.** News data is untrusted external content. The system includes a measurable defense layer against prompt injection attacks embedded in news headlines — a real threat in production financial AI systems.

---

## Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │         CommitteeProvider        │
                        │  (drop-in BaseProvider for       │
                        │   backtest.py / simulation.py)   │
                        └──────────────┬──────────────────┘
                                       │ parallel dispatch
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼                        ▼
   ┌─────────────────┐    ┌─────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ TechnicalAnalyst│    │ FundamentalAnalyst   │   │  NewsAnalyst     │   │  RiskManager     │
   │                 │    │                      │   │                  │   │                  │
   │ RSI, MACD, BB,  │    │ SEC EDGAR 10-Q/10-K  │   │ Tavily news      │   │ Concentration,   │
   │ SMA, ATR,       │    │ via RAG (chunked +   │   │ search +         │   │ drawdown, ATR    │
   │ candlestick     │    │ embedded, disk cache)│   │ injection defense│   │ volatility       │
   └────────┬────────┘    └──────────┬──────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                        │                        │                       │
            └────────────────────────┴────────────────────────┴───────────────────────┘
                                                    │ AgentVotes
                                                    ▼
                                          ┌──────────────────┐
                                          │   Coordinator    │◄── MemoryStore
                                          │   (synthesizes   │    (past lessons,
                                          │    votes +       │     cosine retrieval)
                                          │    memory)       │
                                          └────────┬─────────┘
                                                   │ TradingDecision
                                                   ▼
                                          ┌──────────────────┐
                                          │ ReflectionAgent  │──► MemoryStore
                                          │ (async, bounded  │    (writes new lesson)
                                          │  thread pool)    │
                                          └──────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Each agent is an independent LLM call | Prevents one agent's framing from contaminating another's reasoning |
| Coordinator uses a separate (stronger) model | Vote synthesis requires understanding nuance; specialist agents only need to read structured data |
| `CommitteeProvider` is a drop-in `BaseProvider` | Backtest and simulation code required zero changes to support the committee — v1 and v2 are swappable at runtime |
| Parallel agent execution | Four LLM calls in parallel vs. sequential; reduces wall-clock latency by ~3–4× |
| Bounded `ThreadPoolExecutor` for reflection | Prevents 252 simultaneous reflection threads during a full-year backtest |
| Retrieved news in a separate message, not the system prompt | Channel separation is a structural defense against prompt injection — the model is told explicitly the second message is untrusted data |

---

## Application Modes

### 1. Live Trading

**What it does:** Runs a real-time paper-trading simulation against live market data pulled from Yahoo Finance. The configured provider (v1 single-LLM or v2 committee) makes a trading decision on each incoming tick. Portfolio value, trades, and a price chart update live in the browser.

**Business value:** Demonstrates that the agentic committee operates within the latency constraints of intraday decision-making — the committee's parallel architecture means adding three more specialists does not add three times the latency. This is critical for production viability.

---

### 2. Backtest

**What it does:** Runs the provider bar-by-bar over a historical OHLCV series for a chosen ticker, date range, and starting budget. Produces a complete equity curve, trade log, Sharpe ratio, max drawdown, and win rate. Both v1 (single LLM) and v2 (committee) can be backtested independently for direct comparison.

**Business value:** Historical simulation is the standard validation method in quantitative finance before any strategy is deployed with real capital. By exposing this to the committee architecture, the project lets stakeholders ask the same question they would ask of any strategy: *does this actually work on historical data, across different market conditions?*

---

### 3. Committee

**What it does:** Runs the full multi-agent committee on a single ticker snapshot and renders each agent's vote in an individual card showing: action (BUY / SELL / HOLD / STRONG\_BUY / STRONG\_SELL), confidence score, reasoning, and key evidence bullets. The Coordinator's synthesis and any retrieved memory lessons are shown beneath the agent cards.

**Business value:** This is the transparency layer. In a real deployment, a portfolio manager or compliance officer needs to understand *why* the system reached a decision, not just what it decided. The committee view provides a full audit trail — which specialists agreed, which dissented, what evidence each cited, and what past experience the Coordinator consulted. This directly addresses the explainability requirements increasingly mandated by financial regulators.

**Agents in detail:**

| Agent | Data Source | Role |
|---|---|---|
| TechnicalAnalyst | RSI, MACD, Bollinger Bands, SMA20/50, ATR, candlestick patterns | Reads price action and momentum |
| FundamentalAnalyst | SEC EDGAR 10-Q and 10-K filings (RAG retrieval) | Assesses revenue trends, earnings, debt, cash flow |
| NewsAnalyst | Live news via Tavily search API | Evaluates recent sentiment and material events |
| RiskManager | Portfolio concentration, drawdown from peak, ATR volatility | Asks whether the portfolio can afford to act on bullish signals |
| Coordinator | All agent votes + MemoryStore lessons | Synthesizes into a single committed decision |
| ReflectionAgent | Coordinator decision + outcome | Writes a one-sentence lesson to persistent memory |

---

### 4. Evaluation

**What it does:** Runs a structured performance grid comparing **v1** (single-LLM), **v2** (committee), and **buy-and-hold** across multiple tickers and market regimes (bull, bear/crash). For each cell the harness computes: total return %, Sharpe ratio, maximum drawdown %, win rate, turnover, estimated LLM cost in USD, and total trade count.

**Business value:** This is the core business case for the committee architecture. Side-by-side metrics across different market regimes answer the fundamental question that any enterprise AI buyer asks before a procurement decision: *does the more sophisticated (and more expensive) system deliver measurably better risk-adjusted returns?* The cost column makes the tradeoff explicit — the committee consumes more tokens, and the evaluation shows whether those tokens buy better performance.

**Evaluation metrics:**

| Metric | What it measures |
|---|---|
| Total Return % | Absolute P&L relative to starting capital |
| Sharpe Ratio | Return per unit of risk (annualized, daily returns, 4% risk-free rate) |
| Max Drawdown % | Worst peak-to-trough decline — the key risk metric for institutional investors |
| Win Rate | Percentage of closed trades that were profitable |
| Turnover | Number of round-trip trades — higher turnover inflates cost and market impact |
| Est. LLM Cost (USD) | Token cost at published API pricing — quantifies the committee premium |

---

### 5. Adversarial Testing

**What it does:** Runs a suite of 20 crafted attack headlines through the NewsAnalyst to measure the robustness of its prompt injection defenses. Attacks include direct instruction overrides, system prefix spoofing, embedded fake JSON, unicode bidirectional tricks, null byte injection, role confusion, and emotional manipulation. Two benign control headlines that should not alter votes are also included. Results show the clean baseline action, the action under attack, whether the injection succeeded, and the overall attack success rate vs. the <15% target.

**Business value:** News data is the highest-risk input surface in a production financial AI system. News headlines are written by humans with adversarial intent, syndicated through third-party APIs, and fed directly into LLM prompts. A single successful injection that flips a SELL to a STRONG\_BUY is not a theoretical concern — it is a material financial risk. This tab demonstrates that the system has a *measurable* and *testable* defense posture, not just a disclaimer. The toggle between defense-on and defense-off quantifies the value of the defensive layer.

**Defense mechanisms:**

| Mechanism | Implementation |
|---|---|
| Channel separation | Retrieved news is placed in a second `user` message explicitly labeled as untrusted read-only data, structurally separate from the instruction prompt |
| Text sanitization | Lines matching injection patterns (`ignore previous`, `system:`, `override`, `forget context`, `you are now`, etc.) are stripped before the text reaches the model |
| Unicode normalization | NFKC normalization strips bidirectional override characters and null bytes that can hide injected text |
| Truncation | News text is capped at 2000 characters, limiting the attack surface for context window pollution |

---

## Provider Support

TickerAI is provider-agnostic. The `LLMClient` wrapper normalizes all providers to a single `.chat()` interface so agents have no dependency on a specific SDK:

| Provider | Configuration |
|---|---|
| OpenAI (or any OpenAI-compatible endpoint) | API key + optional base URL (e.g. a university gateway) |
| Ollama (local models) | No key required; auto-routes to `http://localhost:11434/v1` |
| Anthropic Claude | Anthropic API key |
| Google Gemini | Google AI API key |

---

## Technical Stack

| Component | Technology |
|---|---|
| UI | Dash + Dash Bootstrap Components |
| Market data | yfinance |
| LLM inference | OpenAI SDK, Anthropic SDK, google-genai |
| News retrieval | Tavily Search API |
| SEC filing retrieval | SEC EDGAR public API |
| Embeddings / RAG | `text-embedding-3-small` + cosine similarity (NumPy) |
| Persistent memory | JSON file + embedding-based retrieval |
| Concurrency | `threading.Thread` + `ThreadPoolExecutor` |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add Tavily key for live news (optional — NewsAnalyst degrades gracefully without it)
echo "TAVILY_API_KEY=tvly-..." > .env

# 3. Run
python app.py
```

Open `http://127.0.0.1:8050` in a browser. Configure your LLM provider via the **Settings** (⚙) panel before using the Committee, Evaluation, or Adversarial tabs.

---

## Project Structure

```
app.py               — Dash UI; all callbacks; 5 tabs
state.py             — AppState dataclass shared by all callbacks
providers.py         — BaseProvider ABC + single-LLM providers
backtest.py          — Bar-by-bar historical backtest engine
simulation.py        — Live tick simulation loop
engine.py            — TradingEngine; builds context dict from OHLCV + indicators
committee.py         — CommitteeProvider; parallel multi-agent orchestration
llm_client.py        — Provider-agnostic LLM wrapper (OpenAI / Claude / Gemini / Ollama)
memory.py            — MemoryStore; JSON persistence + embedding-based retrieval
rag.py               — RAGStore; SEC EDGAR fetch, chunk, embed, disk cache
eval_harness.py      — Sharpe, drawdown, win rate, evaluation grid runner
adversarial.py       — 20 injection attacks, defense measurement

agents/
  base.py            — Agent ABC + AgentVote dataclass
  technical.py       — TechnicalAnalyst
  fundamental.py     — FundamentalAnalyst
  news.py            — NewsAnalyst + prompt injection defense
  risk.py            — RiskManager
  coordinator.py     — Coordinator
  reflection.py      — ReflectionAgent
```
