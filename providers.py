import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TradingDecision:
    action: str        # BUY | SELL | HOLD | STRONG_BUY | STRONG_SELL
    confidence: float  # 0.0–1.0
    reasoning: str     # free-text explanation from AI


_VALID_ACTIONS = {"BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"}

# ---------------------------------------------------------------------------
# Personalities — each changes the AI's opening persona and decision style
# ---------------------------------------------------------------------------
PERSONALITIES = {
    "balanced": {
        "label": "Balanced Trader",
        "persona": (
            "You are a balanced algorithmic trader. You weigh both short-term momentum "
            "and longer-term trend quality equally. You aim for a mix of capital "
            "preservation and growth, making trades when signals clearly align and "
            "avoiding overtrading. You are comfortable using the full range of actions "
            "(BUY, SELL, STRONG_BUY, STRONG_SELL, HOLD) as appropriate."
        ),
    },
    "vfa": {
        "label": "Virtual Financial Advisor",
        "persona": (
            "You are a conservative Virtual Financial Advisor (VFA) with a long-term "
            "perspective. Your primary goals are capital preservation and steady "
            "portfolio growth over months and years — not quick gains. You STRONGLY "
            "prefer HOLD and only recommend BUY when there is a clear, sustained "
            "uptrend (price above SMA50 and SMA200, RSI between 40-65, positive MACD "
            "histogram). Only recommend SELL when there is clear trend deterioration "
            "and significant downside risk. You almost never use STRONG_BUY or "
            "STRONG_SELL. Minimise trade frequency — fewer, higher-conviction trades "
            "are far better than many marginal ones. When in doubt, HOLD."
        ),
    },
    "day_trader": {
        "label": "Day Trader",
        "persona": (
            "You are an aggressive day trader focused exclusively on maximising "
            "short-term intraday profits. You act decisively and quickly on momentum "
            "signals — hesitation costs money. Volume spikes, MACD crossovers, RSI "
            "momentum shifts, and Bollinger Band breakouts are your primary triggers. "
            "Use STRONG_BUY and STRONG_SELL freely when multiple signals align — "
            "that is where the biggest returns come from. You accept higher risk for "
            "higher reward. You are comfortable entering and exiting positions rapidly. "
            "Prioritise action over caution; a missed move is a missed profit."
        ),
    },
    "swing_trader": {
        "label": "Swing Trader",
        "persona": (
            "You are a swing trader looking to capture multi-day to multi-week price "
            "moves. You buy pullbacks in established uptrends and sell bounces in "
            "established downtrends. RSI dipping below 40 in an uptrend is a strong "
            "buy signal; RSI pushing above 60 in a downtrend is a strong sell signal. "
            "A MACD histogram turning from negative to positive confirms a buy setup; "
            "negative-turning confirms a sell. Hammer and bullish engulfing patterns "
            "at support are high-conviction entries. You hold positions for days to "
            "weeks, so you are not fazed by single-bar noise."
        ),
    },
    "contrarian": {
        "label": "Contrarian",
        "persona": (
            "You are a contrarian trader who profits by going against prevailing "
            "market sentiment. You BUY when others are fearful: RSI below 35, price "
            "well below SMA20, hammer or doji patterns signalling exhaustion, and "
            "high relative volume on down days (capitulation). You SELL when others "
            "are greedy: RSI above 65, price far above SMA20, bearish engulfing at "
            "highs. You are most interested in extreme readings — moderate signals "
            "warrant HOLD. The bigger the recent move in one direction, the more you "
            "lean against it."
        ),
    },
}


def _build_prompt(context: dict) -> str:
    signals    = context.get("signals", {})
    portfolio  = context.get("portfolio", {})
    historical = context.get("historical", {})
    recent_trades = context.get("recent_trades", [])
    personality_key = context.get("personality", "balanced")
    persona = PERSONALITIES.get(personality_key, PERSONALITIES["balanced"])["persona"]

    buy_thresh  = context.get("buy_thresh",  0.65)
    sell_thresh = context.get("sell_thresh", 0.35)

    if recent_trades:
        trades_str = "\n".join(
            f"  {t.get('date', '?')}: {t.get('type', '?')} "
            f"{t.get('qty', '?')} share(s) @ ${float(t.get('price', 0)):.2f}"
            for t in recent_trades
        )
    else:
        trades_str = "  No trades yet."

    # Historical context block — only rendered if data is present
    def _fmt_pct(v):
        if v is None:
            return "N/A"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"

    hist_section = ""
    if historical:
        n     = historical.get("bars_available", 0)
        trend = historical.get("trend_alignment", "unknown")
        hist_section = f"""
HISTORICAL CONTEXT (computed from last {n} bars):
  Return  5 bars:    {_fmt_pct(historical.get('return_5bar'))}
  Return 20 bars:    {_fmt_pct(historical.get('return_20bar'))}
  Return 60 bars:    {_fmt_pct(historical.get('return_60bar'))}
  Period high:       ${historical.get('period_high', 0):.4f}  (current is {_fmt_pct(historical.get('pct_from_high'))} from high)
  Period low:        ${historical.get('period_low', 0):.4f}  (current is {_fmt_pct(historical.get('pct_from_low'))} from low)
  Trend alignment:   {trend}
"""

    return f"""{persona}
You are making a paper trading decision (no real money at risk).

TICKER: {context.get('ticker', 'UNKNOWN')}
CURRENT PRICE: ${context.get('current_price', 0):.4f}

OHLCV (current bar):
  Open:   ${context.get('open', 0):.4f}
  High:   ${context.get('high', 0):.4f}
  Low:    ${context.get('low', 0):.4f}
  Volume: {context.get('volume', 0):,}
{hist_section}
TECHNICAL INDICATORS:
  RSI (14):          {signals.get('rsi', 50):.1f}  [overbought >70, oversold <30]
  MACD Histogram:    {signals.get('macd_hist', 0):.4f}  [positive = bullish momentum, negative = bearish]
  Bollinger %B:      {signals.get('bb_pct', 0.5):.3f}  [0=lower band, 0.5=midline, 1=upper band]
  Price vs SMA20:    {signals.get('price_vs_sma20', 0):+.2%}
  Price vs SMA50:    {signals.get('price_vs_sma50', 0):+.2%}

CANDLESTICK PATTERNS (current bar):
  Hammer:             {signals.get('is_hammer', False)}
  Bullish Engulfing:  {signals.get('is_bull_engulf', False)}
  Bearish Engulfing:  {signals.get('is_bear_engulf', False)}
  Doji:               {signals.get('is_doji', False)}

PORTFOLIO STATE:
  Cash:        ${portfolio.get('cash', 0):.2f}
  Shares Held: {portfolio.get('shares', 0)}
  Total Value: ${portfolio.get('total_value', 0):.2f}

RECENT TRADE HISTORY (last 5):
{trades_str}

TRADING PARAMETERS:
  Buy confidence threshold:  {buy_thresh}  — lean toward BUY only if your confidence meets or exceeds this
  Sell confidence threshold: {sell_thresh} — lean toward SELL only if your confidence is at or below this

Based on the above data and your trading personality, provide your decision.

You MUST respond with ONLY a valid JSON object in this exact format, with no other text:
{{
  "action": "HOLD",
  "confidence": 0.75,
  "reasoning": "Brief explanation of your decision in 1-3 sentences."
}}

The "action" field must be exactly one of: BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
The "confidence" field must be a number between 0.0 and 1.0
The "reasoning" field must be a non-empty string"""


def _parse_response(text: str) -> TradingDecision:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        return TradingDecision(
            action="HOLD", confidence=0.5,
            reasoning=f"Parse error: no JSON found in response: {text[:200]}"
        )

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        return TradingDecision(
            action="HOLD", confidence=0.5,
            reasoning=f"Parse error: invalid JSON — {e}"
        )

    action = str(data.get("action", "HOLD")).upper().strip()
    if action not in _VALID_ACTIONS:
        action = "HOLD"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = str(data.get("reasoning", "")).strip() or "(no reasoning provided)"

    return TradingDecision(action=action, confidence=confidence, reasoning=reasoning)


class BaseProvider(ABC):
    @abstractmethod
    def decide(self, context: dict) -> TradingDecision:
        """Make a trading decision given the market context dict."""
        ...


class OllamaProvider(BaseProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def decide(self, context: dict) -> TradingDecision:
        import requests
        prompt = _build_prompt(context)
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=120,
            )
            if resp.status_code == 404:
                return TradingDecision(
                    "HOLD", 0.5,
                    f"Model '{self.model}' not found in Ollama. "
                    f"Pull it first: ollama pull {self.model}",
                )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return TradingDecision("HOLD", 0.5, f"Ollama error: {data['error']}")
            text = data.get("response", "")
            return _parse_response(text)
        except requests.exceptions.ConnectionError:
            return TradingDecision("HOLD", 0.5, "Ollama connection refused — is the server running?")
        except requests.exceptions.Timeout:
            return TradingDecision("HOLD", 0.5, "Ollama request timed out after 120 seconds")
        except Exception as e:
            return TradingDecision("HOLD", 0.5, f"Ollama error: {e}")


class ClaudeProvider(BaseProvider):
    def __init__(self, model: str, api_key: str):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.model = model

    def decide(self, context: dict) -> TradingDecision:
        prompt = _build_prompt(context)
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            return _parse_response(text)
        except Exception as e:
            return TradingDecision("HOLD", 0.5, f"Claude error: {e}")


class GeminiProvider(BaseProvider):
    def __init__(self, model: str, api_key: str):
        try:
            from google.genai import Client
            self._client = Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )
        self.model = model

    def decide(self, context: dict) -> TradingDecision:
        prompt = _build_prompt(context)
        try:
            resp = self._client.models.generate_content(model=self.model, contents=prompt)
            text = resp.text
            return _parse_response(text)
        except Exception as e:
            return TradingDecision("HOLD", 0.5, f"Gemini error: {e}")


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.model = model

    def decide(self, context: dict) -> TradingDecision:
        prompt = _build_prompt(context)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            text = resp.choices[0].message.content or ""
            return _parse_response(text)
        except Exception as e:
            return TradingDecision("HOLD", 0.5, f"OpenAI error: {e}")


def create_provider(name: str, model: str, api_key: str = "") -> BaseProvider:
    name = name.lower().strip()
    if name == "ollama":
        return OllamaProvider(model=model or "llama3")
    elif name == "claude":
        if not api_key:
            raise ValueError("API key required for Claude provider")
        return ClaudeProvider(model=model or "claude-sonnet-4-6", api_key=api_key)
    elif name == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini provider")
        return GeminiProvider(model=model or "gemini-2.0-flash", api_key=api_key)
    elif name == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI provider")
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: '{name}'. Choose from: ollama, claude, gemini, openai")
