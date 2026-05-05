import json
import math
import os
import threading
from datetime import date, timedelta

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ; no-op if file absent

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Input, Output, State, MATCH, dcc, html

from backtest import BacktestResult, run_backtest
from simulation import run_simulation
from state import state

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "TickerAI"

# ---------------------------------------------------------------------------
# Reusable layout helpers
# ---------------------------------------------------------------------------

def _label(text):
    return html.Label(text, className="mt-3 mb-1 fw-semibold small text-secondary")


def _card(header, body_id):
    return dbc.Card(
        [dbc.CardHeader(header, className="py-2 small text-secondary"),
         dbc.CardBody(html.H5("—", id=body_id, className="mb-0"))],
        className="h-100",
    )


# ---------------------------------------------------------------------------
# Chart helpers — defined before layout so they can be used as initial values
# ---------------------------------------------------------------------------

def _make_empty_fig(message: str):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#222", plot_bgcolor="#222",
        font_color="#aaa",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        annotations=[dict(text=message, x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False,
                          font=dict(size=16, color="#aaa"))],
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------
_CLAUDE_MODELS = [
    {"label": "claude-sonnet-4-6",         "value": "claude-sonnet-4-6"},
    {"label": "claude-opus-4-7",           "value": "claude-opus-4-7"},
    {"label": "claude-haiku-4-5-20251001", "value": "claude-haiku-4-5-20251001"},
]
_GEMINI_MODELS = [
    {"label": "gemini-2.0-flash", "value": "gemini-2.0-flash"},
    {"label": "gemini-1.5-pro",   "value": "gemini-1.5-pro"},
    {"label": "gemini-1.5-flash", "value": "gemini-1.5-flash"},
]


# ---------------------------------------------------------------------------
# Backtest default dates
# ---------------------------------------------------------------------------
_BT_DEFAULT_END   = date.today() - timedelta(days=1)
_BT_DEFAULT_START = _BT_DEFAULT_END - timedelta(days=182)

# ---------------------------------------------------------------------------
# Provider offcanvas (settings pane)
# ---------------------------------------------------------------------------
provider_offcanvas = dbc.Offcanvas(
    [
        html.H6("AI Provider Settings", className="text-uppercase text-secondary mb-3"),
        html.Label("Provider", className="mb-1 fw-semibold small text-secondary"),
        dbc.Select(
            id="dd-provider",
            options=[
                {"label": "Ollama (Local)",     "value": "ollama"},
                {"label": "Claude (Anthropic)", "value": "claude"},
                {"label": "Google Gemini",      "value": "gemini"},
                {"label": "OpenAI",             "value": "openai"},
            ],
            value="ollama",
            size="sm",
            className="mb-3",
        ),
        dbc.Row([
            dbc.Col(
                html.Label("Model", className="mb-1 fw-semibold small text-secondary"),
                width=True,
            ),
            dbc.Col(
                dbc.Button("↻", id="btn-refresh-models", size="sm", color="secondary",
                           outline=True, className="py-0 px-1 mb-1",
                           title="Refresh Ollama model list"),
                width="auto", id="refresh-btn-col",
            ),
        ], className="align-items-end g-0"),
        dbc.Select(
            id="dd-ai-model",
            options=[{"label": "llama3", "value": "llama3"}],
            value="llama3",
            size="sm",
            className="mb-3",
        ),
        dbc.Input(
            id="inp-model-custom",
            type="text",
            size="sm",
            placeholder="e.g. gpt-4o-mini, gpt-oss-20b…",
            value="gpt-oss-20b",
            className="mb-3",
            style={"display": "none"},
        ),
        html.Div([
            html.Label("API Key", className="mb-1 fw-semibold small text-secondary"),
            dbc.Input(id="inp-api-key", type="password", size="sm",
                      placeholder="Paste API key…"),
        ], id="api-key-container", style={"display": "none"}, className="mb-3"),
        html.Div([
            html.Label("API Base URL", className="mb-1 fw-semibold small text-secondary"),
            dbc.Input(id="inp-base-url", type="text", size="sm",
                      value="https://hub.kelley.iu.edu/llmapi/v1",
                      placeholder="https://api.openai.com/v1  (leave blank for default)"),
            html.P("Set this if you use a custom / compatible endpoint.",
                   className="text-secondary mt-1 mb-0", style={"fontSize": "0.72rem"}),
        ], id="base-url-container", style={"display": "none"}, className="mb-3"),
        html.Label("Personality", className="mb-1 fw-semibold small text-secondary"),
        dbc.Select(
            id="dd-personality",
            options=[
                {"label": "Balanced Trader",          "value": "balanced"},
                {"label": "Virtual Financial Advisor", "value": "vfa"},
                {"label": "Day Trader",                "value": "day_trader"},
                {"label": "Swing Trader",              "value": "swing_trader"},
                {"label": "Contrarian",                "value": "contrarian"},
            ],
            value="balanced",
            size="sm",
            className="mb-3",
        ),
        dbc.Button("Apply Config", id="btn-apply-provider", color="primary",
                   size="sm", className="w-100 mb-2"),
        html.Div(id="provider-msg", className="mt-1 small"),
        html.Hr(className="my-3"),
        html.Div([
            html.Label("Pull Model", className="mb-1 fw-semibold small text-secondary"),
            dbc.InputGroup([
                dbc.Input(id="inp-pull-model", size="sm",
                          placeholder="e.g. llama3, phi3:mini, mistral:7b"),
                dbc.Button("Pull", id="btn-pull-model", color="secondary", size="sm"),
            ]),
            html.Div(id="pull-status-display", className="small text-secondary mt-2",
                     style={"fontFamily": "monospace", "wordBreak": "break-all"}),
        ], id="pull-section"),
    ],
    id="provider-offcanvas",
    title="",
    placement="end",
    style={"width": "360px"},
    is_open=False,
)

# ---------------------------------------------------------------------------
# Live sidebar
# ---------------------------------------------------------------------------
sidebar = dbc.Card(
    dbc.CardBody([
        html.H6("Settings", className="text-uppercase text-secondary mb-3"),

        _label("Ticker Symbol"),
        dbc.Input(id="inp-ticker", value="NVDA", type="text", size="sm"),

        _label("Starting Budget ($)"),
        dbc.Input(id="inp-budget", value=1000, type="number", min=1, size="sm"),

        _label("Candle Interval"),
        dbc.Select(
            id="dd-candle",
            options=[{"label": l, "value": v} for l, v in [
                ("1 Minute", "1m"), ("5 Minutes", "5m"), ("15 Minutes", "15m")
            ]],
            value="1m",
        ),

        html.Hr(className="my-3"),

        _label("Loss Limit (%) — 0 to disable"),
        dbc.Input(id="inp-drawdown", type="number", min=0, max=100,
                  step=1, value=20, size="sm"),
        html.Div(id="drawdown-msg"),

        html.Hr(className="my-3"),

        dbc.Button("Start Simulation", id="btn-start", color="success",
                   size="sm", className="w-100 mb-2"),
        dbc.Button("Stop Simulation", id="btn-stop", color="danger",
                   size="sm", className="w-100 mb-2"),

        html.Div(id="sidebar-msg", className="mt-2 small"),
    ]),
    className="h-100",
)

# ---------------------------------------------------------------------------
# Backtest sidebar
# ---------------------------------------------------------------------------
bt_sidebar = dbc.Card(
    dbc.CardBody([
        html.H6("Backtest Settings", className="text-uppercase text-secondary mb-3"),

        _label("Ticker Symbol"),
        dbc.Input(id="bt-ticker", value="NVDA", type="text", size="sm"),

        _label("Test Window Start"),
        dbc.Input(id="bt-start-date", type="date",
                  value=_BT_DEFAULT_START.isoformat(), size="sm"),

        _label("Test Window End"),
        dbc.Input(id="bt-end-date", type="date",
                  value=_BT_DEFAULT_END.isoformat(), size="sm"),

        _label("Starting Budget ($)"),
        dbc.Input(id="bt-budget", value=1000, type="number", min=1, size="sm"),

        html.Hr(className="my-3"),

        _label("Loss Limit (%) — 0 to disable"),
        dbc.Input(id="bt-inp-drawdown", type="number", min=0, max=100,
                  step=1, value=20, size="sm"),
        html.Div(id="bt-drawdown-msg"),

        html.Hr(className="my-3"),

        dbc.Button("Run Backtest", id="bt-run-btn", color="info",
                   size="sm", className="w-100 mb-2"),

        html.Div(id="bt-sidebar-msg", className="mt-2 small"),
    ]),
    className="h-100",
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        provider_offcanvas,

        # Header — pill is absolutely centred so it's never pushed by title/button widths
        html.Div(
            [
                html.Div(html.H4("TickerAI", className="mb-0"),
                         style={"position": "absolute", "left": "0"}),

                html.Div(
                    dbc.RadioItems(
                        id="mode-switch",
                        options=[
                            {"label": "Live",        "value": "live"},
                            {"label": "Test",        "value": "backtest"},
                            {"label": "Committee",   "value": "committee"},
                            {"label": "Evaluation",  "value": "eval"},
                            {"label": "Adversarial", "value": "adversarial"},
                        ],
                        value="live",
                        inline=True,
                        input_class_name="btn-check",
                        label_class_name="btn btn-sm",
                        label_checked_class_name="active",
                        class_name="btn-group",
                    ),
                    className="pill-switcher",
                ),

                html.Div(
                    [
                        html.Div(id="status-badge"),
                        dbc.Button("⚙", id="btn-open-settings", color="secondary",
                                   outline=True, size="sm",
                                   style={"fontSize": "1rem", "lineHeight": "1"}),
                    ],
                    className="d-flex align-items-center gap-2",
                    style={"position": "absolute", "right": "0"},
                ),
            ],
            style={
                "position":      "relative",
                "display":       "flex",
                "alignItems":    "center",
                "justifyContent":"center",
                "padding":       "1rem 0",
                "borderBottom":  "1px solid rgba(255,255,255,0.15)",
                "marginBottom":  "1rem",
            },
        ),

        # ----------------------------------------------------------------
        # LIVE content
        # ----------------------------------------------------------------
        html.Div([
            dbc.Row([
                dbc.Col(sidebar, width=3),
                dbc.Col([
                    dcc.Graph(id="candle-chart", style={"height": "420px"}),
                    dbc.Row([
                        dbc.Col(_card("Portfolio Value", "metric-value"),  width=4),
                        dbc.Col(_card("AI Confidence",   "metric-prob"),   width=4),
                        dbc.Col(_card("Last Action",     "metric-action"), width=4),
                    ], className="mt-3 g-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H6("AI Reasoning",
                                    className="text-secondary text-uppercase mb-1 small mt-3"),
                            html.Div(
                                "Configure an AI provider and start the simulation to see"
                                " reasoning here.",
                                id="reasoning-display",
                                className="small text-secondary p-2",
                                style={
                                    "background":   "#1a1a1a",
                                    "borderRadius": "4px",
                                    "minHeight":    "48px",
                                    "whiteSpace":   "pre-wrap",
                                    "fontFamily":   "monospace",
                                },
                            ),
                        ]),
                    ]),
                ], width=9),
            ], className="mb-3 g-3"),

            dbc.Row([
                dbc.Col([
                    html.H6("Trade Log", className="text-secondary text-uppercase mb-2"),
                    html.Div(id="trade-log"),
                ]),
            ]),
        ], id="live-content"),

        # ----------------------------------------------------------------
        # BACKTEST content
        # ----------------------------------------------------------------
        html.Div([
            dbc.Row([
                dbc.Col(bt_sidebar, width=3),
                dbc.Col([
                    dcc.Graph(id="bt-chart", style={"height": "420px"},
                              figure=_make_empty_fig("Run a backtest to see results.")),
                    html.Div([
                        dbc.Progress(id="bt-progress", value=0, striped=True, animated=True,
                                     color="info", className="mt-2",
                                     style={"height": "18px"}),
                        html.P("Querying AI for each bar…",
                               className="text-secondary small text-center mt-1 mb-0"),
                    ], id="bt-progress-container", style={"display": "none"}),
                    dbc.Row([
                        dbc.Col(_card("Final Portfolio Value", "bt-metric-value"), width=4),
                        dbc.Col(_card("Net P&L",               "bt-metric-pnl"),   width=4),
                        dbc.Col(_card("Total Trades",          "bt-metric-trades"),width=4),
                    ], className="mt-3 g-3"),
                ], width=9),
            ], className="mb-3 g-3"),

            dbc.Row([
                dbc.Col([
                    html.H6("Backtest Trade Log",
                            className="text-secondary text-uppercase mb-2"),
                    html.Div(id="bt-trade-log"),
                ]),
            ]),
        ], id="bt-content", style={"display": "none"}),

        # ----------------------------------------------------------------
        # COMMITTEE content
        # ----------------------------------------------------------------
        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H6("Committee Settings", className="text-uppercase text-secondary mb-3"),
                        html.P("Uses the provider configured via ⚙ Settings.",
                               className="small text-secondary mb-2"),
                        _label("Ticker Symbol"),
                        dbc.Input(id="cm-ticker", value="NVDA", type="text", size="sm",
                                  className="mb-3"),
                        dbc.Button("Run Committee", id="cm-run-btn", color="primary",
                                   size="sm", className="w-100 mb-2"),
                        html.Div(id="cm-status-msg", className="mt-2 small"),
                    ])), className="h-100"
                , width=3),
                dbc.Col([
                    html.Div([
                        dbc.Progress(id="cm-progress", value=0, striped=True, animated=True,
                                     color="info", style={"height": "18px"}),
                        html.P(id="cm-progress-label", className="text-secondary small text-center mt-1 mb-0"),
                    ], id="cm-progress-container", style={"display": "none"}, className="mb-3"),
                    # Analyst cards row
                    dbc.Row(id="cm-agent-cards", className="g-3 mb-3"),
                    # Coordinator card
                    html.Div(id="cm-coordinator-card"),
                    # Memory lessons panel
                    html.Div([
                        html.H6("Memory Lessons (top-3 relevant)",
                                className="text-secondary text-uppercase mt-3 mb-1 small"),
                        html.Div(id="cm-lessons-panel"),
                    ]),
                ], width=9),
            ], className="g-3"),
        ], id="committee-content", style={"display": "none"}),

        # ----------------------------------------------------------------
        # EVALUATION content
        # ----------------------------------------------------------------
        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H6("Evaluation Settings", className="text-uppercase text-secondary mb-3"),
                        html.P("Uses the provider configured via ⚙ Settings.",
                               className="small text-secondary mb-1"),
                        html.P("2 regimes × 10 bars each (bull 2023, crash 2020). "
                               "V2 uses Technical + Risk agents only.",
                               className="small text-muted mb-2"),
                        _label("Tickers (comma-separated)"),
                        dbc.Input(id="ev-tickers", value="AAPL,NVDA,TSLA",
                                  type="text", size="sm", className="mb-2"),
                        _label("Starting Budget ($)"),
                        dbc.Input(id="ev-budget", value=10000, type="number",
                                  min=1, size="sm", className="mb-3"),
                        dbc.Button("Run Evaluation Grid", id="ev-run-btn", color="info",
                                   size="sm", className="w-100 mb-2"),
                        html.Div(id="ev-status-msg", className="mt-2 small"),
                    ])), className="h-100"
                , width=3),
                dbc.Col([
                    html.Div([
                        dbc.Progress(id="ev-progress", value=0, striped=True, animated=True,
                                     color="info", style={"height": "18px"}),
                        html.P(id="ev-progress-label", className="text-secondary small text-center mt-1 mb-0"),
                    ], id="ev-progress-container", style={"display": "none"}, className="mb-3"),
                    html.Div(id="ev-grid-table"),
                ], width=9),
            ], className="g-3"),
        ], id="eval-content", style={"display": "none"}),

        # ----------------------------------------------------------------
        # ADVERSARIAL content
        # ----------------------------------------------------------------
        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H6("Adversarial Settings", className="text-uppercase text-secondary mb-3"),
                        html.P("Uses the provider configured via ⚙ Settings.",
                               className="small text-secondary mb-2"),
                        _label("Ticker context for test"),
                        dbc.Input(id="adv-ticker", value="AAPL", type="text",
                                  size="sm", className="mb-3"),
                        dbc.Checklist(
                            id="adv-defense-toggle",
                            options=[{"label": "Defenses enabled", "value": "on"}],
                            value=["on"],
                            className="mb-3",
                        ),
                        dbc.Button("Run Attack Suite", id="adv-run-btn", color="danger",
                                   size="sm", className="w-100 mb-2"),
                        html.Div(id="adv-status-msg", className="mt-2 small"),
                    ])), className="h-100"
                , width=3),
                dbc.Col([
                    html.Div([
                        dbc.Progress(id="adv-progress", value=0, striped=True, animated=True,
                                     color="danger", style={"height": "18px"}),
                        html.P(id="adv-progress-label", className="text-secondary small text-center mt-1 mb-0"),
                    ], id="adv-progress-container", style={"display": "none"}, className="mb-3"),
                    html.Div(id="adv-summary", className="mb-3"),
                    html.Div(id="adv-results-table"),
                ], width=9),
            ], className="g-3"),
        ], id="adversarial-content", style={"display": "none"}),

        dcc.Interval(id="interval",      interval=5000, n_intervals=0),
        dcc.Interval(id="bt-interval",   interval=500,  n_intervals=0, disabled=True),
        dcc.Interval(id="cm-interval",   interval=1000, n_intervals=0, disabled=True),
        dcc.Interval(id="ev-interval",   interval=1000, n_intervals=0, disabled=True),
        dcc.Interval(id="adv-interval",  interval=1000, n_intervals=0, disabled=True),
        dcc.Store(id="store-poll-ms", data=5000),
    ],
    fluid=True,
    className="px-4",
)

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_bt_figure(result: BacktestResult):
    df = result.candle_df
    if df.empty:
        return _make_empty_fig("No data to display.")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    _buy_types  = {"BUY", "STRONG_BUY"}
    _sell_types = {"SELL", "STRONG_SELL", "STOP-LOSS", "TAKE-PROFIT"}

    buy_trades  = [t for t in result.trades if t.trade_type in _buy_types]
    sell_trades = [t for t in result.trades if t.trade_type in _sell_types]

    if buy_trades:
        fig.add_trace(go.Scatter(
            x=[t.date for t in buy_trades],
            y=[t.price for t in buy_trades],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=14, color="#26a69a",
                        line=dict(width=1, color="#ffffff")),
            text=[
                f"{t.trade_type} {t.qty} share{'s' if t.qty != 1 else ''}"
                f" @ ${t.price:,.2f}<br>{t.reason[:120]}"
                for t in buy_trades
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    if sell_trades:
        fig.add_trace(go.Scatter(
            x=[t.date for t in sell_trades],
            y=[t.price for t in sell_trades],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=14, color="#ef5350",
                        line=dict(width=1, color="#ffffff")),
            text=[
                f"{t.trade_type} {t.qty} share{'s' if t.qty != 1 else ''}"
                f" @ ${t.price:,.2f}<br>{t.reason[:120]}"
                for t in sell_trades
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        uirevision="bt-chart",
        paper_bgcolor="#222", plot_bgcolor="#1a1a1a",
        font_color="#ccc",
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#333"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def _build_bt_trade_table(trades):
    if not trades:
        return html.P("No trades executed.", className="text-secondary small")

    _buy_types = {"BUY", "STRONG_BUY"}
    rows = [
        html.Tr([
            html.Td(t.date, className="text-secondary small"),
            html.Td(dbc.Badge(
                ("★ " if t.trade_type.startswith("STRONG") else "") + t.trade_type,
                color="success" if t.trade_type in _buy_types else "danger",
                className="px-2",
            )),
            html.Td(f'{t.qty} share{"s" if t.qty != 1 else ""}'),
            html.Td(f"${t.price:,.2f}"),
            html.Td(f"${t.portfolio_value:,.2f}"),
            html.Td(
                t.reason[:300] + ("…" if len(t.reason) > 300 else ""),
                className="text-secondary small",
            ),
        ])
        for t in trades
    ]
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Date"), html.Th("Type"), html.Th("Qty"),
            html.Th("Price"), html.Th("Portfolio Value"), html.Th("AI Reasoning"),
        ])),
         html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# 0. Mode toggle
@app.callback(
    Output("live-content",         "style"),
    Output("bt-content",           "style"),
    Output("committee-content",    "style"),
    Output("eval-content",         "style"),
    Output("adversarial-content",  "style"),
    Input("mode-switch",           "value"),
)
def toggle_mode(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    modes = ["live", "backtest", "committee", "eval", "adversarial"]
    return tuple(show if m == mode else hide for m in modes)


# 0b. Open/close settings offcanvas
@app.callback(
    Output("provider-offcanvas", "is_open"),
    Input("btn-open-settings",   "n_clicks"),
    State("provider-offcanvas",  "is_open"),
    prevent_initial_call=True,
)
def toggle_settings_pane(_, is_open):
    return not is_open


def _fetch_ollama_models(base_url="http://localhost:11434"):
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        return [{"label": n, "value": n} for n in names] if names else []
    except Exception:
        return []


# 1. Populate model dropdown and toggle API key / pull section
@app.callback(
    Output("api-key-container",  "style"),
    Output("refresh-btn-col",    "style"),
    Output("pull-section",       "style"),
    Output("dd-ai-model",        "options"),
    Output("dd-ai-model",        "value"),
    Output("dd-ai-model",        "style"),
    Output("inp-model-custom",   "style"),
    Output("base-url-container", "style"),
    Input("dd-provider",         "value"),
    Input("btn-refresh-models",  "n_clicks"),
    State("dd-ai-model",         "value"),
)
def on_provider_change(provider, _refresh, current_model):
    show = {"display": "block"}
    hide = {"display": "none"}
    if provider == "claude":
        return (show, hide, hide, _CLAUDE_MODELS, _CLAUDE_MODELS[0]["value"], show, hide, hide)
    elif provider == "gemini":
        return (show, hide, hide, _GEMINI_MODELS, _GEMINI_MODELS[0]["value"], show, hide, hide)
    elif provider == "openai":
        return (show, hide, hide, [], "", hide, show, show)
    else:  # ollama
        options = _fetch_ollama_models()
        if not options:
            options = [{"label": "(no models found — pull one below)", "value": ""}]
        default = (current_model
                   if any(o["value"] == current_model for o in options)
                   else options[0]["value"])
        return (hide, show, show, options, default, show, hide, hide)


# 2a. Pull Ollama model (background thread with streaming progress)
def _pull_ollama(model_name: str):
    import requests
    with state.lock:
        state.pull_active = True
        state.pull_status = f"Starting: {model_name}…"
    try:
        with requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code == 404:
                with state.lock:
                    state.pull_status = f"Unknown model '{model_name}' — check the name and try again"
                return
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if "error" in data:
                    with state.lock:
                        state.pull_status = f"Error: {data['error']}"
                    return
                status    = data.get("status", "")
                total     = data.get("total", 0)
                completed = data.get("completed", 0)
                if total and completed:
                    pct = completed / total * 100
                    msg = f"{status}  {pct:.1f}%  ({completed/1e9:.2f} / {total/1e9:.2f} GB)"
                else:
                    msg = status
                with state.lock:
                    state.pull_status = msg
        with state.lock:
            state.pull_status = f"✓ {model_name} ready — click ↻ to refresh the list"
    except requests.exceptions.ConnectionError:
        with state.lock:
            state.pull_status = "Ollama connection refused — is the server running?"
    except Exception as exc:
        with state.lock:
            state.pull_status = f"Pull failed: {exc}"
    finally:
        with state.lock:
            state.pull_active = False


@app.callback(
    Output("pull-status-display", "children", allow_duplicate=True),
    Input("btn-pull-model", "n_clicks"),
    State("inp-pull-model", "value"),
    prevent_initial_call=True,
)
def start_pull(_, model_name):
    model_name = (model_name or "").strip()
    if not model_name:
        return dbc.Alert("Enter a model name to pull.", color="warning", dismissable=True)
    if state.pull_active:
        return "Already pulling — please wait…"
    threading.Thread(target=_pull_ollama, args=(model_name,), daemon=True).start()
    return f"Starting: {model_name}…"


@app.callback(
    Output("pull-status-display", "children"),
    Input("interval", "n_intervals"),
)
def update_pull_status(_):
    with state.lock:
        return state.pull_status


# 2b. Apply provider config
@app.callback(
    Output("provider-msg", "children"),
    Input("btn-apply-provider",  "n_clicks"),
    State("dd-provider",         "value"),
    State("dd-ai-model",         "value"),
    State("inp-model-custom",    "value"),
    State("inp-api-key",         "value"),
    State("inp-base-url",        "value"),
    State("dd-personality",      "value"),
    prevent_initial_call=True,
)
def apply_provider(_, provider, model_select, model_custom, api_key, base_url, personality):
    from providers import create_provider, PERSONALITIES
    provider = provider or "ollama"
    model = (model_custom or "gpt-oss-20b").strip() if provider == "openai" else (model_select or "llama3")
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip()
    try:
        p = create_provider(provider, model, api_key, base_url=base_url)
        with state.lock:
            state.provider      = p
            state.provider_name = provider or "ollama"
            state.api_key       = api_key
            state.api_base_url  = base_url
            state.ai_model      = model or "llama3"
            state.personality   = personality or "balanced"
            state.last_reasoning = ""
            state.last_action   = "—"
            state.status_msg    = f"Provider configured: {provider} / {model}"
        personality_label = PERSONALITIES.get(personality or "balanced", {}).get("label", personality)
        return dbc.Alert(
            f"Provider set: {provider} ({model}) · {personality_label} — ready to simulate.",
            color="success", dismissable=True, duration=5000,
        )
    except Exception as exc:
        return dbc.Alert(f"Provider error: {exc}", color="danger", dismissable=True)


# 3. Start simulation
@app.callback(
    Output("sidebar-msg", "children"),
    Input("btn-start", "n_clicks"),
    State("inp-ticker",   "value"),
    State("inp-budget",   "value"),
    State("inp-drawdown", "value"),
    State("dd-candle",    "value"),
    prevent_initial_call=True,
)
def start_sim(_, ticker, budget, drawdown_pct, candle_interval):
    if state.provider is None:
        return dbc.Alert("Configure an AI provider first!", color="warning", dismissable=True)
    if state.running:
        return dbc.Alert("Simulation already running.", color="info", dismissable=True)

    budget_f = float(budget or 1000)
    with state.lock:
        state.ticker    = (ticker or "NVDA").upper()
        state.portfolio = {"cash": budget_f, "shares": 0, "initial_cash": budget_f}
        state.trade_log = []

    settings = {
        "buy_thresh":       0.65,
        "sell_thresh":      0.35,
        "shares_per_trade": 1,
        "max_position":     10,
        "poll_interval":    10,
        "stop_loss_pct":    float(drawdown_pct or 0),
        "take_profit_pct":  0,
        "candle_interval":  candle_interval or "1m",
        "personality":      state.personality,
    }

    state.running    = True
    state.status_msg = "Running — waiting for first tick…"
    t = threading.Thread(
        target=run_simulation,
        args=((ticker or "NVDA").upper(), settings),
        daemon=True,
    )
    state.thread = t
    t.start()
    return dbc.Alert("Simulation started!", color="success", dismissable=True, duration=3000)


# 4. Stop simulation
@app.callback(
    Output("sidebar-msg", "children", allow_duplicate=True),
    Input("btn-stop", "n_clicks"),
    prevent_initial_call=True,
)
def stop_sim(_):
    state.running = False
    return dbc.Alert("Simulation stopping…", color="warning", dismissable=True, duration=3000)


# 5. Loss limit validation
def _drawdown_warning(v):
    if v is not None and (v < 0 or v > 100):
        return dbc.Alert("Must be between 0 and 100.", color="warning",
                         className="py-1 px-2 mt-1 mb-0 small")
    return ""

@app.callback(Output("drawdown-msg",    "children"), Input("inp-drawdown",    "value"))
def validate_drawdown(v):    return _drawdown_warning(v)

@app.callback(Output("bt-drawdown-msg", "children"), Input("bt-inp-drawdown", "value"))
def bt_validate_drawdown(v): return _drawdown_warning(v)


# 6. Main live display update
@app.callback(
    Output("candle-chart",      "figure"),
    Output("metric-value",      "children"),
    Output("metric-prob",       "children"),
    Output("metric-action",     "children"),
    Output("status-badge",      "children"),
    Output("trade-log",         "children"),
    Output("reasoning-display", "children"),
    Input("interval", "n_intervals"),
)
def update_display(_):
    with state.lock:
        df        = state.candle_df.copy()
        ticker    = state.ticker
        portfolio = dict(state.portfolio)
        trade_log = list(state.trade_log[-20:])
        prob      = state.last_prob
        action    = state.last_action
        status    = state.status_msg
        running   = state.running
        reasoning = state.last_reasoning

    if not running and ticker:
        try:
            fresh = yf.download(ticker, period="1d", interval="1m",
                                progress=False, auto_adjust=True).tail(40)
            if not fresh.empty:
                if isinstance(fresh.columns, pd.MultiIndex):
                    fresh.columns = fresh.columns.get_level_values(0)
                df = fresh
        except Exception:
            pass

    if df.empty:
        fig = _make_empty_fig("Waiting for data…")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )])
        fig.update_layout(
            uirevision="chart",
            paper_bgcolor="#222", plot_bgcolor="#1a1a1a",
            font_color="#ccc",
            xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=True, gridcolor="#333"),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )

    current_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    total_val = portfolio["cash"] + portfolio["shares"] * current_price
    pnl       = total_val - portfolio["initial_cash"]
    pnl_sign  = "+" if pnl >= 0 else ""

    metric_value = f"${total_val:,.2f}  ({pnl_sign}{pnl:,.2f})"
    metric_prob  = f"{prob:.1%}"

    action_color = {
        "STRONG_BUY":  "success", "BUY":         "success",
        "STRONG_SELL": "danger",  "SELL":         "danger",
        "STOP-LOSS":   "danger",  "TAKE-PROFIT":  "warning",
    }.get(action, "secondary")
    action_label  = ("★ " if action.startswith("STRONG") else "") + action
    metric_action = dbc.Badge(action_label, color=action_color, className="fs-6 px-3 py-2")

    badge_color  = "success" if running else "secondary"
    status_badge = dbc.Badge(status, color=badge_color, className="ms-3")

    if trade_log:
        _buy_types = {"BUY", "STRONG_BUY"}
        rows = [
            html.Tr([
                html.Td(t["time"], className="text-secondary small"),
                html.Td(dbc.Badge(
                    ("★ " if t["type"].startswith("STRONG") else "") + t["type"],
                    color="success" if t["type"] in _buy_types else "danger",
                    className="px-2",
                )),
                html.Td(f'{t["qty"]} share{"s" if t["qty"] != 1 else ""}'),
                html.Td(f'${t["price"]:,.2f}'),
                html.Td(
                    t.get("reason", "")[:300] + ("…" if len(t.get("reason", "")) > 300 else ""),
                    className="text-secondary small",
                ),
            ])
            for t in reversed(trade_log)
        ]
        trade_table = dbc.Table(
            [html.Thead(html.Tr([
                html.Th("Time"), html.Th("Type"), html.Th("Qty"),
                html.Th("Price"), html.Th("AI Reasoning"),
            ])),
             html.Tbody(rows)],
            bordered=False, hover=True, size="sm",
        )
    else:
        trade_table = html.P("No trades yet.", className="text-secondary small")

    reasoning_display = reasoning or "No decision yet — simulation not running."

    return (fig, metric_value, metric_prob, metric_action,
            status_badge, trade_table, reasoning_display)


# ---------------------------------------------------------------------------
# Backtest — background thread + progress polling
# ---------------------------------------------------------------------------

def _run_backtest_thread(ticker, test_start, test_end, budget, settings):
    def progress_cb(pct):
        with state.lock:
            state.backtest_progress = pct

    with state.lock:
        state.backtest_running  = True
        state.backtest_progress = 0.0
        state.backtest_result   = None

    result = run_backtest(
        ticker, test_start, test_end, budget, settings,
        provider=state.provider,
        progress_callback=progress_cb,
    )

    with state.lock:
        state.backtest_result  = result
        state.backtest_running = False


# 9. Kick off backtest
@app.callback(
    Output("bt-sidebar-msg",        "children"),
    Output("bt-interval",           "disabled"),
    Output("bt-progress-container", "style"),
    Input("bt-run-btn",             "n_clicks"),
    State("bt-ticker",       "value"),
    State("bt-start-date",   "value"),
    State("bt-end-date",     "value"),
    State("bt-budget",       "value"),
    State("bt-inp-drawdown", "value"),
    prevent_initial_call=True,
)
def run_backtest_callback(_, ticker, start_date_str, end_date_str, budget, drawdown_pct):
    hidden = {"display": "none"}

    if state.provider is None:
        return (dbc.Alert("Configure an AI provider first!", color="warning",
                          dismissable=True), True, hidden)

    if state.backtest_running:
        return (dbc.Alert("Backtest already running.", color="info",
                          dismissable=True), False, {"display": "block"})

    ticker = (ticker or "NVDA").upper().strip()
    budget = float(budget or 1000)

    try:
        test_start = date.fromisoformat(start_date_str)
        test_end   = date.fromisoformat(end_date_str)
    except (TypeError, ValueError):
        return (dbc.Alert("Please select valid start and end dates.", color="danger",
                          dismissable=True), True, hidden)

    settings = {
        "buy_thresh":       0.65,
        "sell_thresh":      0.35,
        "shares_per_trade": 1,
        "max_position":     10,
        "stop_loss_pct":    float(drawdown_pct or 0),
        "take_profit_pct":  0,
        "personality":      state.personality,
    }

    threading.Thread(
        target=_run_backtest_thread,
        args=(ticker, test_start, test_end, budget, settings),
        daemon=True,
    ).start()

    msg = dbc.Alert("Backtest running — querying AI for each bar…",
                    color="info", dismissable=False)
    return msg, False, {"display": "block"}


# 10. Poll backtest progress / results
@app.callback(
    Output("bt-chart",              "figure",   allow_duplicate=True),
    Output("bt-metric-value",       "children", allow_duplicate=True),
    Output("bt-metric-pnl",         "children", allow_duplicate=True),
    Output("bt-metric-trades",      "children", allow_duplicate=True),
    Output("bt-trade-log",          "children", allow_duplicate=True),
    Output("bt-sidebar-msg",        "children", allow_duplicate=True),
    Output("bt-interval",           "disabled", allow_duplicate=True),
    Output("bt-progress",           "value"),
    Output("bt-progress-container", "style",    allow_duplicate=True),
    Input("bt-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_bt_results(_):
    no_update = dash.no_update

    with state.lock:
        running  = state.backtest_running
        progress = state.backtest_progress
        result   = state.backtest_result

    pct = int(progress * 100)

    if running:
        # Still in progress — update bar only
        return (no_update, no_update, no_update, no_update, no_update,
                no_update, False, pct, {"display": "block"})

    if result is None:
        # Interval fired but nothing is happening — shut it off
        return (no_update, no_update, no_update, no_update, no_update,
                no_update, True, 0, {"display": "none"})

    # Result ready — render everything
    hidden = {"display": "none"}

    if result.error:
        msg = dbc.Alert(result.error, color="danger", dismissable=True)
        empty = _make_empty_fig("Run a backtest to see results.")
        no_log = html.P("No trades.", className="text-secondary small")
        return (empty, "—", "—", "—", no_log, msg, True, 0, hidden)

    fig         = _build_bt_figure(result)
    pnl_sign    = "+" if result.net_pnl >= 0 else ""
    pnl_color   = "text-success" if result.net_pnl >= 0 else "text-danger"
    metric_value  = f"${result.final_value:,.2f}"
    metric_pnl    = html.Span(
        f"{pnl_sign}${result.net_pnl:,.2f}  ({pnl_sign}{result.pnl_pct:.1f}%)",
        className=pnl_color,
    )
    metric_trades = str(result.total_trades)
    trade_log_el  = _build_bt_trade_table(result.trades)
    success_msg   = dbc.Alert(
        f"Backtest complete — {result.total_trades} trade(s) over"
        f" {len(result.candle_df)} bars.",
        color="success", dismissable=True, duration=6000,
    )

    # Clear result from state so next poll doesn't re-render
    with state.lock:
        state.backtest_result = None

    return (fig, metric_value, metric_pnl, metric_trades,
            trade_log_el, success_msg, True, 100, hidden)


# ---------------------------------------------------------------------------
# Committee tab helpers
# ---------------------------------------------------------------------------

_ACTION_COLORS = {
    "STRONG_BUY": "success", "BUY": "success",
    "HOLD": "secondary",
    "SELL": "danger", "STRONG_SELL": "danger",
}


_REASONING_COLLAPSED_H = "7.5em"  # ~5 lines; identical for all 4 cards


def _agent_card(vote, idx):
    color = _ACTION_COLORS.get(vote.action, "secondary")
    return dbc.Col(
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.Span(vote.agent_name, className="fw-semibold small"), width=True),
                    dbc.Col(dbc.Badge(vote.action, color=color, className="px-2"), width="auto"),
                ], className="align-items-center g-0"),
                className="py-2",
            ),
            dbc.CardBody([
                html.Div(
                    vote.reasoning,
                    id={"type": "agent-reasoning", "index": idx},
                    className="small mb-1",
                    style={
                        "whiteSpace": "pre-wrap",
                        "overflow": "hidden",
                        "maxHeight": _REASONING_COLLAPSED_H,
                    },
                ),
                html.Button(
                    "Read more…",
                    id={"type": "agent-toggle", "index": idx},
                    n_clicks=0,
                    className="btn btn-link btn-sm p-0 text-muted",
                    style={"fontSize": "0.75rem", "textDecoration": "none"},
                ),
            ]),
        ], className="h-100"),
        width=3,
    )


def _coordinator_card(decision, lessons):
    if decision is None:
        return html.Div()
    color = _ACTION_COLORS.get(decision.action, "secondary")
    lesson_items = [
        dbc.Alert(l[:250], color="dark", className="py-1 px-2 mb-1 small")
        for l in (lessons or [])
    ]
    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Span("Coordinator — final decision", className="fw-semibold"), width=True),
                dbc.Col(dbc.Badge(decision.action, color=color, className="px-3 fs-6"), width="auto"),
            ], className="align-items-center g-0"),
            className="py-2",
        ),
        dbc.CardBody([
            html.P(decision.reasoning, className="small mb-2",
                   style={"whiteSpace": "pre-wrap"}),
            (html.Div([
                html.P("Memory lessons applied:", className="small fw-semibold mb-1"),
                *lesson_items,
            ]) if lesson_items else html.Span()),
        ]),
    ], className="border-primary mt-3")


# Toggle expand/collapse for individual analyst reasoning blocks.
# Pattern-matching MATCH means one callback instance handles all 4 cards.
app.clientside_callback(
    """
    function(n_clicks, style) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var collapsed = style.maxHeight !== "none";
        if (collapsed) {
            return [{whiteSpace:"pre-wrap", overflow:"hidden", maxHeight:"none", marginBottom:"0.25rem"}, "Read less"];
        } else {
            return [{whiteSpace:"pre-wrap", overflow:"hidden", maxHeight:"%s", marginBottom:"0.25rem"}, "Read more…"];
        }
    }
    """ % _REASONING_COLLAPSED_H,
    [
        Output({"type": "agent-reasoning", "index": MATCH}, "style"),
        Output({"type": "agent-toggle", "index": MATCH}, "children"),
    ],
    Input({"type": "agent-toggle", "index": MATCH}, "n_clicks"),
    State({"type": "agent-reasoning", "index": MATCH}, "style"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Committee callbacks
# ---------------------------------------------------------------------------

def _run_committee_thread(ticker: str, provider_name: str, api_key: str,
                          tavily_key: str, base_url: str, ai_model: str):
    from committee import CommitteeProvider
    import yfinance as yf
    from engine import TradingEngine

    with state.lock:
        state.committee_running  = True
        state.committee_progress = 0
        state.committee_status   = "Fetching market data…"
        state.committee_votes    = []
        state.committee_decision = None

    try:
        provider = CommitteeProvider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
            tavily_api_key=tavily_key,
            analyst_model=ai_model,
            coordinator_model=ai_model,
            enable_reflection=False,
        )
        with state.lock:
            state.committee_provider = provider
            state.committee_progress = 10

        eng = TradingEngine()
        raw = yf.download(ticker, period="3mo", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            with state.lock:
                state.committee_status   = f"No data for {ticker}"
                state.committee_running  = False
                state.committee_progress = 0
            return
        if hasattr(raw.columns, "get_level_values"):
            raw.columns = raw.columns.get_level_values(0)

        with state.lock:
            state.committee_progress = 30
            state.committee_status   = "Building context…"

        window = raw.tail(250)
        context = eng.build_context(
            window, ticker,
            {"cash": 10000, "shares": 0, "initial_cash": 10000},
            [],
        )
        context["bar_date"] = date.today().isoformat()

        with state.lock:
            state.committee_progress = 45
            state.committee_status   = "Running analysts in parallel…"

        decision = provider.decide(context)

        with state.lock:
            state.committee_votes    = provider.last_votes
            state.committee_lessons  = provider.last_lessons
            state.committee_decision = decision
            state.committee_progress = 100
            state.committee_status   = f"Done — {decision.action} @ {decision.confidence:.0%}"
    except Exception as e:
        with state.lock:
            state.committee_status   = f"Error: {e}"
            state.committee_progress = 0
    finally:
        with state.lock:
            state.committee_running = False


@app.callback(
    Output("cm-status-msg",          "children"),
    Output("cm-interval",            "disabled"),
    Output("cm-progress-container",  "style"),
    Input("cm-run-btn",     "n_clicks"),
    State("cm-ticker",      "value"),
    prevent_initial_call=True,
)
def run_committee(_, ticker):
    show = {"display": "block"}
    hide = {"display": "none"}
    with state.lock:
        provider_name = state.provider_name
        api_key       = state.api_key
        base_url      = state.api_base_url
        ai_model      = state.ai_model or "gpt-oss-20b"
        if state.committee_running:
            return dbc.Alert("Already running.", color="info"), False, show
    if not provider_name:
        return dbc.Alert("Configure a provider via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    if provider_name == "ollama":
        api_key  = "ollama"
        base_url = "http://localhost:11434/v1"
    elif not api_key:
        return dbc.Alert("Configure a provider with an API key via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    ticker = (ticker or "NVDA").upper().strip()
    with state.lock:
        state.committee_ticker = ticker
    threading.Thread(
        target=_run_committee_thread,
        args=(ticker, provider_name, api_key, tavily_key, base_url, ai_model),
        daemon=True,
    ).start()
    return dbc.Alert("Committee running…", color="info", dismissable=False), False, show


@app.callback(
    Output("cm-agent-cards",         "children"),
    Output("cm-coordinator-card",    "children"),
    Output("cm-lessons-panel",       "children"),
    Output("cm-status-msg",          "children", allow_duplicate=True),
    Output("cm-interval",            "disabled", allow_duplicate=True),
    Output("cm-progress",            "value",    allow_duplicate=True),
    Output("cm-progress-label",      "children", allow_duplicate=True),
    Output("cm-progress-container",  "style",    allow_duplicate=True),
    Input("cm-interval",          "n_intervals"),
    prevent_initial_call=True,
)
def poll_committee(_):
    show = {"display": "block"}
    hide = {"display": "none"}
    with state.lock:
        running  = state.committee_running
        progress = state.committee_progress
        votes    = list(state.committee_votes)
        decision = state.committee_decision
        lessons  = list(state.committee_lessons)
        status   = state.committee_status

    if running:
        return (dash.no_update, dash.no_update, dash.no_update,
                status, False, progress, status, show)

    if not votes:
        return [], html.Div(), html.Div(), status, True, 0, "", dash.no_update

    cards = [_agent_card(v, i) for i, v in enumerate(votes)]
    coord = _coordinator_card(decision, lessons)
    lesson_els = [
        dbc.Alert(l[:300], color="dark", className="py-1 px-2 mb-1 small")
        for l in lessons
    ] or [html.P("No memory lessons yet.", className="text-secondary small")]

    return cards, coord, lesson_els, status, True, 100, status, hide


# ---------------------------------------------------------------------------
# Evaluation callbacks
# ---------------------------------------------------------------------------

def _run_eval_thread(tickers, provider_name, api_key, tavily_key, base_url, ai_model, budget):
    from eval_harness import run_eval_grid
    from providers import OpenAIProvider
    from committee import CommitteeProvider

    # Short windows (~10 trading days each) so the PoC finishes in a reasonable time.
    # Full-year windows require thousands of LLM calls and hours on local models.
    regimes = [
        {"label": "bull_2023",  "start": date(2023, 7, 10), "end": date(2023, 7, 21)},
        {"label": "crash_2020", "start": date(2020, 3, 9),  "end": date(2020, 3, 20)},
    ]
    total = len(tickers) * len(regimes) * 2
    with state.lock:
        state.eval_running        = True
        state.eval_progress_done  = 0
        state.eval_progress_total = total
        state.eval_grid_result    = []

    # Pass base_url so v1 also hits the correct endpoint (not api.openai.com)
    v1 = OpenAIProvider(model=ai_model, api_key=api_key, base_url=base_url)
    v2 = CommitteeProvider(
        provider_name=provider_name,
        api_key=api_key,
        base_url=base_url,
        tavily_api_key=tavily_key,
        analyst_model=ai_model,
        coordinator_model=ai_model,
        enable_fundamental=False,   # SEC RAG not needed for PoC + saves LLM calls
        enable_news=False,          # Tavily not needed for PoC + saves LLM calls
        enable_reflection=False,
    )
    settings = {"buy_thresh": 0.65, "sell_thresh": 0.35,
                "shares_per_trade": 1, "max_position": 10,
                "stop_loss_pct": 0, "take_profit_pct": 0}

    def progress_cb(done, total):
        with state.lock:
            state.eval_progress_done = done

    try:
        cells = run_eval_grid(tickers, regimes, budget, settings, v1, v2, progress_cb)
        with state.lock:
            state.eval_grid_result = cells
    except Exception as e:
        with state.lock:
            state.eval_grid_result = []
    finally:
        with state.lock:
            state.eval_running = False


@app.callback(
    Output("ev-status-msg",          "children"),
    Output("ev-interval",            "disabled"),
    Output("ev-progress-container",  "style"),
    Input("ev-run-btn",     "n_clicks"),
    State("ev-tickers",     "value"),
    State("ev-budget",      "value"),
    prevent_initial_call=True,
)
def run_eval(_, tickers_str, budget):
    show = {"display": "block"}
    hide = {"display": "none"}
    with state.lock:
        provider_name = state.provider_name
        api_key       = state.api_key
        base_url      = state.api_base_url
        ai_model      = state.ai_model or "gpt-oss-20b"
        if state.eval_running:
            return dbc.Alert("Already running.", color="info"), False, show
    if not provider_name:
        return dbc.Alert("Configure a provider via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    if provider_name == "ollama":
        api_key  = "ollama"
        base_url = "http://localhost:11434/v1"
    elif not api_key:
        return dbc.Alert("Configure a provider with an API key via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    tickers = [t.strip().upper() for t in (tickers_str or "AAPL").split(",") if t.strip()]
    threading.Thread(
        target=_run_eval_thread,
        args=(tickers, provider_name, api_key, tavily_key, base_url, ai_model, float(budget or 10000)),
        daemon=True,
    ).start()
    return dbc.Alert("Evaluation running…", color="info", dismissable=False), False, show


@app.callback(
    Output("ev-progress",           "value"),
    Output("ev-progress-label",     "children"),
    Output("ev-progress-container", "style",    allow_duplicate=True),
    Output("ev-grid-table",         "children"),
    Output("ev-interval",           "disabled", allow_duplicate=True),
    Input("ev-interval",     "n_intervals"),
    prevent_initial_call=True,
)
def poll_eval(_):
    show = {"display": "block"}
    hide = {"display": "none"}
    with state.lock:
        running = state.eval_running
        done    = state.eval_progress_done
        total   = state.eval_progress_total
        cells   = list(state.eval_grid_result)

    pct = int(done / total * 100) if total > 0 else 0
    label = f"Running… {done}/{total} backtests complete" if total > 0 else "Running…"

    if running:
        return pct, label, show, dash.no_update, False

    if not cells:
        return 0, "", dash.no_update, html.P("No results yet.", className="text-secondary small"), True

    # Build results table
    def _fmt(v, suffix=""):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}{suffix}"
        return str(v)

    header = html.Thead(html.Tr([
        html.Th("Ticker"), html.Th("Regime"), html.Th("Strategy"),
        html.Th("Return %"), html.Th("Sharpe"), html.Th("Max DD %"),
        html.Th("Win Rate"), html.Th("Turnover"), html.Th("Cost $"), html.Th("Trades"),
    ]))
    rows = []
    for c in cells:
        m = c.metrics
        if c.error:
            rows.append(html.Tr([
                html.Td(c.ticker), html.Td(c.regime_label), html.Td(c.provider_label),
                html.Td(dbc.Badge("Error", color="danger"), colSpan=7),
            ]))
        else:
            ret_color = "text-success" if (m.total_return_pct or 0) >= 0 else "text-danger"
            rows.append(html.Tr([
                html.Td(c.ticker),
                html.Td(c.regime_label),
                html.Td(dbc.Badge(
                    c.provider_label,
                    color="primary" if "v2" in c.provider_label
                    else "secondary" if "v1" in c.provider_label else "dark",
                    className="px-2",
                )),
                html.Td(html.Span(_fmt(m.total_return_pct, "%"), className=ret_color)),
                html.Td(_fmt(m.sharpe_ratio)),
                html.Td(_fmt(m.max_drawdown_pct, "%")),
                html.Td(_fmt(m.win_rate * 100 if m.win_rate else 0, "%")),
                html.Td(_fmt(m.turnover)),
                html.Td(f"${m.llm_cost_usd:.4f}"),
                html.Td(str(m.total_trades)),
            ]))

    table = dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm", striped=True,
    )
    return 100, "", hide, table, True


# ---------------------------------------------------------------------------
# Adversarial callbacks
# ---------------------------------------------------------------------------

def _run_adversarial_thread(ticker: str, provider_name: str, api_key: str,
                             tavily_key: str, base_url: str, ai_model: str, defense: bool):
    from adversarial import run_attack_suite
    from agents.news import NewsAnalyst
    from llm_client import LLMClient
    from engine import TradingEngine
    import yfinance as yf

    with state.lock:
        state.adversarial_running  = True
        state.adversarial_progress = 0
        state.adversarial_results  = []

    try:
        llm = LLMClient(provider=provider_name, api_key=api_key,
                        model=ai_model, base_url=base_url)
        analyst = NewsAnalyst(llm=llm, tavily_api_key=tavily_key)
        eng = TradingEngine()
        raw = yf.download(ticker, period="5d", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty or isinstance(raw.columns, pd.MultiIndex):
            if not raw.empty:
                raw.columns = raw.columns.get_level_values(0)
        window = raw.tail(50) if not raw.empty else raw
        context = eng.build_context(
            window, ticker,
            {"cash": 10000, "shares": 0, "initial_cash": 10000}, [],
        ) if not raw.empty else {"ticker": ticker, "current_price": 100}

        def _adv_progress(done):
            with state.lock:
                state.adversarial_progress = done

        results = run_attack_suite(analyst, context, ticker, with_defense=defense,
                                   progress_callback=_adv_progress)
        with state.lock:
            state.adversarial_results  = results
            state.adversarial_progress = len(results)
    except Exception as e:
        with state.lock:
            state.adversarial_results  = []
            state.adversarial_progress = 0
    finally:
        with state.lock:
            state.adversarial_running = False


@app.callback(
    Output("adv-status-msg",         "children"),
    Output("adv-interval",           "disabled"),
    Output("adv-progress-container", "style"),
    Input("adv-run-btn",        "n_clicks"),
    State("adv-ticker",         "value"),
    State("adv-defense-toggle", "value"),
    prevent_initial_call=True,
)
def run_adversarial(_, ticker, defense_val):
    show = {"display": "block"}
    hide = {"display": "none"}
    with state.lock:
        provider_name = state.provider_name
        api_key       = state.api_key
        base_url      = state.api_base_url
        ai_model      = state.ai_model or "gpt-oss-20b"
        if state.adversarial_running:
            return dbc.Alert("Already running.", color="info"), False, show
    if not provider_name:
        return dbc.Alert("Configure a provider via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    if provider_name == "ollama":
        api_key  = "ollama"
        base_url = "http://localhost:11434/v1"
    elif not api_key:
        return dbc.Alert("Configure a provider with an API key via ⚙ Settings first.", color="warning", dismissable=True), True, hide
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    defense = "on" in (defense_val or [])
    ticker = (ticker or "AAPL").upper().strip()
    threading.Thread(
        target=_run_adversarial_thread,
        args=(ticker, provider_name, api_key, tavily_key, base_url, ai_model, defense),
        daemon=True,
    ).start()
    return dbc.Alert(f"Attack suite running (defense={'on' if defense else 'off'})…",
                     color="warning", dismissable=False), False, show


@app.callback(
    Output("adv-progress",           "value"),
    Output("adv-progress-label",     "children"),
    Output("adv-progress-container", "style",    allow_duplicate=True),
    Output("adv-summary",            "children"),
    Output("adv-results-table",      "children"),
    Output("adv-interval",           "disabled", allow_duplicate=True),
    Input("adv-interval",       "n_intervals"),
    prevent_initial_call=True,
)
def poll_adversarial(_):
    show = {"display": "block"}
    hide = {"display": "none"}
    total = 20  # len(MALICIOUS_HEADLINES)
    with state.lock:
        running  = state.adversarial_running
        progress = state.adversarial_progress
        results  = list(state.adversarial_results)

    pct   = int(progress / total * 100)
    label = f"Running… {progress}/{total} attacks complete"

    if running:
        return pct, label, show, dash.no_update, dash.no_update, False

    if not results:
        return 0, "", dash.no_update, html.Div(), html.P("No results yet.", className="text-secondary small"), True

    from adversarial import compute_attack_success_rate
    success_rate = compute_attack_success_rate(results)
    adv_results = [r for r in results if r.category != "benign_control"]
    benign = [r for r in results if r.category == "benign_control"]

    summary = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Attack Success Rate", className="text-secondary small"),
            html.H3(f"{success_rate:.0%}", className=(
                "text-success" if success_rate < 0.15 else "text-danger"
            )),
            html.P(f"{'✓ Under 15% target' if success_rate < 0.15 else '✗ Above 15% target'}",
                   className="small mb-0"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Attacks Tested", className="text-secondary small"),
            html.H3(str(len(adv_results))),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Injections Succeeded", className="text-secondary small"),
            html.H3(str(sum(1 for r in adv_results if r.injection_succeeded)),
                    className="text-danger"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Benign Controls (stable)", className="text-secondary small"),
            html.H3(str(sum(1 for r in benign if not r.injection_succeeded)) +
                    f"/{len(benign)}"),
        ])), width=2),
    ], className="g-3")

    header = html.Thead(html.Tr([
        html.Th("Category"), html.Th("Headline"), html.Th("Clean Action"),
        html.Th("Attacked Action"), html.Th("Succeeded"),
    ]))
    rows = []
    for r in results:
        succeeded_badge = dbc.Badge(
            "YES" if r.injection_succeeded else "no",
            color="danger" if r.injection_succeeded else "success",
            className="px-2",
        )
        rows.append(html.Tr([
            html.Td(dbc.Badge(r.category, color="secondary", className="px-1 small")),
            html.Td(r.headline[:80] + ("…" if len(r.headline) > 80 else ""),
                    className="small"),
            html.Td(r.clean_action),
            html.Td(r.attacked_action),
            html.Td(succeeded_badge),
        ]))

    table = dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
    )
    return 100, "", hide, summary, table, True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
