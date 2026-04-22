import numpy as np
import pandas as pd


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class TradingEngine:
    """Stateless feature-engineering wrapper. Computes technical indicators
    from OHLCV data and assembles context dicts for AI providers."""

    def _compute_features(self, df):
        """Add all indicator columns to df in-place and return it."""
        c = df['Close']
        o = df['Open']
        h = df['High']
        l = df['Low']
        v = df['Volume']

        df['Return'] = c.pct_change()
        df['SMA_20'] = c.rolling(20).mean()
        df['SMA_50'] = c.rolling(50).mean()
        df['SMA_200'] = c.rolling(200).mean()
        df['Price_vs_SMA20'] = c / df['SMA_20'] - 1
        df['Price_vs_SMA50'] = c / df['SMA_50'] - 1
        df['Price_vs_SMA200'] = c / df['SMA_200'] - 1

        vol_avg = v.rolling(20).mean()
        df['Vol_Rel'] = v / vol_avg
        df['Vol_Spike'] = (v > 2 * vol_avg).astype(int)

        df['RSI_14'] = _rsi(c)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

        std20 = c.rolling(20).std()
        bb_upper = df['SMA_20'] + 2 * std20
        bb_lower = df['SMA_20'] - 2 * std20
        df['BB_Pct'] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        hl = h - l
        hc = (h - c.shift(1)).abs()
        lc = (l - c.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR_Norm'] = tr.rolling(14).mean() / c

        candle_range = (h - l).replace(0, np.nan)
        body = (c - o).abs()
        df['Body_Ratio'] = body / candle_range
        top_of_body = pd.concat([c, o], axis=1).max(axis=1)
        bot_of_body = pd.concat([c, o], axis=1).min(axis=1)
        df['Upper_Wick'] = (h - top_of_body) / candle_range
        df['Lower_Wick'] = (bot_of_body - l) / candle_range

        df['Is_Hammer'] = (
            (df['Lower_Wick'] >= 0.55) &
            (df['Body_Ratio'] <= 0.30) &
            (df['Upper_Wick'] <= 0.15)
        ).astype(int)

        df['Is_Doji'] = (df['Body_Ratio'] <= 0.05).astype(int)

        prev_bear = c.shift(1) < o.shift(1)
        curr_bull = c > o
        engulfs_up = (o <= c.shift(1)) & (c >= o.shift(1))
        df['Is_Bull_Engulf'] = (prev_bear & curr_bull & engulfs_up).astype(int)

        prev_bull = c.shift(1) > o.shift(1)
        curr_bear = c < o
        engulfs_dn = (o >= c.shift(1)) & (c <= o.shift(1))
        df['Is_Bear_Engulf'] = (prev_bull & curr_bear & engulfs_dn).astype(int)

        return df

    def prepare_live_features(self, df):
        """
        Compute indicators from a candle df (needs ≥200 rows for SMA_200).
        Returns (feature_list, signals_dict).
        """
        df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        df = self._compute_features(df)
        row = df.iloc[-1]

        def _safe(val, fallback=0.0):
            try:
                v = float(val)
                return fallback if (np.isnan(v) or np.isinf(v)) else v
            except Exception:
                return fallback

        signals = {
            "rsi":            _safe(row['RSI_14'], 50.0),
            "macd_hist":      _safe(row['MACD_Hist']),
            "bb_pct":         _safe(row['BB_Pct'], 0.5),
            "price_vs_sma20": _safe(row['Price_vs_SMA20']),
            "price_vs_sma50": _safe(row['Price_vs_SMA50']),
            "is_hammer":      bool(row['Is_Hammer']),
            "is_bull_engulf": bool(row['Is_Bull_Engulf']),
            "is_bear_engulf": bool(row['Is_Bear_Engulf']),
            "is_doji":        bool(row['Is_Doji']),
        }

        # features list kept for API compatibility but not used by AI providers
        features = [_safe(row.get(col, 0.0)) for col in [
            'Return', 'SMA_20', 'SMA_50', 'SMA_200',
            'Price_vs_SMA20', 'Price_vs_SMA50', 'Price_vs_SMA200',
            'Vol_Rel', 'Vol_Spike', 'RSI_14',
            'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'BB_Pct', 'ATR_Norm', 'Body_Ratio', 'Upper_Wick', 'Lower_Wick',
            'Is_Hammer', 'Is_Bull_Engulf', 'Is_Bear_Engulf', 'Is_Doji',
        ]]

        return features, signals

    def build_context(self, df, ticker, portfolio, trade_log,
                      buy_thresh=0.65, sell_thresh=0.35, personality="balanced"):
        """
        Build the full context dict passed to a provider's decide() method.
        Combines OHLCV snapshot, computed signals, historical summary,
        portfolio state, and recent trade history.
        """
        df_norm = df.copy()
        if isinstance(df_norm.columns, pd.MultiIndex):
            df_norm.columns = df_norm.columns.get_level_values(0)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_norm.columns and isinstance(df_norm[col], pd.DataFrame):
                df_norm[col] = df_norm[col].iloc[:, 0]

        _, signals = self.prepare_live_features(df)
        last  = df_norm.iloc[-1]
        close = float(last['Close'])

        # ------------------------------------------------------------------
        # Historical summary — synthesises the 250-bar window into statistics
        # that give the AI the context a "training window" would provide.
        # ------------------------------------------------------------------
        c = df_norm['Close'].astype(float)
        n = len(c)

        def _pct_change(periods):
            if n > periods:
                try:
                    v = (c.iloc[-1] / c.iloc[-1 - periods] - 1) * 100
                    return round(float(v), 2) if np.isfinite(v) else None
                except Exception:
                    return None
            return None

        period_n  = min(252, n)
        hi        = float(c.tail(period_n).max())
        lo        = float(c.tail(period_n).min())
        sma20     = float(c.tail(20).mean())  if n >= 20  else None
        sma50     = float(c.tail(50).mean())  if n >= 50  else None
        sma200    = float(c.tail(200).mean()) if n >= 200 else None

        trend_parts = []
        for label, val in [("SMA20", sma20), ("SMA50", sma50), ("SMA200", sma200)]:
            if val is not None:
                trend_parts.append(f"{'ABOVE' if close > val else 'BELOW'} {label}")

        historical = {
            "bars_available": n,
            "return_5bar":    _pct_change(5),
            "return_20bar":   _pct_change(20),
            "return_60bar":   _pct_change(60),
            "period_high":    round(hi, 4),
            "period_low":     round(lo, 4),
            "pct_from_high":  round((close / hi - 1) * 100, 2) if hi else None,
            "pct_from_low":   round((close / lo - 1) * 100, 2) if lo else None,
            "trend_alignment": ", ".join(trend_parts) if trend_parts else "unknown",
        }

        # ------------------------------------------------------------------
        total_value = portfolio['cash'] + portfolio['shares'] * close

        recent = []
        for t in trade_log[-5:]:
            if isinstance(t, dict):
                recent.append({
                    "date":  t.get('time', t.get('date', '?')),
                    "type":  t.get('type', '?'),
                    "qty":   t.get('qty', 0),
                    "price": t.get('price', 0.0),
                })
            else:
                recent.append({
                    "date":  getattr(t, 'date', '?'),
                    "type":  getattr(t, 'trade_type', '?'),
                    "qty":   getattr(t, 'qty', 0),
                    "price": getattr(t, 'price', 0.0),
                })

        return {
            "ticker":        ticker,
            "current_price": round(close, 4),
            "open":          round(float(last['Open']), 4),
            "high":          round(float(last['High']), 4),
            "low":           round(float(last['Low']), 4),
            "volume":        int(float(last['Volume'])),
            "signals":       signals,
            "historical":    historical,
            "personality":   personality,
            "portfolio": {
                "cash":        round(portfolio['cash'], 2),
                "shares":      portfolio['shares'],
                "total_value": round(total_value, 2),
            },
            "recent_trades": recent,
            "buy_thresh":    buy_thresh,
            "sell_thresh":   sell_thresh,
        }
