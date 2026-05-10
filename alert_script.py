import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ── Configuration (set these as GitHub Secrets) ───────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
GIST_TOKEN         = os.environ["GIST_TOKEN"]
GIST_ID            = os.environ["GIST_ID"]

# ── Symbols — comma-separated list e.g. "GC=F,BTC-USD,AAPL,ETH-USD"
# Set SYMBOLS as a GitHub Variable, or edit the default here
SYMBOLS  = [s.strip() for s in os.environ.get("SYMBOLS", "GC=F,EURUSD=X,GBPUSD=X,AUDUSD=X,USDCAD=X,USDJPY=X,NZDUSD=X").split(",")]
INTERVAL = os.environ.get("INTERVAL", "5m")

# ── Indicator Settings ────────────────────────────────────────────────────────
ATR_LEN         = int(os.environ.get("ATR_LEN",          "10"))
FACTOR          = float(os.environ.get("FACTOR",          "3.0"))
TRAINING_PERIOD = int(os.environ.get("TRAINING_PERIOD",  "100"))
HIGH_VOL_PCT    = float(os.environ.get("HIGH_VOL_PCT",    "0.75"))
MID_VOL_PCT     = float(os.environ.get("MID_VOL_PCT",     "0.50"))
LOW_VOL_PCT     = float(os.environ.get("LOW_VOL_PCT",     "0.25"))

# Friendly display names (shown in Telegram message)
DISPLAY_NAMES = {
    # Forex pairs
    "XAUUSD=X": "XAU/USD (Gold)",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDJPY=X": "USD/JPY",
    "NZDUSD=X": "NZD/USD",
    # Commodities
    "GC=F":     "Gold Futures",
    "SI=F":     "Silver",
    "CL=F":     "Crude Oil",
    # Crypto
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    # Stocks
    "AAPL":     "Apple",
    "SPY":      "S&P 500 ETF",
}

# ─────────────────────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML"
    })
    resp.raise_for_status()
    print(f"[Telegram] Sent: {message}")

# ─────────────────────────────────────────────────────────────────────────────
#  GIST STATE  — stores last alerted bar per symbol, e.g.:
#  { "GC=F": "2026-05-10 08:00:00", "BTC-USD": "2026-05-10 09:00:00" }
# ─────────────────────────────────────────────────────────────────────────────
def _gist_headers():
    return {"Authorization": f"token {GIST_TOKEN}"}

def read_gist_state() -> dict:
    resp = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=_gist_headers())
    resp.raise_for_status()
    content = list(resp.json()["files"].values())[0]["content"]
    return json.loads(content)

def write_gist_state(state: dict):
    resp  = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=_gist_headers())
    fname = list(resp.json()["files"].keys())[0]
    requests.patch(
        f"https://api.github.com/gists/{GIST_ID}",
        headers=_gist_headers(),
        json={"files": {fname: {"content": json.dumps(state, indent=2)}}}
    ).raise_for_status()

# ─────────────────────────────────────────────────────────────────────────────
#  INDICATOR LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def calculate_atr(df: pd.DataFrame, length: int) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def kmeans_3(values: np.ndarray, high_pct: float, mid_pct: float, low_pct: float) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    centroids = np.array([
        vmin + (vmax - vmin) * high_pct,
        vmin + (vmax - vmin) * mid_pct,
        vmin + (vmax - vmin) * low_pct,
    ])
    for _ in range(300):
        dists     = np.abs(values[:, None] - centroids[None, :])
        labels    = dists.argmin(axis=1)
        new_cents = np.array([
            values[labels == k].mean() if (labels == k).any() else centroids[k]
            for k in range(3)
        ])
        if np.allclose(new_cents, centroids):
            break
        centroids = new_cents
    return centroids

def compute_supertrend(df: pd.DataFrame, factor: float, adaptive_atr: pd.Series) -> np.ndarray:
    hl2   = ((df["High"] + df["Low"]) / 2).values
    close = df["Close"].values
    atr   = adaptive_atr.values
    upper     = hl2 + factor * atr
    lower     = hl2 - factor * atr
    direction = np.zeros(len(df))
    st        = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if np.isnan(atr[i]) or np.isnan(atr[i - 1]):
            continue
        lower[i] = lower[i] if (lower[i] > lower[i-1] or close[i-1] < lower[i-1]) else lower[i-1]
        upper[i] = upper[i] if (upper[i] < upper[i-1] or close[i-1] > upper[i-1]) else upper[i-1]
        if np.isnan(st[i-1]):
            direction[i] = 1
        elif st[i-1] == upper[i-1]:
            direction[i] = -1 if close[i] > upper[i] else 1
        else:
            direction[i] =  1 if close[i] < lower[i] else -1
        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return direction

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK ONE SYMBOL
# ─────────────────────────────────────────────────────────────────────────────
def check_symbol(symbol: str, state: dict) -> tuple:
    """Returns (updated_state, alert_was_sent)."""
    print(f"\n{'='*50}")
    print(f"[{symbol}] Checking...")

    # 1. Download data
    period_map = {
        "1m": "7d",  "5m": "60d", "15m": "60d",
        "30m": "60d", "1h": "730d", "4h": "730d", "1d": "5y"
    }
    period = period_map.get(INTERVAL, "730d")
    df = yf.download(symbol, period=period, interval=INTERVAL,
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().iloc[-(TRAINING_PERIOD + 300):]

    if len(df) < TRAINING_PERIOD + 10:
        print(f"[{symbol}] Not enough data — skipping.")
        return state, False

    # 2. ATR + Adaptive ATR via K-Means
    atr = calculate_atr(df, ATR_LEN)
    adaptive_atr = pd.Series(np.nan, index=df.index)
    for i in range(TRAINING_PERIOD - 1, len(df)):
        window = atr.values[i - TRAINING_PERIOD + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        centroids        = kmeans_3(window, HIGH_VOL_PCT, MID_VOL_PCT, LOW_VOL_PCT)
        dists            = np.abs(centroids - atr.values[i])
        adaptive_atr.iloc[i] = centroids[dists.argmin()]

    # 3. SuperTrend direction
    direction     = compute_supertrend(df, FACTOR, adaptive_atr)
    prev_dir      = direction[-3]
    curr_dir      = direction[-2]
    last_bar_time = str(df.index[-2])
    print(f"[{symbol}] Bar: {last_bar_time} | prev={prev_dir} curr={curr_dir}")

    # 4. Deduplication — each symbol tracked separately in state
    if last_bar_time == state.get(symbol, ""):
        print(f"[{symbol}] Already alerted for this bar — skipping.")
        return state, False

    # 5. Detect crossover
    if prev_dir == 1 and curr_dir == -1:
        emoji, signal, detail = "🟢", "Bullish Trend Shift", "SuperTrend flipped <b>UP</b> — potential buy signal"
    elif prev_dir == -1 and curr_dir == 1:
        emoji, signal, detail = "🔴", "Bearish Trend Shift", "SuperTrend flipped <b>DOWN</b> — potential sell signal"
    else:
        print(f"[{symbol}] No crossover detected.")
        return state, False

    # 6. Send Telegram alert
    display = DISPLAY_NAMES.get(symbol, symbol)
    message = (
        f"{emoji} <b>{signal}</b>\n\n"
        f"📊 Symbol:    <b>{display} ({symbol})</b>\n"
        f"⏱ Timeframe: <b>{INTERVAL}</b>\n"
        f"🕐 Bar Time:  <b>{last_bar_time}</b>\n\n"
        f"{detail}\n\n"
        f"<i>ML Adaptive SuperTrend Alert</i>"
    )
    send_telegram(message)
    state[symbol] = last_bar_time
    return state, True

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — loops through all symbols
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[Config] Symbols:  {SYMBOLS}")
    print(f"[Config] Interval: {INTERVAL}")

    state         = read_gist_state()
    state_changed = False

    for symbol in SYMBOLS:
        try:
            state, alerted = check_symbol(symbol, state)
            if alerted:
                state_changed = True
        except Exception as e:
            print(f"[{symbol}] Error: {e}")

    # Save state only if at least one alert was sent
    if state_changed:
        write_gist_state(state)
        print("\n[Done] State updated in Gist.")
    else:
        print("\n[Done] No alerts sent. State unchanged.")

if __name__ == "__main__":
    main()
