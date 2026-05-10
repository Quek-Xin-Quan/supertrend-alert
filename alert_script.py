import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ── Configuration (set these as GitHub Secrets / Variables) ───────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
GIST_TOKEN         = os.environ["GIST_TOKEN"]   # GitHub personal access token
GIST_ID            = os.environ["GIST_ID"]       # Gist ID for state tracking

# ── Indicator Settings (set as GitHub Variables, or edit defaults here) ───────
SYMBOL          = os.environ.get("SYMBOL",          "BTC-USD")  # e.g. AAPL, ETH-USD
INTERVAL        = os.environ.get("INTERVAL",        "1h")       # 1m,5m,15m,1h,1d
ATR_LEN         = int(os.environ.get("ATR_LEN",     "10"))
FACTOR          = float(os.environ.get("FACTOR",    "3.0"))
TRAINING_PERIOD = int(os.environ.get("TRAINING_PERIOD", "100"))
HIGH_VOL_PCT    = float(os.environ.get("HIGH_VOL_PCT",  "0.75"))
MID_VOL_PCT     = float(os.environ.get("MID_VOL_PCT",   "0.50"))
LOW_VOL_PCT     = float(os.environ.get("LOW_VOL_PCT",   "0.25"))

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
#  GIST STATE  (prevents duplicate alerts for the same bar)
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
        json={"files": {fname: {"content": json.dumps(state)}}}
    ).raise_for_status()

# ─────────────────────────────────────────────────────────────────────────────
#  ATR  (Exponential, matching Pine Script ta.atr)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_atr(df: pd.DataFrame, length: int) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

# ─────────────────────────────────────────────────────────────────────────────
#  K-MEANS CLUSTERING  (3 clusters: high / medium / low volatility)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_3(values: np.ndarray, high_pct: float, mid_pct: float, low_pct: float) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    centroids = np.array([
        vmin + (vmax - vmin) * high_pct,
        vmin + (vmax - vmin) * mid_pct,
        vmin + (vmax - vmin) * low_pct,
    ])
    for _ in range(300):
        dists     = np.abs(values[:, None] - centroids[None, :])  # (N, 3)
        labels    = dists.argmin(axis=1)
        new_cents = np.array([
            values[labels == k].mean() if (labels == k).any() else centroids[k]
            for k in range(3)
        ])
        if np.allclose(new_cents, centroids):
            break
        centroids = new_cents
    return centroids  # [high_centroid, mid_centroid, low_centroid]

# ─────────────────────────────────────────────────────────────────────────────
#  SUPERTREND
# ─────────────────────────────────────────────────────────────────────────────
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

        # Carry-forward band logic (matches Pine Script)
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
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Download OHLCV data
    period_map = {
        "1m": "7d",  "5m": "60d", "15m": "60d",
        "30m": "60d", "1h": "730d", "4h": "730d", "1d": "5y"
    }
    period = period_map.get(INTERVAL, "730d")
    print(f"[Data] Fetching {SYMBOL} | interval={INTERVAL} | period={period}")
    df = yf.download(SYMBOL, period=period, interval=INTERVAL,
                     auto_adjust=True, progress=False)

    # Flatten multi-index columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().iloc[-(TRAINING_PERIOD + 300):]

    if len(df) < TRAINING_PERIOD + 10:
        print("[Error] Not enough data to compute indicator.")
        return

    # 2. ATR
    atr = calculate_atr(df, ATR_LEN)

    # 3. Adaptive ATR via K-Means for each bar
    adaptive_atr = pd.Series(np.nan, index=df.index)
    for i in range(TRAINING_PERIOD - 1, len(df)):
        window = atr.values[i - TRAINING_PERIOD + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        centroids = kmeans_3(window, HIGH_VOL_PCT, MID_VOL_PCT, LOW_VOL_PCT)
        curr_val  = atr.values[i]
        dists     = np.abs(centroids - curr_val)
        adaptive_atr.iloc[i] = centroids[dists.argmin()]

    # 4. SuperTrend
    direction = compute_supertrend(df, FACTOR, adaptive_atr)

    # 5. Use index -2 (last CLOSED/confirmed bar); -1 may still be forming
    prev_dir      = direction[-3]
    curr_dir      = direction[-2]
    last_bar_time = str(df.index[-2])
    print(f"[Signal] Last closed bar: {last_bar_time} | prev_dir={prev_dir} | curr_dir={curr_dir}")

    # 6. Deduplication — skip if we already alerted for this bar
    state        = read_gist_state()
    last_alerted = state.get("last_alerted_bar", "")
    if last_bar_time == last_alerted:
        print(f"[Skip] Already alerted for bar {last_bar_time}.")
        return

    # 7. Detect crossover and send Telegram
    if prev_dir == 1 and curr_dir == -1:
        emoji  = "🟢"
        signal = "Bullish Trend Shift"
        detail = "SuperTrend flipped <b>UP</b> — potential buy signal"
    elif prev_dir == -1 and curr_dir == 1:
        emoji  = "🔴"
        signal = "Bearish Trend Shift"
        detail = "SuperTrend flipped <b>DOWN</b> — potential sell signal"
    else:
        print("[Info] No crossover detected. No alert sent.")
        return

    message = (
        f"{emoji} <b>{signal}</b>\n\n"
        f"📊 Symbol:    <b>{SYMBOL}</b>\n"
        f"⏱ Timeframe: <b>{INTERVAL}</b>\n"
        f"🕐 Bar Time:  <b>{last_bar_time}</b>\n\n"
        f"{detail}\n\n"
        f"<i>ML Adaptive SuperTrend Alert</i>"
    )
    send_telegram(message)

    # 8. Save state so we don't re-alert for the same bar
    state["last_alerted_bar"] = last_bar_time
    write_gist_state(state)
    print(f"[Done] Alert sent and state updated.")

if __name__ == "__main__":
    main()
