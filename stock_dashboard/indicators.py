"""
indicators.py
計算所有技術指標，回傳附加指標欄位的 DataFrame。
"""
import pandas as pd
import numpy as np


# ── 基礎計算 ─────────────────────────────────────────────────
def _sma(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w, min_periods=w).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── KD ───────────────────────────────────────────────────────
def calc_kd(df: pd.DataFrame, period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    RSV = (Close - LowestLow_n) / (HighestHigh_n - LowestLow_n) × 100
    K  = 前K × 2/3 + RSV × 1/3
    D  = 前D × 2/3 + K  × 1/3
    J  = 3K - 2D
    """
    ll = df["Low"].rolling(period, min_periods=1).min()
    hh = df["High"].rolling(period, min_periods=1).max()
    denom = (hh - ll).replace(0, np.nan)
    rsv = ((df["Close"] - ll) / denom * 100).fillna(50)

    k_vals, d_vals = [], []
    k, d = 50.0, 50.0
    for r in rsv:
        k = k * 2 / 3 + r / 3
        d = d * 2 / 3 + k / 3
        k_vals.append(k)
        d_vals.append(d)

    K = pd.Series(k_vals, index=df.index)
    D = pd.Series(d_vals, index=df.index)
    J = 3 * K - 2 * D
    return K, D, J


# ── RSI ──────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────
def calc_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line   = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, sig)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ──────────────────────────────────────────
def calc_bollinger(
    series: pd.Series, w: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = _sma(series, w)
    std   = series.rolling(w).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


# ── 主函式 ───────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """接收原始 OHLCV DataFrame，回傳附帶所有指標欄位的 DataFrame。"""
    if df is None or df.empty or len(df) < 10:
        return df

    result = df.copy()
    c = result["Close"]
    v = result["Volume"]

    # SMA
    for w in [5, 10, 20, 60, 120]:
        result[f"sma{w}"] = _sma(c, w)

    # Volume MA
    result["vol_ma5"]  = _sma(v, 5)
    result["vol_ma20"] = _sma(v, 20)

    # KD
    K, D, J = calc_kd(result)
    result["K"] = K
    result["D"] = D
    result["J"] = J

    # RSI
    result["rsi"] = calc_rsi(c)

    # MACD
    macd, sig, hist = calc_macd(c)
    result["macd"]        = macd
    result["macd_signal"] = sig
    result["macd_hist"]   = hist

    # Bollinger
    bb_up, bb_mid, bb_lo = calc_bollinger(c)
    result["bb_upper"]  = bb_up
    result["bb_middle"] = bb_mid
    result["bb_lower"]  = bb_lo

    # 近 N 日高低
    result["high_20"] = result["High"].rolling(20).max()
    result["low_20"]  = result["Low"].rolling(20).min()
    result["high_60"] = result["High"].rolling(60).max()
    result["low_60"]  = result["Low"].rolling(60).min()

    return result
