import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.data_provider import get_historical_ohlcv

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_20d",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "macd_diff",
    "vol_20"
]

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute 14-day Wilder's Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))
    
    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, Signal line, and MACD Histogram (diff)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_diff = macd_line - signal_line
    return macd_line, signal_line, macd_diff

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate causal quantitative features on OHLCV DataFrame.
    Strictly preserves temporal causality: features for row t use only data <= t.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    
    # 1. Lagged percentage returns
    df["return_1d"] = (close / close.shift(1)) - 1.0
    df["return_2d"] = (close / close.shift(2)) - 1.0
    df["return_5d"] = (close / close.shift(5)) - 1.0
    df["return_20d"] = (close / close.shift(20)) - 1.0
    
    # 2. Moving average ratios (trend distance)
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    df["sma_20_ratio"] = (close / sma_20) - 1.0
    df["sma_50_ratio"] = (close / sma_50) - 1.0
    
    # 3. RSI 14
    df["rsi_14"] = compute_rsi(close, period=14)
    
    # 4. MACD 12/26/9
    _, _, macd_diff = compute_macd(close, fast=12, slow=26, signal=9)
    df["macd_diff"] = macd_diff
    
    # 5. Rolling 20-day annualized volatility of log returns
    log_ret = np.log(close / close.shift(1))
    df["vol_20"] = log_ret.rolling(window=20).std() * math.sqrt(252)
    
    # 6. Binary Target: Next session's close-to-close return > 0
    # Y_t = 1 if Close_{t+1} > Close_t else 0
    # Target date is explicitly session t+1's date
    next_close = close.shift(-1)
    df["target_date"] = df["date"].shift(-1)
    df["next_close"] = next_close
    df["next_return"] = (next_close - close) / close
    df["target"] = np.where(df["next_close"].isna(), np.nan, (next_close > close).astype(float))
    
    return df

def get_engineered_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load OHLCV from SQLite, engineer features, drop initial warmup NaNs,
    and calculate dynamic empirical target distribution.
    """
    records = get_historical_ohlcv()
    if not records or len(records) < 60:
        return pd.DataFrame(), {
            "total_records": 0,
            "usable_samples": 0,
            "up_count": 0,
            "down_count": 0,
            "up_ratio": 0.0,
            "message": "Insufficient historical data in SQLite. Please trigger data sync."
        }
    
    raw_df = pd.DataFrame(records)
    feat_df = calculate_features(raw_df)
    
    # Drop warmup rows where 50-day features are NaN
    # Note: Keep the final row (where target is NaN because tomorrow's close is pending)
    valid_mask = feat_df[FEATURE_COLUMNS].notna().all(axis=1)
    df_clean = feat_df[valid_mask].reset_index(drop=True)
    
    # Dynamic empirical class balance on labeled historical sessions (excluding pending latest row)
    labeled_df = df_clean[df_clean["target"].notna()]
    total_labeled = len(labeled_df)
    up_count = int((labeled_df["target"] == 1).sum()) if total_labeled > 0 else 0
    down_count = int((labeled_df["target"] == 0).sum()) if total_labeled > 0 else 0
    up_ratio = (up_count / total_labeled) if total_labeled > 0 else 0.0
    
    stats = {
        "total_records": len(records),
        "usable_samples": len(df_clean),
        "labeled_samples": total_labeled,
        "up_count": up_count,
        "down_count": down_count,
        "up_ratio": round(up_ratio, 4),
        "up_percentage": round(up_ratio * 100, 2),
        "feature_columns": FEATURE_COLUMNS,
        "latest_session_date": str(df_clean["date"].iloc[-1]) if not df_clean.empty else None,
        "latest_close": float(df_clean["close"].iloc[-1]) if not df_clean.empty else None,
    }
    
    return df_clean, stats
