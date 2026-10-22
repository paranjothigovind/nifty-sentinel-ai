import datetime
import json
import sqlite3
import urllib.request
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import yfinance as yf

from backend.config import TICKER_SYMBOL, LOOKBACK_YEARS, KNOWN_INDIAN_HOLIDAYS
from backend.database import get_connection

def fetch_yfinance_ohlcv(symbol: str = TICKER_SYMBOL, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV from yfinance with fallback to direct Yahoo Finance query.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(years * 365.25))
    
    df: Optional[pd.DataFrame] = None
    source_name = "yfinance"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")
        if df is not None and not df.empty:
            df = df.reset_index()
            # Standardize date column
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"yfinance fetch error: {e}. Falling back to Yahoo Finance direct chart API...")

    # Fallback to direct Yahoo Finance API if yfinance returned empty or failed
    if df is None or df.empty or len(df) < 100:
        source_name = "yahoo_direct_v8"
        period1 = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
        period2 = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d&includePrePost=false"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                indicators = result["indicators"]["quote"][0]
                
                rows = []
                for i, ts in enumerate(timestamps):
                    dt_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                    o = indicators.get("open", [None])[i]
                    h = indicators.get("high", [None])[i]
                    l = indicators.get("low", [None])[i]
                    c = indicators.get("close", [None])[i]
                    v = indicators.get("volume", [0])[i]
                    if o is not None and h is not None and l is not None and c is not None:
                        rows.append({"Date": dt_str, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v or 0})
                df = pd.DataFrame(rows)
        except Exception as e:
            print(f"Direct Yahoo fallback error: {e}")

    if df is None or df.empty:
        raise ValueError(f"Could not retrieve real market data for {symbol} from yfinance or Yahoo Finance API.")

    # Normalize column names
    col_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=col_map)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna().drop_duplicates(subset=["Date"]).sort_values("Date")
    df["source_name"] = source_name
    return df

def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate OHLC invariants, check for date continuity, and verify market session integrity.
    """
    anomalies: List[Dict[str, Any]] = []
    valid_rows = []
    
    for _, row in df.iterrows():
        d_str = str(row["Date"])
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])
        v = float(row["Volume"])
        
        # Check invariants
        reasons = []
        if h < l:
            reasons.append("High < Low")
        if h < o:
            reasons.append("High < Open")
        if h < c:
            reasons.append("High < Close")
        if l > o:
            reasons.append("Low > Open")
        if l > c:
            reasons.append("Low > Close")
        if c <= 0 or o <= 0:
            reasons.append("Non-positive price")
        if v < 0:
            reasons.append("Negative volume")
            
        if reasons:
            anomalies.append({"date": d_str, "reasons": reasons, "ohlcv": [o, h, l, c, v]})
        else:
            valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    if valid_df.empty:
        raise ValueError("All ingested rows failed OHLC validation checks.")

    # Missing trading day calendar audit (scan Mon-Fri weekdays)
    min_date = datetime.datetime.strptime(valid_df["Date"].min(), "%Y-%m-%d").date()
    max_date = datetime.datetime.strptime(valid_df["Date"].max(), "%Y-%m-%d").date()
    existing_dates = set(valid_df["Date"])
    
    curr = min_date
    missing_weekdays = []
    holiday_matches = []
    
    while curr <= max_date:
        if curr.weekday() < 5:  # Mon=0 ... Fri=4
            dt_s = curr.strftime("%Y-%m-%d")
            if dt_s not in existing_dates:
                mm_dd = curr.strftime("%m-%d")
                holiday_name = KNOWN_INDIAN_HOLIDAYS.get(mm_dd)
                if holiday_name:
                    holiday_matches.append({"date": dt_s, "holiday": holiday_name})
                else:
                    missing_weekdays.append(dt_s)
        curr += datetime.timedelta(days=1)

    audit_summary = {
        "total_sessions": len(valid_df),
        "start_date": valid_df["Date"].min(),
        "end_date": valid_df["Date"].max(),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies[:10],
        "missing_weekday_count": len(missing_weekdays),
        "missing_weekdays_sample": missing_weekdays[:10],
        "holiday_matches": len(holiday_matches),
        "status": "VALID" if len(anomalies) == 0 else "WARNED",
    }
    return valid_df, audit_summary

def sync_nifty_data(symbol: str = TICKER_SYMBOL, years: int = LOOKBACK_YEARS) -> Dict[str, Any]:
    """
    Fetch, validate, and persist 5 years of daily NIFTY 50 OHLCV data into SQLite.
    """
    raw_df = fetch_yfinance_ohlcv(symbol, years)
    valid_df, audit_summary = validate_ohlcv_data(raw_df)
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_name = raw_df["source_name"].iloc[0] if "source_name" in raw_df.columns else "yfinance"

    with get_connection() as conn:
        cursor = conn.cursor()
        for _, row in valid_df.iterrows():
            cursor.execute("""
                INSERT INTO nifty_daily_ohlcv (date, open, high, low, close, volume, source_ts, data_source, is_simulated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    source_ts=excluded.source_ts,
                    data_source=excluded.data_source
            """, (
                str(row["Date"]),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Volume"]),
                now_iso,
                source_name
            ))
            
        cursor.execute("""
            INSERT INTO data_quality_logs (
                audit_ts, total_sessions, start_date, end_date,
                missing_weekday_count, holiday_matches, anomaly_count, status, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso,
            audit_summary["total_sessions"],
            audit_summary["start_date"],
            audit_summary["end_date"],
            audit_summary["missing_weekday_count"],
            audit_summary["holiday_matches"],
            audit_summary["anomaly_count"],
            audit_summary["status"],
            json.dumps(audit_summary)
        ))
        conn.commit()

    return {
        "status": "success",
        "synced_at": now_iso,
        "symbol": symbol,
        "source": source_name,
        "audit": audit_summary
    }

def get_historical_ohlcv(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve validated OHLCV records from SQLite ordered by date ascending."""
    query = "SELECT date, open, high, low, close, volume, source_ts, data_source, is_simulated FROM nifty_daily_ohlcv ORDER BY date ASC"
    if limit:
        query = f"SELECT * FROM ({query} DESC LIMIT {int(limit)}) ORDER BY date ASC"
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def get_latest_data_quality_log() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent data quality audit log."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM data_quality_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            d = dict(row)
            try:
                d["details"] = json.loads(d["details_json"])
            except Exception:
                d["details"] = {}
            return d
        return None
