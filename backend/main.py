import sys
import os
import platform
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import TICKER_SYMBOL, DEFAULT_MIN_TRAIN_252, DEFAULT_MIN_TRAIN_750, DEFAULT_REFIT_INTERVAL, DB_PATH
from backend.database import init_db, get_connection
from backend.data_provider import sync_nifty_data, get_historical_ohlcv, get_latest_data_quality_log
from backend.features import get_engineered_dataset
from backend.backtest import run_walk_forward_evaluation, compare_walk_forward_configurations
from backend.predictions import generate_next_session_prediction, get_predictions_history, audit_and_reconcile_predictions
from backend.market_data import market_data_router

# Ensure database tables exist on startup
init_db()

app = FastAPI(
    title="NIFTY Sentinel AI - Quantitative Research API",
    description="Educational quantitative research backend for evaluating NIFTY 50 directional predictability.",
    version="1.0.0"
)

# Register Intraday Market Data Layer
app.include_router(market_data_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SyncRequest(BaseModel):
    symbol: Optional[str] = TICKER_SYMBOL
    years: Optional[int] = 5

class BacktestRequest(BaseModel):
    min_train_size: Optional[int] = DEFAULT_MIN_TRAIN_252
    refit_interval: Optional[int] = DEFAULT_REFIT_INTERVAL

@app.get("/health")
def health_check():
    session_count = 0
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM nifty_daily_ohlcv")
            session_count = cursor.fetchone()[0]
    except Exception as e:
        pass
        
    return {
        "status": "healthy",
        "service": "fastapi_quant_backend",
        "python_version": platform.python_version(),
        "database_sessions": session_count,
        "ticker": TICKER_SYMBOL
    }

@app.get("/system-status")
def system_status():
    """Detailed audit of runtime environment and component packages."""
    import sklearn
    import pandas
    import numpy
    import yfinance
    
    session_count = 0
    pred_count = 0
    backtest_count = 0
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM nifty_daily_ohlcv")
        session_count = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM model_predictions")
        pred_count = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM backtest_runs")
        backtest_count = cursor.fetchone()[0]

    return {
        "runtime": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "fastapi_port": 8050,
            "database_path": str(DB_PATH),
            "database_exists": DB_PATH.exists(),
        },
        "packages": {
            "fastapi": "0.160.0",
            "scikit_learn": sklearn.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "yfinance": yfinance.__version__,
        },
        "database_counts": {
            "nifty_daily_ohlcv": session_count,
            "model_predictions": pred_count,
            "backtest_runs": backtest_count
        },
        "data_quality_audit": get_latest_data_quality_log()
    }

@app.post("/sync")
def sync_data(req: SyncRequest = SyncRequest()):
    """Fetch live NIFTY 50 OHLCV data from yfinance, run calendar checks, and persist to SQLite."""
    try:
        result = sync_nifty_data(symbol=req.symbol or TICKER_SYMBOL, years=req.years or 5)
        # Reconcile any pending predictions
        audit_and_reconcile_predictions()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(limit: Optional[int] = Query(None, description="Number of recent sessions to retrieve")):
    """Retrieve historical daily OHLCV records."""
    records = get_historical_ohlcv(limit=limit)
    return {
        "ticker": TICKER_SYMBOL,
        "count": len(records),
        "data": records
    }

@app.get("/features")
def get_features():
    """Retrieve causal engineered features and dynamic empirical class balance."""
    df, stats = get_engineered_dataset()
    if df.empty:
        return {
            "status": "empty",
            "stats": stats,
            "sample_features": []
        }
    
    # Return summary + latest 50 rows of features for inspection (convert NaNs to None for valid JSON)
    records = df.tail(50).to_dict(orient="records")
    clean_sample = []
    for r in records:
        clean_row = {}
        for k, v in r.items():
            if isinstance(v, float) and (v != v):  # v != v detects NaN
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean_sample.append(clean_row)

    return {
        "status": "ready",
        "stats": stats,
        "features": clean_sample
    }

@app.post("/backtest")
def run_backtest(req: BacktestRequest = BacktestRequest()):
    """Run walk-forward out-of-sample backtest with specified minimum training window."""
    try:
        result = run_walk_forward_evaluation(
            min_train_size=req.min_train_size or DEFAULT_MIN_TRAIN_252,
            refit_interval=req.refit_interval or DEFAULT_REFIT_INTERVAL
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare")
def compare_windows(refit_interval: int = Query(DEFAULT_REFIT_INTERVAL, description="Trading sessions between model refits (default: 60 = quarterly)")):
    """Run and compare 252-session vs 750-session initial training window configurations with fair common-dates evaluation."""
    try:
        return compare_walk_forward_configurations(refit_interval=refit_interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prediction/latest")
def get_latest_prediction(min_train_size: int = DEFAULT_MIN_TRAIN_252):
    """Generate and persist timestamped out-of-sample forecast for next session."""
    try:
        pred = generate_next_session_prediction(min_train_size=min_train_size)
        if "error" in pred:
            raise HTTPException(status_code=400, detail=pred["error"])
        return pred
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions(limit: int = 50):
    """Retrieve persisted prediction ledger with realized outcome status."""
    return {
        "count": limit,
        "predictions": get_predictions_history(limit=limit)
    }

@app.get("/manifest")
def get_manifest():
    """Report implementation status of all system components."""
    return {
        "manifest": [
            {
                "name": "Python 3.10 FastAPI Backend",
                "status": "LIVE",
                "source": "backend/main.py on 127.0.0.1:8050",
                "notes": "Native Python service executing all quantitative computation"
            },
            {
                "name": "Node.js Express Gateway",
                "status": "LIVE",
                "source": "server.ts on port 3000",
                "notes": "Single public port proxying /api/quant/* and serving Vite SPA"
            },
            {
                "name": "Data Ingestion Provider",
                "status": "LIVE",
                "source": "yfinance / Yahoo Finance v8 direct API",
                "notes": "5-year daily NIFTY 50 (^NSEI) OHLCV records"
            },
            {
                "name": "SQLite Repository",
                "status": "LIVE",
                "source": "backend/nifty_sentinel.db",
                "notes": "Persistent tables with explicit source timestamps and audit logs"
            },
            {
                "name": "Causal Feature Engineering",
                "status": "LIVE",
                "source": "backend/features.py",
                "notes": "Lagged returns, moving averages, RSI-14, MACD, HV-20 with zero lookahead"
            },
            {
                "name": "Dynamic Class Balancer",
                "status": "LIVE",
                "source": "backend/features.py",
                "notes": "Calculates exact empirical target distribution from ingested data"
            },
            {
                "name": "Walk-Forward Engine (252 & 750)",
                "status": "LIVE",
                "source": "backend/backtest.py",
                "notes": "Expanding out-of-sample evaluation comparing LogReg vs Majority Baseline"
            },
            {
                "name": "Prediction Ledger & Audit",
                "status": "LIVE",
                "source": "backend/predictions.py",
                "notes": "Timestamped forecast storage and automatic realization reconciliation"
            },
            {
                "name": "Automated Test Suite",
                "status": "LIVE",
                "source": "backend/tests/",
                "notes": "Unit tests for leakage, timing invariants, OHLC integrity, and metrics"
            },
            {
                "name": "Synthetic Test Fixtures",
                "status": "TEST_ONLY",
                "source": "backend/tests/fixtures/",
                "notes": "Isolated synthetic OHLCV fixtures used exclusively during automated tests"
            },
            {
                "name": "Gemini Explanation Layer",
                "status": "LIVE",
                "source": "server.ts via @google/genai",
                "notes": "Server-side qualitative interpretation of verified quant results"
            },
            {
                "name": "Intraday Market Data Layer",
                "status": "LIVE",
                "source": "backend/market_data/ (FastAPI /market-data)",
                "notes": "Normalized index candles, options chains, freshness validator, and paper trading safety engine"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8050, reload=False)
