import sqlite3
import json
from typing import Optional, List, Dict, Any
from backend.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize database tables with strict causal schemas."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Historical OHLCV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nifty_daily_ohlcv (
                date TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source_ts TEXT NOT NULL,
                data_source TEXT NOT NULL,
                is_simulated INTEGER DEFAULT 0
            )
        """)
        
        # 2. Data quality and calendar audit logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_ts TEXT NOT NULL,
                total_sessions INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                missing_weekday_count INTEGER NOT NULL,
                holiday_matches INTEGER NOT NULL,
                anomaly_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
        """)
        
        # 3. Model prediction ledger (persisted predictions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_ts TEXT NOT NULL,
                session_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                p_up REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                min_train_size INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                actual_close REAL,
                target_close REAL,
                actual_direction INTEGER,
                is_realized INTEGER DEFAULT 0
            )
        """)
        
        # 4. Backtest evaluation run records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts TEXT NOT NULL,
                min_train_size INTEGER NOT NULL,
                test_sessions INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                balanced_accuracy REAL NOT NULL,
                brier_score REAL NOT NULL,
                log_loss REAL NOT NULL,
                baseline_accuracy REAL NOT NULL,
                baseline_brier REAL NOT NULL,
                metrics_json TEXT NOT NULL
            )
        """)
        
        conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
