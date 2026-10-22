import datetime
import json
from typing import Dict, Any, List, Optional
import numpy as np

from backend.config import DEFAULT_MIN_TRAIN_252
from backend.database import get_connection
from backend.features import get_engineered_dataset, FEATURE_COLUMNS
from backend.models import QuantLogisticRegression

def generate_next_session_prediction(min_train_size: int = DEFAULT_MIN_TRAIN_252) -> Dict[str, Any]:
    """
    Generate an out-of-sample prediction for the upcoming trading session
    using all historical data up to today's close.
    Persists the prediction to SQLite model_predictions.
    """
    df, stats = get_engineered_dataset()
    if df.empty or stats["usable_samples"] < min_train_size:
        return {
            "error": "Insufficient historical data to generate prediction."
        }
    
    # Historical training set is all rows with known targets
    train_df = df[df["target"].notna()].reset_index(drop=True)
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values.astype(int)
    
    # Fit model on all available historical labeled sessions
    model = QuantLogisticRegression(C=1.0)
    model.fit(X_train, y_train, feature_names=FEATURE_COLUMNS)
    
    # Latest session is the very last row in df
    latest_row = df.iloc[-1]
    latest_date = str(latest_row["date"])
    latest_close = float(latest_row["close"])
    
    X_latest = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
    p_up = float(model.predict_proba(X_latest)[0, 1])
    predicted_class = 1 if p_up >= 0.5 else 0
    
    # Calculate estimated next trading date (skipping weekends)
    curr_dt = datetime.datetime.strptime(latest_date, "%Y-%m-%d").date()
    next_dt = curr_dt + datetime.timedelta(days=1)
    while next_dt.weekday() >= 5:  # Saturday or Sunday
        next_dt += datetime.timedelta(days=1)
    target_date = next_dt.strftime("%Y-%m-%d")
    
    feature_vals = {col: round(float(latest_row[col]), 5) for col in FEATURE_COLUMNS}
    feature_importances = model.get_feature_importances()
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if already predicted for this session
        cursor.execute("SELECT id FROM model_predictions WHERE session_date = ? AND model_type = 'LogisticRegression'", (latest_date,))
        existing = cursor.fetchone()
        
        if not existing:
            cursor.execute("""
                INSERT INTO model_predictions (
                    prediction_ts, session_date, target_date, p_up, predicted_class,
                    model_type, min_train_size, features_json, actual_close, target_close,
                    actual_direction, is_realized
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                now_iso,
                latest_date,
                target_date,
                p_up,
                predicted_class,
                "LogisticRegression",
                min_train_size,
                json.dumps(feature_vals),
                latest_close,
                None,
                None
            ))
            conn.commit()

    return {
        "prediction_ts": now_iso,
        "session_date": latest_date,
        "target_date": target_date,
        "latest_close": latest_close,
        "p_up": round(p_up, 4),
        "p_down": round(1.0 - p_up, 4),
        "predicted_direction": "UP (Bullish Close)" if predicted_class == 1 else "DOWN (Bearish Close)",
        "confidence_delta": round(abs(p_up - 0.5), 4),
        "training_samples_used": len(train_df),
        "features": feature_vals,
        "feature_attributions": feature_importances,
        "status": "PENDING_REALIZATION",
        "note": "Timestamped out-of-sample forecast. Will be compared with realized close when market concludes."
    }

def audit_and_reconcile_predictions() -> int:
    """
    Examine previously pending predictions in SQLite, check if target session close
    has since been recorded, and update realization outcomes.
    """
    updated_count = 0
    with get_connection() as conn:
        cursor = conn.cursor()
        # Find pending predictions
        cursor.execute("SELECT id, session_date, target_date, predicted_class FROM model_predictions WHERE is_realized = 0")
        pending = cursor.fetchall()
        
        for row in pending:
            pred_id = row["id"]
            target_date = row["target_date"]
            pred_class = row["predicted_class"]
            
            # Check if target_date exists in nifty_daily_ohlcv
            cursor.execute("SELECT close FROM nifty_daily_ohlcv WHERE date = ?", (target_date,))
            target_row = cursor.fetchone()
            
            if target_row:
                target_close = float(target_row["close"])
                # Get session_date close
                cursor.execute("SELECT close FROM nifty_daily_ohlcv WHERE date = ?", (row["session_date"],))
                session_row = cursor.fetchone()
                
                if session_row:
                    session_close = float(session_row["close"])
                    actual_dir = 1 if target_close > session_close else 0
                    
                    cursor.execute("""
                        UPDATE model_predictions SET
                            target_close = ?,
                            actual_direction = ?,
                            is_realized = 1
                        WHERE id = ?
                    """, (target_close, actual_dir, pred_id))
                    updated_count += 1
                    
        conn.commit()
    return updated_count

def get_predictions_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve persisted prediction records with realization status."""
    audit_and_reconcile_predictions()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, prediction_ts, session_date, target_date, p_up, predicted_class,
                   model_type, actual_close, target_close, actual_direction, is_realized, features_json
            FROM model_predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        
        result = []
        for r in rows:
            d = dict(r)
            if d["is_realized"]:
                is_correct = (d["predicted_class"] == d["actual_direction"])
                d["outcome_status"] = "CORRECT" if is_correct else "INCORRECT"
            else:
                d["outcome_status"] = "PENDING"
            try:
                d["features"] = json.loads(d["features_json"])
            except Exception:
                d["features"] = {}
            result.append(d)
        return result
