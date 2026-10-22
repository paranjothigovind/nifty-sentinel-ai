# NIFTY Sentinel AI (Phase 1) - Quantitative Research Engine & Terminal

An educational quantitative research platform for evaluating the statistical directional predictability of the NIFTY 50 (`^NSEI`) closing price using strict walk-forward machine learning, causality enforcement, and probability calibration.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    React 19 + TypeScript SPA                 │
│  - Bloomberg Dark Financial Research Terminal                │
│  - Price Candlestick & Volume Chart with Moving Averages     │
│  - Dynamic Empirical Class Balancer & Causal Features Table  │
│  - Walk-Forward Out-of-Sample Matrix (252 vs 750 Seeds)      │
│  - Reliability Calibration Diagrams & Brier / Log Loss Grids │
│  - Persisted Prediction Ledger & Realization Audit           │
│  - System Architecture & Component Manifest Inspector        │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP REST (/api/*)
┌──────────────────────────────▼───────────────────────────────┐
│         App Server & Proxy Gateway (Express, Port 3000)      │
│  - Proxies /api/quant/* to internal Python FastAPI (Port 8050)│
│  - Serves Vite development middleware / production SPA       │
│  - Integrates Gemini 3.8 Flash for qualitative explanations  │
└──────────────────────────────┬───────────────────────────────┘
                               │ Local IPC / HTTP Port 8050
┌──────────────────────────────▼───────────────────────────────┐
│          Quantitative Backend (Python 3.10 + FastAPI)        │
│  - Live yfinance OHLCV Data Provider & Fallback Engine       │
│  - SQLite Causal Storage (nifty_daily_ohlcv, logs, ledger)   │
│  - Causal Feature Engineering (Lagged returns, RSI, MACD, HV)│
│  - Expanding Walk-Forward Engine (Zero Lookahead Leakage)    │
│  - L2 Logistic Regression & Majority-Class Baseline Models   │
└──────────────────────────────────────────────────────────────┘
```

---

## Automated Test Suite

Run the full automated quantitative test suite covering feature leakage, chronological timing invariants, OHLC data integrity, probability metrics, and API routes:

```bash
python3 -m unittest discover -s backend/tests -p "test_*.py"
```

### Included Test Modules:
1. `backend/tests/test_leakage.py`: Proves bitwise feature invariance when future data is appended; asserts target is never present in the feature space.
2. `backend/tests/test_walk_forward.py`: Enforces chronological walk-forward timing invariants ($\max(\mathcal{I}_{\text{train}}) < k_{\text{test}}$).
3. `backend/tests/test_data_integrity.py`: Validates OHLC consistency rules ($H \ge L$, $H \ge O$, $H \ge C$, $L \le O$, $L \le C$, $V \ge 0$) and trading calendar gap checks.
4. `backend/tests/test_metrics.py`: Verifies Brier score, Log Loss, and reliability calibration bin calculations against mathematical benchmarks.
5. `backend/tests/test_common_dates_comparison.py`: Verifies fair evaluation of 252- and 750-session seeds over identical 434 sessions; enforces loud assertion failures on any signal/target date or label mismatch.
6. `backend/tests/test_statistical_audit.py`: Validates independent calibration bin calculations, Wilson score intervals, unverified hypothesis standards for tail buckets, paired bootstrap, and zero-crossing difference intervals.
7. `backend/tests/test_preprocessing_isolation.py`: Verifies scaler and preprocessor fit strictly on training data with zero validation/test leakage.
8. `backend/tests/test_timing_off_by_one.py`: Enforces target realization timing ($y_k = \mathbf{1}\{Close_{k+1} > Close_k\}$) and absence of lookahead.
9. `backend/tests/test_confidence_intervals.py`: Validates analytical Wilson score intervals and balanced accuracy variance formulas.
10. `backend/tests/test_refit_interval.py`: Validates configurable refit interval defaulting to 60 trading sessions (quarterly cadence).
11. `backend/tests/test_api.py`: Validates FastAPI route contracts (`/health`, `/system-status`, `/manifest`).

---

## Clean Environment Setup & Running the Tests

To run the complete test suite from a clean environment:

```bash
# 1. Create and activate a clean Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install all runtime and test dependencies from requirements manifest
pip install -r requirements.txt

# 3. Run the complete automated test suite
python3 -m unittest discover -s backend/tests -p "test_*.py"
```

### 2. Start Full-Stack Dev Server (Express + FastAPI + Vite)
```bash
npm run dev
```
The Express server on port 3000 will automatically start and supervise the Python FastAPI service on `127.0.0.1:8050` while proxying `/api/quant/*` requests and serving the React interface.

---

## Quantitative Methodology & Guarantees

1. **Zero Data Snooping**: Features for session $t$ use strictly data known at or before session $t$'s market close (15:30 IST).
2. **Expanding Walk-Forward Validation**: Models predicting session $k$ are fitted strictly on sessions $[0, k-1]$.
3. **Dual Window Comparison**: Out-of-sample performance is evaluated for both **252-session** (~1-year seed) and **750-session** (~3-year seed) initial training windows without hyperparameter re-tuning.
4. **Dynamic Empirical Target Balance**: The unconditional base rate $P(\text{Up session})$ is computed directly from the historical database (e.g. 52.28% from 1,184 labeled sessions), never hardcoded.
5. **No Fake Results**: All predictions, metrics, and charts are derived exclusively from verified historical data and live walk-forward execution.
6. **No Automated Trading**: Phase 1 is strictly an educational research platform.
