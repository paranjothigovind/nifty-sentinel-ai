# NIFTY Sentinel AI (Phase 1) - Quantitative Research Engine & Terminal

An educational quantitative research platform for evaluating statistical directional predictability of the NIFTY 50 index using walk-forward machine learning, strict leakage prevention, and calibrated probabilistic modeling.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> The following refined specifications govern the Phase 1 quantitative architecture and execution model:
> 1. **Verified Python FastAPI Environment**: Python 3.10 with `pip`, `fastapi`, `uvicorn`, `yfinance`, `pandas`, `numpy`, and `scikit-learn` runs as an active quantitative microservice bound to `127.0.0.1:8000`. If any external dependency fails, the system provides an audited fallback with pure standard-library mathematical routines, and explicitly reports service health via `/api/quant/system-status`.
> 2. **Express Port 3000 Gateway**: The Node.js Express server (`server.ts`) owns the single public port 3000 required by Google AI Studio, serves the React Vite frontend, and transparently proxies all `/api/quant/*` traffic to the FastAPI backend while supervising the Python process.
> 3. **Dual Walk-Forward Window Configurations (252 vs 750 Sessions)**: Out-of-sample evaluation will support both a **252-session** (~1 trading year seed) and a **750-session** (~3 trading years seed) expanding initial training window. Both configurations will be evaluated side-by-side on identical subsequent sessions without tuning on final test results.
> 4. **Dynamic Empirical Target Balance**: The historical class balance ($P(\text{Up session})$) is never hardcoded. It is computed dynamically from the ingested SQLite dataset ($\sum \mathbb{I}[C_{t+1} > C_t] / N$) and rendered alongside exact sample counts (e.g., `671 Up / 579 Down (53.68%)`).
> 5. **Strict No-Fabrication Policy**: Zero fake predictions, simulated metrics, or placeholder price feeds. All metrics, probabilities, and historical charts come exclusively from verified SQLite data and out-of-sample walk-forward runs. Empty states will guide the user to trigger live ingestion or backtesting.
> 6. **Automated Test Suite**: Comprehensive tests covering feature leakage, walk-forward chronological timing, OHLC calendar integrity, probabilistic scoring (Brier score & Log Loss), and FastAPI route integration.
> 7. **Component Implementation Manifest**: An explicit status ledger reporting implemented components, isolated test fixtures, and external dependencies.

---

### 1. Overview & Core Concept

- **What It Does**: NIFTY Sentinel AI ingests 5 years of daily NIFTY 50 (`^NSEI`) OHLCV records into a local SQLite repository, validates trading session calendars and checks for missing dates, computes causal technical indicators (lagged returns, moving averages, 14-day RSI, MACD 12/26/9, and rolling volatility), labels next-session close-to-close directional returns ($Y_{t} = \mathbb{I}[C_{t+1} > C_t]$), and conducts an expanding walk-forward out-of-sample backtest comparing a Logistic Regression model against an unconditional Majority-Class baseline.
- **Target Audience / Persona**: Quantitative researchers, algorithmic traders, financial engineering students, and market participants seeking disciplined statistical rigor without marketing hype or data snooping.
- **Key Value**: Replaces black-box "prediction" illusions with empirical reality: calibrated probabilities, Brier scores, log loss, balanced accuracy, and transparent verification that directional market prediction is a near 50/50 problem subject to regimes and volatility clustering.

---

### 2. User Experience & Visual Design

#### Key User Flows
1. **Overview & Ingestion Audit**: The user enters the terminal to see the latest ingestion timestamp, session count, data integrity score, missing-session diagnostic report, and the current NIFTY 50 price chart with volume and indicator overlays.
2. **Feature & Target Inspector**: Interactive exploration of causal feature distributions (1d/5d/20d returns, RSI, MACD histogram, 20d annualized volatility) paired with correlation heatmaps and the **dynamically computed empirical target balance**.
3. **Walk-Forward Experimentation (252 vs 750 Windows)**: The user inspects or triggers walk-forward evaluation, toggling or comparing the 252-session seed against the 750-session seed. A progressive visualizer displays out-of-sample rolling accuracy, cumulative Brier score convergence, and a reliability calibration curve comparing predicted probabilities against realized outcomes.
4. **Live Next-Session Forecast**: Displays the latest timestamped out-of-sample prediction ($P(\text{Up session})$) with feature contributions, stored permanently in the SQLite prediction ledger with pending verification against tomorrow's actual market close.
5. **Educational AI Analysis (Gemini)**: An on-demand "Interpret Regime" button sends the verified quantitative summary (model weights, recent volatility, calibration curve metrics) to Gemini to generate an academic critique of why the model leans bullish/bearish and where statistical uncertainty lies.
6. **System Status & Architecture Audit**: A dedicated drawer displaying runtime architecture, Python version, FastAPI Uvicorn PID, database path, and component manifest.

#### Visual Identity & Theme
- **Aesthetic Direction**: High-density institutional financial terminal (Bloomberg / FactSet posture) emphasizing mathematical clarity, low visual noise, and structured tabular grids.
- **Color Palette**:
  - Dominant Neutral (60%): `#090C10` deep obsidian background, `#121721` elevated panel background, `#1E2638` hairline panel borders.
  - Structural Typography (30%): `#E6EDF3` high-contrast primary text, `#8B949E` muted metadata, `#30363D` grid lines.
  - Accent Budget (10%): `#238636` / `#3FB950` institutional profit green, `#DA3633` / `#F85149` alert crimson, `#58A6FF` terminal cyan for model traces, `#D29922` amber for data warnings.
- **Typography & Hierarchy**:
  - Display & Headings: `Plus Jakarta Sans` (bold, geometric, clean).
  - Body Prose: `Public Sans` / system-ui (compact, legible).
  - Metrics & Telemetry: `JetBrains Mono` / `tabular-nums` for all price ticks, dates, timestamps, percentages, and matrix cells.
- **Anti-Slop Discipline**: No decorative pills, no code-comment headings (`// 01`), no pulsing status dots, no fake innovation scores, no cluttered card sandwiches.

---

### 3. Key Product Decisions & Trade-Offs

#### Decision 1: Causal Feature Construction & Strict Purging
- **Chosen Approach**: All features computed for session $t$ use data strictly known at $t$'s market close (15:30 IST). The target $y_t$ is defined as $\text{sign}(C_{t+1} - C_t)$. The walk-forward loop fits models only on $[0, t-1]$ when predicting for period $t$.
- **Why**: Eliminates lookahead bias (lookahead leakage is the #1 flaw in retail finance AI projects).
- **Alternatives Considered**: Training on random cross-validation splits (K-Fold) — rejected because standard K-Fold causes massive temporal leakage and inflated fake accuracy.

#### Decision 2: Dual Backend Architecture (FastAPI + Express Port 3000 Gateway)
- **Chosen Approach**: Python quantitative engine (`fastapi`, `uvicorn`, `sqlite3`, mathematical modeling) running as an internal service bound to `127.0.0.1:8000`, orchestrated and proxied seamlessly through the Node.js/Express server on port 3000.
- **Why**: Satisfies the environment constraint (dev server must bind to port 3000 in the AI Studio preview iframe) while providing true Python FastAPI REST endpoints for all data ingestion, feature calculations, and model fitting.
- **Alternatives Considered**: Pure client-side JavaScript math — rejected because user specifically requested Python + FastAPI for the quant backend.

#### Decision 3: Calibrated Probabilities Over Hard Binary Labels
- **Chosen Approach**: Evaluate models using Brier Score, Log Loss, and Reliability Calibration bins alongside Directional Accuracy and Balanced Accuracy.
- **Why**: In market direction forecasting, an uncalibrated 51% accuracy model with overconfident 90% probabilities will bankrupt a strategy; probability calibration is the hallmark of professional quantitative engineering.
- **Alternatives Considered**: Evaluating only Raw Accuracy — rejected because a naive model predicting 100% "Up" achieves ~53% accuracy during bull markets without learning any signal.

#### Decision 4: Comparative Windowing (252 vs 750 Sessions)
- **Chosen Approach**: Evaluate two distinct expanding window regimes:
  - 252 sessions (rapidly adapts to recent 1-year regime shifts, smaller training sample)
  - 750 sessions (requires 3 years of stability before first prediction, smoother estimates, fewer out-of-sample evaluation points)
- **Why**: Demonstrates how sample size vs regime adaptation affects quantitative stability.

---

### 4. Technical Architecture & Data Strategy

#### System & Component Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Client: React 19 + TypeScript                        │
│ ┌───────────────────┐ ┌────────────────────┐ ┌───────────────────────────┐ │
│ │ Price & Ingestion │ │ Feature Inspector  │ │ Walk-Forward Evaluation   │ │
│ │  - Candlestick/MA │ │  - Lagged Returns  │ │  - 252 vs 750 Windows     │ │
│ │  - Data Integrity │ │  - Dynamic Balance │ │  - Calibration Curves     │ │
│ └─────────┬─────────┘ └─────────┬──────────┘ └─────────────┬─────────────┘ │
│           │                     │                          │               │
│ ┌─────────▼─────────────────────▼──────────────────────────▼─────────────┐ │
│ │             Prediction Ledger, System Audit & Model Comparator         │ │
│ │       - Timestamped session predictions vs actual realized direction   │ │
│ │       - Architecture & Component Manifest Inspector                    │ │
│ │       - Optional Gemini quantitative explanation drawer                │ │
│ └──────────────────────────────────────┬─────────────────────────────────┘ │
└────────────────────────────────────────┼───────────────────────────────────┘
                                         │ HTTP REST (/api/*)
┌────────────────────────────────────────▼───────────────────────────────────┐
│              App Server & Proxy Gateway (Node Express, Port 3000)          │
│               - Serves Vite React frontend & handles Gemini SDK            │
│               - Proxies /api/quant/* to internal Python FastAPI Service    │
│               - Spawns & monitors FastAPI Uvicorn process (Port 8000)      │
└────────────────────────────────────────┬───────────────────────────────────┘
                                         │ Local IPC / HTTP Port 8000
┌────────────────────────────────────────▼───────────────────────────────────┐
│               Quantitative Backend (Python 3.10 + FastAPI)                │
│ ┌────────────────────┐ ┌───────────────────┐ ┌───────────────────────────┐ │
│ │ Data Provider      │ │ Feature Engine    │ │ Walk-Forward Evaluator    │ │
│ │  - yfinance / NSE  │ │  - RSI, MACD, HV  │ │  - 252 & 750 seed configs │ │
│ │  - Integrity check │ │  - Leakage checks │ │  - LogReg vs Majority     │ │
│ └─────────┬──────────┘ └─────────┬─────────┘ └─────────────┬─────────────┘ │
│           │                      │                         │               │
│ ┌─────────▼──────────────────────▼─────────────────────────▼─────────────┐ │
│ │                       SQLite Quantitative Repository                   │ │
│ │   - nifty_daily_ohlcv (date, open, high, low, close, volume, source_ts)│ │
│ │   - data_quality_logs (audit_ts, missing_sessions, anomalies)          │ │
│ │   - model_predictions (session_date, p_up, predicted_class, actual)   │ │
│ │   - backtest_metrics (metric_name, value, run_ts, window_type)         │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

#### API Endpoints Specification
- `GET /api/quant/health`: Health status, database session counts, Python version, latest session date.
- `GET /api/quant/system-status`: Detailed environment audit (FastAPI PID, active packages, SQLite table sizes).
- `POST /api/quant/sync`: Fetch/update 5-year NIFTY 50 OHLCV data, run calendar missing-session check, persist to SQLite.
- `GET /api/quant/history`: Retrieve validated OHLCV records with moving averages and volatility.
- `GET /api/quant/features`: Retrieve engineered features (lag_1, lag_2, lag_5, rsi_14, macd_diff, vol_20), target labels, and **dynamic empirical class balance**.
- `POST /api/quant/backtest`: Execute expanding walk-forward cross-validation (accepts `min_train_size`: 252 or 750); returns out-of-sample timeline, metrics (accuracy, balanced accuracy, Brier score, log loss, calibration bins), and feature coefficients.
- `GET /api/quant/prediction/latest`: Return current prediction for tomorrow's session with feature attribution.
- `GET /api/quant/predictions`: Return historical prediction ledger with realization status.
- `POST /api/explain`: Server-side Gemini endpoint (`@google/genai`) to generate quantitative executive summary without data synthesis.

#### Automated Test Suite Specifications
1. **Feature Leakage Test (`test_leakage.py`)**: Asserts that target $Y_t = \text{sign}(C_{t+1} - C_t)$ is strictly not present in $X_t$, and features computed up to session $t$ remain bitwise identical when new future sessions $t+1 \dots t+k$ are added to the dataset.
2. **Walk-Forward Timing Invariant (`test_walk_forward.py`)**: Asserts that at prediction session $k$, the model training index set $\mathcal{I}_{\text{train}} \subset [0, k-1]$ with $\max(\mathcal{I}_{\text{train}}) < k$. Tests both 252 and 750 minimum initial windows.
3. **Data Integrity Test (`test_data_integrity.py`)**: Tests OHLC consistency invariants ($H \ge L$, $H \ge O$, $H \ge C$, $L \le O$, $L \le C$, $V \ge 0$), duplicate timestamp rejection, and trading calendar gap flagging.
4. **Probability & Metric Tests (`test_metrics.py`)**: Tests Brier score, Log Loss, and Balanced Accuracy against known mathematical edge cases (e.g. perfect prediction, uniform 0.5 uncertainty, and inverted predictions).
5. **API & Proxy Integration Test (`test_api.py`)**: Tests end-to-end HTTP request and response contracts through FastAPI.

#### Component Manifest & Verification Matrix
| Component | Status | Source / Execution Context | Notes |
| :--- | :--- | :--- | :--- |
| **Python FastAPI Microservice** | LIVE | `backend/main.py` (Port 8000) | Native Python 3.10 with Uvicorn |
| **Express Gateway & Vite Host** | LIVE | `server.ts` (Port 3000) | Proxies `/api/quant/*`, serves React SPA |
| **Data Provider (yfinance)** | LIVE | `backend/data_provider.py` | Live Yahoo Finance query for `^NSEI` |
| **SQLite Repository** | LIVE | `backend/nifty_sentinel.db` | Causal schema with explicit timestamps |
| **Feature Engineering** | LIVE | `backend/features.py` | Lagged returns, SMA/EMA, RSI, MACD, HV |
| **Walk-Forward Engine (252 & 750)** | LIVE | `backend/backtest.py` | Expanding window cross-validation |
| **Logistic Regression & Baseline** | LIVE | `backend/models.py` | L2-regularized logistic regression & majority |
| **Prediction Ledger** | LIVE | SQLite `predictions` table | Persisted, auditable forecast records |
| **Dashboard UI** | LIVE | React 19 + TypeScript | Bloomberg dark theme with dense data grids |
| **Automated Test Suite** | LIVE | `backend/tests/` | 5 comprehensive unit & integration suites |
| **Synthetic Test Fixtures** | TEST ONLY | `backend/tests/fixtures/` | Explicitly marked `[SIMULATED FIXTURE]` |
| **Gemini AI Explanation** | OPTIONAL | `@google/genai` (Server-side) | Qualitative analysis of quant outputs only |
