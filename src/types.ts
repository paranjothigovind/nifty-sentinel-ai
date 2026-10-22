export interface OHLCVRecord {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source_ts: string;
  data_source: string;
  is_simulated: number;
}

export interface DataQualityAudit {
  id?: number;
  audit_ts: string;
  total_sessions: number;
  start_date: string;
  end_date: string;
  missing_weekday_count: number;
  holiday_matches: number;
  anomaly_count: number;
  status: 'VALID' | 'WARNED' | 'CORRUPTED';
  details?: {
    anomalies?: Array<{ date: string; reasons: string[]; ohlcv: number[] }>;
    missing_weekdays_sample?: string[];
  };
}

export interface FeatureStats {
  total_records: number;
  usable_samples: number;
  labeled_samples: number;
  up_count: number;
  down_count: number;
  up_ratio: number;
  up_percentage: number;
  feature_columns: string[];
  latest_session_date: string | null;
  latest_close: number | null;
  message?: string;
}

export interface FeatureRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  return_1d: number;
  return_2d: number;
  return_5d: number;
  return_20d: number;
  sma_20_ratio: number;
  sma_50_ratio: number;
  rsi_14: number;
  macd_diff: number;
  vol_20: number;
  target?: number | null;
  next_close?: number | null;
  next_return?: number | null;
}

export interface CalibrationBin {
  bin_index: number;
  range: string;
  bin_low: number;
  bin_high: number;
  sample_count: number;
  is_small_sample?: boolean;
  sample_warning?: string | null;
  mean_predicted_prob: number;
  empirical_realized_rate: number;
  calibration_ci_95?: [number, number];
  calibration_error: number;
}

export interface FeatureImportance {
  feature: string;
  coefficient: number;
  odds_ratio: number;
  direction: 'BULLISH' | 'BEARISH';
  abs_importance: number;
}

export interface TimelinePoint {
  signal_date?: string;
  target_date?: string;
  date?: string;
  close: number;
  actual: number;
  model_prob: number;
  model_pred: number;
  baseline_prob: number;
  rolling_accuracy: number;
  rolling_base_accuracy: number;
}

export interface BacktestMetrics {
  accuracy: number;
  accuracy_ci_95?: [number, number];
  balanced_accuracy: number;
  balanced_accuracy_ci_95?: [number, number];
  brier_score: number;
  log_loss: number;
  correct_sessions?: number;
  total_sessions?: number;
}

export interface EdgeReport {
  beats_accuracy: boolean;
  accuracy_diff: number;
  beats_balanced_accuracy: boolean;
  balanced_acc_diff: number;
  beats_brier: boolean;
  brier_diff: number;
  beats_logloss: boolean;
  logloss_diff: number;
  p_value_mcnemar: number;
  is_statistically_significant: boolean;
  verdict: string;
  summary_conclusion: string;
}

export interface BacktestResult {
  min_train_size: number;
  refit_interval?: number;
  refit_interval_desc?: string;
  eval_start_idx?: number | null;
  total_sessions: number;
  evaluated_test_sessions: number;
  test_start_date: string;
  test_end_date: string;
  target_start_date?: string;
  target_end_date?: string;
  metrics: {
    model: BacktestMetrics;
    baseline: BacktestMetrics;
    edge: EdgeReport;
  };
  calibration: CalibrationBin[];
  feature_importances: FeatureImportance[];
  timeline_sample: TimelinePoint[];
  error?: string;
}

export interface CommonDatesComparison {
  common_sample_count: number;
  common_start_date: string;
  common_end_date: string;
  refit_interval: number;
  models: {
    seed_252: {
      accuracy: number;
      accuracy_ci_95: [number, number];
      balanced_accuracy: number;
      brier_score: number;
      log_loss: number;
      beats_baseline: boolean;
    };
    seed_750: {
      accuracy: number;
      accuracy_ci_95: [number, number];
      balanced_accuracy: number;
      brier_score: number;
      log_loss: number;
      beats_baseline: boolean;
    };
    baseline: {
      accuracy: number;
      accuracy_ci_95: [number, number];
      balanced_accuracy: number;
      brier_score: number;
      log_loss: number;
    };
  };
  head_to_head: {
    accuracy_diff_750_vs_252: number;
    brier_diff_750_vs_252: number;
    superior_seed: string;
    fair_comparison_note: string;
  };
}

export interface WindowComparison {
  window_252: BacktestResult;
  window_750: BacktestResult;
  window_252_on_common?: BacktestResult;
  common_dates_comparison?: CommonDatesComparison;
  refit_interval?: number;
  comparison_note: string;
}

export interface PredictionRecord {
  id?: number;
  prediction_ts: string;
  session_date: string;
  target_date: string;
  latest_close?: number;
  p_up: number;
  p_down?: number;
  predicted_class: number;
  predicted_direction?: string;
  confidence_delta?: number;
  model_type: string;
  training_samples_used?: number;
  features?: Record<string, number>;
  feature_attributions?: FeatureImportance[];
  actual_close?: number | null;
  target_close?: number | null;
  actual_direction?: number | null;
  is_realized?: number;
  outcome_status?: 'CORRECT' | 'INCORRECT' | 'PENDING';
  status?: string;
  note?: string;
  error?: string;
}

export interface SystemStatus {
  runtime: {
    platform: string;
    python_executable: string;
    python_version: string;
    fastapi_port: number;
    database_path: string;
    database_exists: boolean;
  };
  packages: {
    fastapi: string;
    scikit_learn: string;
    pandas: string;
    numpy: string;
    yfinance: string;
  };
  database_counts: {
    nifty_daily_ohlcv: number;
    model_predictions: number;
    backtest_runs: number;
  };
  data_quality_audit: DataQualityAudit | null;
}

export interface ManifestItem {
  name: string;
  status: 'LIVE' | 'TEST_ONLY' | 'EXTERNAL';
  source: string;
  notes: string;
}

export type DataMode = 'LIVE' | 'DELAYED' | 'HISTORICAL' | 'REPLAY' | 'UNAVAILABLE';
export type QualityStatus = 'VALID' | 'STALE' | 'INVALID' | 'UNAVAILABLE' | 'DEGRADED';
export type OptionType = 'CE' | 'PE';

export interface IndexCandle {
  symbol: string;
  interval: string;
  bar_start_utc: string;
  bar_end_utc: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
  provider_id: string;
  market_timestamp_utc: string;
  bar_start_ist?: string | null;
  bar_end_ist?: string | null;
  market_timestamp_ist?: string | null;
  data_mode: DataMode;
  is_delayed: boolean;
  delay_seconds?: number | null;
}

export interface OptionContract {
  underlying: string;
  expiry: string;
  strike: number;
  option_type: OptionType;
  instrument_id: string;
}

export interface OptionQuote {
  contract: OptionContract;
  bid: number | null;
  ask: number | null;
  ltp: number | null;
  volume: number | null;
  open_interest: number | null;
  iv: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
  market_timestamp_utc: string;
  market_timestamp_ist?: string | null;
  provider_id: string;
  data_mode: DataMode;
  quality_status: QualityStatus;
}

export interface MarketDataStatusSummary {
  active_mode: string;
  data_mode: DataMode;
  index_provider: {
    id: string;
    name: string;
    healthy: boolean;
    mode: string;
    last_market_ts_utc?: string | null;
    last_market_ts_ist: string | null;
    age_seconds: number;
  };
  options_provider: {
    id: string;
    name: string;
    healthy: boolean;
    mode: string;
    contracts_count: number;
    last_market_ts_utc?: string | null;
    last_market_ts_ist: string | null;
    age_seconds: number;
  };
  market_session: {
    is_open: boolean;
    session_note: string;
    trading_hours: string;
  };
  safety_rules: {
    completed_5m_bar_required: boolean;
    max_index_delay_sec: number;
    max_option_quote_age_sec: number;
    live_trading_permitted: boolean;
  };
  is_paper_trading_ready: boolean;
  trade_signal_blocked_reasons: string[];
}

export interface PaperEntrySafetyCheck {
  is_entry_permitted: boolean;
  block_reasons: string[];
  underlying_symbol: string;
  contract_id?: string | null;
  directional_signal: string;
  data_mode: DataMode;
  disclaimer: string;
}

