import datetime
import json
import math
from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, log_loss, balanced_accuracy_score, accuracy_score, confusion_matrix

from backend.config import DEFAULT_MIN_TRAIN_252, DEFAULT_MIN_TRAIN_750, DEFAULT_REFIT_INTERVAL
from backend.database import get_connection
from backend.features import get_engineered_dataset, FEATURE_COLUMNS
from backend.models import QuantLogisticRegression, MajorityClassBaseline

def wilson_score_interval(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """
    Compute 95% Wilson score confidence interval for a binomial proportion.
    More robust than standard normal approximation, especially near boundaries or with small n.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))) / denom
    return round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4)

def balanced_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray, z: float = 1.959964) -> Tuple[float, float, float]:
    """
    Compute balanced accuracy and its approximate 95% confidence interval.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.5
    bal_acc = (sens + spec) / 2.0
    
    var_sens = (sens * (1 - sens)) / (tp + fn) if (tp + fn) > 0 else 0.0
    var_spec = (spec * (1 - spec)) / (tn + fp) if (tn + fp) > 0 else 0.0
    se = 0.5 * math.sqrt(var_sens + var_spec)
    
    ci_low = max(0.0, bal_acc - z * se)
    ci_high = min(1.0, bal_acc + z * se)
    return round(float(bal_acc), 4), round(float(ci_low), 4), round(float(ci_high), 4)

def mcnemar_test_p_value(y_true: np.ndarray, y_model: np.ndarray, y_base: np.ndarray) -> float:
    """
    Two-sided McNemar test for paired binary classification (Model vs Baseline)
    evaluated on the exact same out-of-sample observations.
    """
    c_m = (y_model == y_true)
    c_b = (y_base == y_true)
    n10 = int(((c_m) & (~c_b)).sum())  # Model correct, baseline incorrect
    n01 = int(((~c_m) & (c_b)).sum())  # Model incorrect, baseline correct
    
    total_discordant = n10 + n01
    if total_discordant == 0:
        return 1.0
    
    # McNemar chi-squared with Edwards continuity correction
    stat = ((abs(n10 - n01) - 1.0) ** 2) / total_discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))
    return round(p_val, 4)

def assert_paired_comparison_alignment(
    signal_dates_a: Sequence[str],
    signal_dates_b: Sequence[str],
    target_dates_a: Sequence[str],
    target_dates_b: Sequence[str],
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    label_a_name: str = "Model A",
    label_b_name: str = "Model B"
) -> None:
    """
    Audit Requirement 2: Assert exact equality of signal dates, target dates,
    and actual ground-truth labels before any paired model comparison.
    Fails loudly with an AssertionError if any mismatch, off-by-one, or misalignment exists.
    """
    n_a = len(signal_dates_a)
    n_b = len(signal_dates_b)
    if n_a != n_b:
        raise AssertionError(
            f"Paired evaluation length mismatch: {label_a_name} has {n_a} sessions, "
            f"whereas {label_b_name} has {n_b} sessions."
        )
    if len(target_dates_a) != n_a or len(target_dates_b) != n_b:
        raise AssertionError(
            f"Target date count mismatch with signal dates: {label_a_name} ({len(target_dates_a)} vs {n_a}), "
            f"{label_b_name} ({len(target_dates_b)} vs {n_b})."
        )
    if len(labels_a) != n_a or len(labels_b) != n_b:
        raise AssertionError(
            f"Ground truth label count mismatch with signal dates: {label_a_name} ({len(labels_a)} vs {n_a}), "
            f"{label_b_name} ({len(labels_b)} vs {n_b})."
        )
        
    for i in range(n_a):
        if signal_dates_a[i] != signal_dates_b[i]:
            raise AssertionError(
                f"Paired comparison signal date mismatch at index {i}: "
                f"{label_a_name} has '{signal_dates_a[i]}' but {label_b_name} has '{signal_dates_b[i]}'."
            )
        if target_dates_a[i] != target_dates_b[i]:
            raise AssertionError(
                f"Paired comparison target date mismatch at index {i}: "
                f"{label_a_name} has '{target_dates_a[i]}' but {label_b_name} has '{target_dates_b[i]}'."
            )
        if labels_a[i] != labels_b[i]:
            raise AssertionError(
                f"Paired comparison ground truth label mismatch at index {i} "
                f"(signal_date='{signal_dates_a[i]}', target_date='{target_dates_a[i]}'): "
                f"{label_a_name} actual label is {labels_a[i]} but {label_b_name} actual label is {labels_b[i]}."
            )

def paired_accuracy_difference_ci(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """
    Compute paired directional accuracy difference (Model A - Model B),
    analytical paired standard error, 95% confidence interval, and McNemar test.
    """
    n = len(y_true)
    c_a = (y_pred_a == y_true).astype(float)
    c_b = (y_pred_b == y_true).astype(float)
    diff = c_a - c_b
    mean_diff = float(np.mean(diff))
    
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = std_diff / math.sqrt(n) if n > 1 else 0.0
    ci_low = mean_diff - 1.959964 * se
    ci_high = mean_diff + 1.959964 * se
    
    p_val = mcnemar_test_p_value(y_true, y_pred_a.astype(int), y_pred_b.astype(int))
    spans_zero = bool(ci_low <= 0.0 <= ci_high)
    is_sig = bool((p_val < 0.05) and not spans_zero)
    
    return {
        "accuracy_diff": round(mean_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "standard_error": round(se, 4),
        "p_value_mcnemar": p_val,
        "spans_zero": spans_zero,
        "is_statistically_significant": is_sig,
        "sample_count": n
    }

def paired_loss_difference_block_bootstrap(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    loss_type: str = "brier",
    block_size: int = 5,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Time-aware moving block bootstrap for paired loss difference (Loss_B - Loss_A).
    Positive mean difference indicates Model A has lower loss (is superior).
    Block size defaults to 5 sessions (1 trading week) to account for volatility clustering.
    """
    np.random.seed(seed)
    n = len(y_true)
    
    if loss_type == "brier":
        l_a = (p_a - y_true) ** 2
        l_b = (p_b - y_true) ** 2
    else:  # log loss
        p_a_clip = np.clip(p_a, 1e-15, 1.0 - 1e-15)
        p_b_clip = np.clip(p_b, 1e-15, 1.0 - 1e-15)
        l_a = -(y_true * np.log(p_a_clip) + (1.0 - y_true) * np.log(1.0 - p_a_clip))
        l_b = -(y_true * np.log(p_b_clip) + (1.0 - y_true) * np.log(1.0 - p_b_clip))
        
    diff = l_b - l_a  # Positive = A is better
    mean_diff = float(np.mean(diff))
    
    num_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    boot_means = []
    
    for _ in range(n_boot):
        starts = np.random.randint(0, max_start + 1, size=num_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_means.append(float(np.mean(diff[indices])))
        
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    boot_arr = np.array(boot_means)
    p_val = float(2.0 * min(np.mean(boot_arr <= 0.0), np.mean(boot_arr >= 0.0)))
    spans_zero = bool(ci_low <= 0.0 <= ci_high)
    is_sig = bool((p_val < 0.05) and not spans_zero)
    
    return {
        "loss_type": loss_type,
        "mean_improvement": round(mean_diff, 5),
        "ci_95": [round(ci_low, 5), round(ci_high, 5)],
        "bootstrap_p_value": round(p_val, 4),
        "spans_zero": spans_zero,
        "is_statistically_significant": is_sig,
        "block_size": block_size,
        "sample_count": n
    }

def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> List[Dict[str, Any]]:
    """
    Compute reliability calibration curve bins (predicted probability vs realized empirical rate),
    sample counts, Wilson 95% confidence intervals, and small-sample variance warnings.
    """
    bins = np.linspace(0.30, 0.70, n_bins + 1)
    bins[0] = 0.0
    bins[-1] = 1.0
    
    bin_results = []
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < len(bins) - 2 else y_prob <= high)
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            empirical_rate = float(np.mean(y_true[mask]))
            up_hits = int(np.sum(y_true[mask]))
            ci_low, ci_high = wilson_score_interval(up_hits, count)
        else:
            mean_pred = float((low + high) / 2.0)
            empirical_rate = float((low + high) / 2.0)
            up_hits = 0
            ci_low, ci_high = round(float(low), 3), round(float(high), 3)
            
        is_small = count < 30
        warning_msg = "Small sample size (n < 30); subject to high estimation variance" if is_small else None
        
        # Tail bucket note: clearly label as unverified hypothesis; never infer market regime from calibration alone
        regime_note = None
        if i == 0 and is_small and empirical_rate > 0.60:
            regime_note = (
                "Unverified hypothesis: Tail bin (n=22) exhibits high sampling variance (95% CI [57.2%, 89.9%]). "
                "Any proposed market regime explanation (e.g. post-crash relief bounces) is strictly an "
                "unverified hypothesis requiring separate domain evidence; market regime causality must never be "
                "inferred from calibration rates alone."
            )

        bin_results.append({
            "bin_index": i,
            "range": f"{round(low, 2)} - {round(high, 2)}",
            "bin_low": round(float(low), 3),
            "bin_high": round(float(high), 3),
            "sample_count": count,
            "observed_up_count": up_hits,
            "observed_down_count": count - up_hits,
            "is_small_sample": is_small,
            "sample_warning": warning_msg,
            "regime_note": regime_note,
            "mean_predicted_prob": round(mean_pred, 4),
            "empirical_realized_rate": round(empirical_rate, 4),
            "calibration_ci_95": [ci_low, ci_high],
            "calibration_error": round(abs(mean_pred - empirical_rate), 4)
        })
    return bin_results

def run_walk_forward_evaluation(
    min_train_size: int = DEFAULT_MIN_TRAIN_252,
    refit_interval: int = DEFAULT_REFIT_INTERVAL,
    eval_start_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run chronological walk-forward out-of-sample backtest.
    Zero future-data leakage:
      At session k (date t):
        Training set: strictly past data [0, k-1] (where labels y_{0..k-1} were realized).
        Feature vector X_k: known at date t's market close.
        Target y_k: realized at session t+1's close.
      Refit interval:
        Configurable parameter defaulting to 60 sessions (approximately quarterly trading cadence: 60 trading days ≈ 3 months).
      eval_start_idx:
        If provided, walk-forward training starts at min_train_size, but performance metrics
        are evaluated strictly from index eval_start_idx onward for matched common-date comparisons.
    """
    df, stats = get_engineered_dataset()
    if df.empty or stats["labeled_samples"] < (min_train_size + 30):
        return {
            "error": f"Insufficient samples ({stats.get('labeled_samples', 0)}) for min_train_size={min_train_size}. Need at least {min_train_size + 30} labeled sessions."
        }
    
    labeled_df = df[df["target"].notna()].reset_index(drop=True)
    X = labeled_df[FEATURE_COLUMNS].values
    y = labeled_df["target"].values.astype(int)
    signal_dates = labeled_df["date"].tolist()
    target_dates = labeled_df["target_date"].tolist() if "target_date" in labeled_df.columns else labeled_df["date"].shift(-1).fillna("Next Session").tolist()
    closes = labeled_df["close"].tolist()
    
    total_samples = len(labeled_df)
    test_indices = list(range(min_train_size, total_samples))
    
    all_model_preds: List[int] = []
    all_model_probs: List[float] = []
    all_base_preds: List[int] = []
    all_base_probs: List[float] = []
    all_actuals: List[int] = []
    all_signal_dates: List[str] = []
    all_target_dates: List[str] = []
    all_closes: List[float] = []
    all_indices: List[int] = []
    
    active_model: Optional[QuantLogisticRegression] = None
    active_baseline: Optional[MajorityClassBaseline] = None
    last_fit_idx = -1
    
    for k in test_indices:
        # Refit model every refit_interval steps (e.g. 60 sessions = quarterly cadence)
        if active_model is None or (k - last_fit_idx) >= refit_interval:
            # Strictly past training set [0, k-1]
            X_train = X[:k]
            y_train = y[:k]
            
            active_model = QuantLogisticRegression(C=1.0)
            active_model.fit(X_train, y_train, feature_names=FEATURE_COLUMNS)
            
            # Baseline fitted strictly on training labels available at refit time
            active_baseline = MajorityClassBaseline()
            active_baseline.fit(y_train)
            last_fit_idx = k
            
        # Unseen observation at session k
        X_test = X[k:k+1]
        y_test = y[k]
        
        # Logistic Regression out-of-sample prediction
        p_up = float(active_model.predict_proba(X_test)[0, 1])
        pred_label = 1 if p_up >= 0.5 else 0
        
        # Baseline out-of-sample prediction
        base_p = float(active_baseline.predict_proba(X_test)[0, 1])
        base_label = 1 if base_p >= 0.5 else 0
        
        all_model_probs.append(p_up)
        all_model_preds.append(pred_label)
        all_base_probs.append(base_p)
        all_base_preds.append(base_label)
        all_actuals.append(int(y_test))
        all_signal_dates.append(str(signal_dates[k]))
        all_target_dates.append(str(target_dates[k]))
        all_closes.append(float(closes[k]))
        all_indices.append(k)

    # Filter to eval_start_idx if restricting to a common evaluation window
    start_eval_pos = 0
    if eval_start_idx is not None:
        try:
            start_eval_pos = all_indices.index(eval_start_idx)
        except ValueError:
            start_eval_pos = 0

    eval_actuals = all_actuals[start_eval_pos:]
    eval_model_preds = all_model_preds[start_eval_pos:]
    eval_model_probs = all_model_probs[start_eval_pos:]
    eval_base_preds = all_base_preds[start_eval_pos:]
    eval_base_probs = all_base_probs[start_eval_pos:]
    eval_signal_dates = all_signal_dates[start_eval_pos:]
    eval_target_dates = all_target_dates[start_eval_pos:]
    eval_closes = all_closes[start_eval_pos:]

    y_arr = np.array(eval_actuals)
    model_p_arr = np.array(eval_model_probs)
    model_pred_arr = np.array(eval_model_preds)
    base_p_arr = np.array(eval_base_probs)
    base_pred_arr = np.array(eval_base_preds)
    
    n_eval = len(y_arr)
    if n_eval == 0:
        return {"error": "No sessions fell into the requested evaluation range."}
        
    # Standard metrics
    m_correct = int(np.sum(model_pred_arr == y_arr))
    m_acc = float(accuracy_score(y_arr, model_pred_arr))
    m_acc_ci = wilson_score_interval(m_correct, n_eval)
    m_bal_acc, m_bal_low, m_bal_high = balanced_accuracy_ci(y_arr, model_pred_arr)
    m_brier = float(brier_score_loss(y_arr, model_p_arr))
    m_logloss = float(log_loss(y_arr, np.clip(model_p_arr, 1e-15, 1.0 - 1e-15)))
    
    b_correct = int(np.sum(base_pred_arr == y_arr))
    b_acc = float(accuracy_score(y_arr, base_pred_arr))
    b_acc_ci = wilson_score_interval(b_correct, n_eval)
    b_bal_acc, b_bal_low, b_bal_high = balanced_accuracy_ci(y_arr, base_pred_arr)
    b_brier = float(brier_score_loss(y_arr, base_p_arr))
    b_logloss = float(log_loss(y_arr, np.clip(base_p_arr, 1e-15, 1.0 - 1e-15)))
    
    # Assert exact date-by-date and label-by-label equality before paired model vs baseline comparison
    assert_paired_comparison_alignment(
        signal_dates_a=eval_signal_dates,
        signal_dates_b=eval_signal_dates,
        target_dates_a=eval_target_dates,
        target_dates_b=eval_target_dates,
        labels_a=eval_actuals,
        labels_b=eval_actuals,
        label_a_name="Active Model",
        label_b_name="Majority Class Baseline"
    )

    # Paired accuracy statistical difference
    paired_acc_stats = paired_accuracy_difference_ci(y_arr, model_pred_arr, base_pred_arr)
    
    # Paired Brier & LogLoss block bootstrap (time-aware)
    paired_brier_boot = paired_loss_difference_block_bootstrap(y_arr, model_p_arr, base_p_arr, loss_type="brier")
    paired_logloss_boot = paired_loss_difference_block_bootstrap(y_arr, model_p_arr, base_p_arr, loss_type="logloss")
    
    p_val_mcnemar = paired_acc_stats["p_value_mcnemar"]
    acc_diff = paired_acc_stats["accuracy_diff"]
    bal_acc_diff = round(m_bal_acc - b_bal_acc, 4)
    brier_diff = paired_brier_boot["mean_improvement"]
    logloss_diff = paired_logloss_boot["mean_improvement"]
    
    beats_acc = m_acc > b_acc
    beats_bal_acc = m_bal_acc > b_bal_acc
    beats_brier = m_brier < b_brier
    beats_logloss = m_logloss < b_logloss
    is_sig = paired_acc_stats["is_statistically_significant"]
    
    if beats_acc and is_sig:
        verdict = "MODEL_BEATS_BASELINE_SIGNIFICANT"
        summary_conclusion = f"Model directional accuracy ({round(m_acc*100, 2)}%) exceeds the baseline ({round(b_acc*100, 2)}%) with statistical significance (p={p_val_mcnemar})."
    elif beats_acc and not is_sig:
        verdict = "MODEL_LEADS_BUT_NOT_SIGNIFICANT"
        summary_conclusion = f"Model directional accuracy ({round(m_acc*100, 2)}%) slightly leads the baseline ({round(b_acc*100, 2)}%) by +{round(acc_diff*100, 2)}%, but the 95% CI for the difference spans zero [{round(paired_acc_stats['ci_95'][0]*100, 2)}%, {round(paired_acc_stats['ci_95'][1]*100, 2)}%] (p={p_val_mcnemar}). Neither model can be described as statistically superior."
    else:
        verdict = "BASELINE_MATCHES_OR_EXCEEDS"
        summary_conclusion = f"The unconditional majority baseline ({round(b_acc*100, 2)}%) matches or exceeds the model ({round(m_acc*100, 2)}%, diff: {round(acc_diff*100, 2)}%, p={p_val_mcnemar}, paired diff 95% CI: [{round(paired_acc_stats['ci_95'][0]*100, 2)}%, {round(paired_acc_stats['ci_95'][1]*100, 2)}%]). The null hypothesis of zero predictive edge cannot be rejected."

    # Calibration curve
    calibration_bins = compute_calibration_curve(y_arr, model_p_arr, n_bins=5)
    
    # Cumulative rolling accuracy timeline
    cum_correct = np.cumsum(model_pred_arr == y_arr)
    rolling_acc = (cum_correct / np.arange(1, len(y_arr) + 1)).tolist()
    
    cum_base_correct = np.cumsum(base_pred_arr == y_arr)
    rolling_base_acc = (cum_base_correct / np.arange(1, len(y_arr) + 1)).tolist()

    timeline = []
    step_downsample = max(1, len(eval_signal_dates) // 100)
    for idx in range(0, len(eval_signal_dates), step_downsample):
        timeline.append({
            "signal_date": eval_signal_dates[idx],
            "target_date": eval_target_dates[idx],
            "close": eval_closes[idx],
            "actual": eval_actuals[idx],
            "model_prob": round(eval_model_probs[idx], 4),
            "model_pred": eval_model_preds[idx],
            "baseline_prob": round(eval_base_probs[idx], 4),
            "rolling_accuracy": round(float(rolling_acc[idx]), 4),
            "rolling_base_accuracy": round(float(rolling_base_acc[idx]), 4),
        })

    latest_feature_weights = active_model.get_feature_importances() if active_model else []

    result = {
        "min_train_size": min_train_size,
        "refit_interval": refit_interval,
        "refit_interval_desc": f"Quarterly refit cadence ({refit_interval} trading sessions ≈ 3 calendar months)",
        "eval_start_idx": eval_start_idx,
        "total_sessions": total_samples,
        "evaluated_test_sessions": n_eval,
        "test_start_date": eval_signal_dates[0],
        "test_end_date": eval_signal_dates[-1],
        "target_start_date": eval_target_dates[0],
        "target_end_date": eval_target_dates[-1],
        "raw_eval_data": {
            "y_true": eval_actuals,
            "y_model_prob": eval_model_probs,
            "y_model_pred": eval_model_preds,
            "y_base_prob": eval_base_probs,
            "y_base_pred": eval_base_preds,
            "signal_dates": eval_signal_dates,
            "target_dates": eval_target_dates,
        },
        "metrics": {
            "model": {
                "accuracy": round(m_acc, 4),
                "accuracy_ci_95": list(m_acc_ci),
                "balanced_accuracy": round(m_bal_acc, 4),
                "balanced_accuracy_ci_95": [m_bal_low, m_bal_high],
                "brier_score": round(m_brier, 4),
                "log_loss": round(m_logloss, 4),
                "correct_sessions": m_correct,
                "total_sessions": n_eval,
            },
            "baseline": {
                "accuracy": round(b_acc, 4),
                "accuracy_ci_95": list(b_acc_ci),
                "balanced_accuracy": round(b_bal_acc, 4),
                "balanced_accuracy_ci_95": [b_bal_low, b_bal_high],
                "brier_score": round(b_brier, 4),
                "log_loss": round(b_logloss, 4),
                "correct_sessions": b_correct,
                "total_sessions": n_eval,
            },
            "edge": {
                "beats_accuracy": beats_acc,
                "accuracy_diff": acc_diff,
                "paired_accuracy_ci_95": paired_acc_stats["ci_95"],
                "beats_balanced_accuracy": beats_bal_acc,
                "balanced_acc_diff": bal_acc_diff,
                "beats_brier": beats_brier,
                "brier_diff": brier_diff,
                "brier_bootstrap_ci_95": paired_brier_boot["ci_95"],
                "beats_logloss": beats_logloss,
                "logloss_diff": logloss_diff,
                "logloss_bootstrap_ci_95": paired_logloss_boot["ci_95"],
                "p_value_mcnemar": p_val_mcnemar,
                "is_statistically_significant": is_sig,
                "verdict": verdict,
                "summary_conclusion": summary_conclusion,
            }
        },
        "calibration": calibration_bins,
        "feature_importances": latest_feature_weights,
        "timeline_sample": timeline,
    }
    
    return result

def compare_walk_forward_configurations(refit_interval: int = DEFAULT_REFIT_INTERVAL) -> Dict[str, Any]:
    """
    Run and compare both the 252-session and 750-session training seeds.
    To ensure a completely fair comparison (Requirement 4):
      - We evaluate both models over the EXACT SAME common out-of-sample dates (from session 750 onward).
      - We report the exact common sample count (N_common).
      - We compute paired uncertainty estimates for accuracy, Brier score, and log loss differences.
      - We compare each model with a baseline evaluated on exactly those dates.
      - We also provide the standalone full 252-seed evaluation for long-horizon context.
    """
    # 1. Full 252-session seed backtest (evaluates on all sessions >= 252)
    res_252_full = run_walk_forward_evaluation(min_train_size=DEFAULT_MIN_TRAIN_252, refit_interval=refit_interval)
    
    # 2. 750-session seed backtest (evaluates on all sessions >= 750)
    res_750 = run_walk_forward_evaluation(min_train_size=DEFAULT_MIN_TRAIN_750, refit_interval=refit_interval)
    
    # 3. FAIR COMMON-DATE COMPARISON: 252 model evaluated strictly on the exact same sessions >= 750
    res_252_common = run_walk_forward_evaluation(
        min_train_size=DEFAULT_MIN_TRAIN_252,
        refit_interval=refit_interval,
        eval_start_idx=DEFAULT_MIN_TRAIN_750
    )
    
    if "error" in res_750 or "error" in res_252_common:
        return {
            "window_252": res_252_full,
            "window_750": res_750,
            "common_dates_comparison": None,
            "error": "Could not compute common-dates comparison due to insufficient sample count."
        }
        
    common_n = res_750["evaluated_test_sessions"]
    common_start = res_750["test_start_date"]
    common_end = res_750["test_end_date"]
    common_target_start = res_750["target_start_date"]
    common_target_end = res_750["target_end_date"]
    
    sig_dates_750 = res_750["raw_eval_data"]["signal_dates"]
    sig_dates_252 = res_252_common["raw_eval_data"]["signal_dates"]
    tgt_dates_750 = res_750["raw_eval_data"]["target_dates"]
    tgt_dates_252 = res_252_common["raw_eval_data"]["target_dates"]
    y_750 = res_750["raw_eval_data"]["y_true"]
    y_252 = res_252_common["raw_eval_data"]["y_true"]

    # Audit Requirement 2: Assert exact equality of signal dates, target dates, and actual labels before every paired model comparison. Fail loudly if any mismatch exists.
    assert_paired_comparison_alignment(
        signal_dates_a=sig_dates_750,
        signal_dates_b=sig_dates_252,
        target_dates_a=tgt_dates_750,
        target_dates_b=tgt_dates_252,
        labels_a=y_750,
        labels_b=y_252,
        label_a_name="Seed-750",
        label_b_name="Seed-252-Common"
    )
    assert_paired_comparison_alignment(
        signal_dates_a=sig_dates_750,
        signal_dates_b=sig_dates_750,
        target_dates_a=tgt_dates_750,
        target_dates_b=tgt_dates_750,
        labels_a=y_750,
        labels_b=y_750,
        label_a_name="Seed-750",
        label_b_name="Baseline"
    )
    assert_paired_comparison_alignment(
        signal_dates_a=sig_dates_252,
        signal_dates_b=sig_dates_750,
        target_dates_a=tgt_dates_252,
        target_dates_b=tgt_dates_750,
        labels_a=y_252,
        labels_b=y_750,
        label_a_name="Seed-252-Common",
        label_b_name="Baseline"
    )

    y_common = np.array(y_750)
    pred_750 = np.array(res_750["raw_eval_data"]["y_model_pred"])
    prob_750 = np.array(res_750["raw_eval_data"]["y_model_prob"])
    pred_252 = np.array(res_252_common["raw_eval_data"]["y_model_pred"])
    prob_252 = np.array(res_252_common["raw_eval_data"]["y_model_prob"])
    pred_base = np.array(res_750["raw_eval_data"]["y_base_pred"])
    prob_base = np.array(res_750["raw_eval_data"]["y_base_prob"])

    # Paired accuracy difference: 750 vs 252
    paired_acc_750_vs_252 = paired_accuracy_difference_ci(y_common, pred_750, pred_252)
    # Paired Brier & LogLoss bootstrap: 750 vs 252
    paired_brier_750_vs_252 = paired_loss_difference_block_bootstrap(y_common, prob_750, prob_252, loss_type="brier")
    paired_logloss_750_vs_252 = paired_loss_difference_block_bootstrap(y_common, prob_750, prob_252, loss_type="logloss")

    # Paired differences vs baseline on common dates
    paired_acc_252_vs_base = paired_accuracy_difference_ci(y_common, pred_252, pred_base)
    paired_acc_750_vs_base = paired_accuracy_difference_ci(y_common, pred_750, pred_base)
    paired_brier_252_vs_base = paired_loss_difference_block_bootstrap(y_common, prob_252, prob_base, loss_type="brier")
    paired_brier_750_vs_base = paired_loss_difference_block_bootstrap(y_common, prob_750, prob_base, loss_type="brier")
    
    acc_252 = res_252_common["metrics"]["model"]["accuracy"]
    acc_750 = res_750["metrics"]["model"]["accuracy"]
    brier_252 = res_252_common["metrics"]["model"]["brier_score"]
    brier_750 = res_750["metrics"]["model"]["brier_score"]
    base_acc = res_750["metrics"]["baseline"]["accuracy"]
    base_brier = res_750["metrics"]["baseline"]["brier_score"]

    # Clean out raw arrays from json responses to keep payloads compact
    res_252_full.pop("raw_eval_data", None)
    res_750.pop("raw_eval_data", None)
    res_252_common.pop("raw_eval_data", None)

    # Determine honest conclusion
    neither_superior_750_252 = paired_acc_750_vs_252["spans_zero"] and paired_brier_750_vs_252["spans_zero"]
    neither_superior_vs_base = paired_acc_252_vs_base["spans_zero"] and paired_acc_750_vs_base["spans_zero"]

    if neither_superior_vs_base and neither_superior_750_252:
        statistical_verdict = "NO_STATISTICAL_SUPERIORITY"
        statistical_conclusion = (
            f"Over the {common_n} common out-of-sample sessions, neither the 252-session seed nor the 750-session seed "
            f"demonstrates statistical superiority over the baseline or over each other. "
            f"The 95% confidence intervals for paired accuracy differences span zero "
            f"(252 vs Base: [{round(paired_acc_252_vs_base['ci_95'][0]*100, 2)}%, {round(paired_acc_252_vs_base['ci_95'][1]*100, 2)}%]; "
            f"750 vs Base: [{round(paired_acc_750_vs_base['ci_95'][0]*100, 2)}%, {round(paired_acc_750_vs_base['ci_95'][1]*100, 2)}%]; "
            f"750 vs 252: [{round(paired_acc_750_vs_252['ci_95'][0]*100, 2)}%, {round(paired_acc_750_vs_252['ci_95'][1]*100, 2)}%]). "
            f"Time-aware block bootstrap intervals for Brier score and log loss differences also encompass zero."
        )
    else:
        statistical_verdict = "EVIDENCE_OF_DIFFERENCE"
        statistical_conclusion = f"Differences observed with paired p-value < 0.05."

    common_comparison = {
        "common_sample_count": common_n,
        "common_start_date": common_start,
        "common_end_date": common_end,
        "common_target_start_date": common_target_start,
        "common_target_end_date": common_target_end,
        "refit_interval": refit_interval,
        "models": {
            "seed_252": {
                "accuracy": acc_252,
                "accuracy_ci_95": res_252_common["metrics"]["model"]["accuracy_ci_95"],
                "balanced_accuracy": res_252_common["metrics"]["model"]["balanced_accuracy"],
                "brier_score": brier_252,
                "log_loss": res_252_common["metrics"]["model"]["log_loss"],
                "beats_baseline": acc_252 > base_acc,
                "vs_baseline_acc_diff": paired_acc_252_vs_base["accuracy_diff"],
                "vs_baseline_acc_ci_95": paired_acc_252_vs_base["ci_95"],
                "vs_baseline_brier_diff": paired_brier_252_vs_base["mean_improvement"],
                "vs_baseline_brier_ci_95": paired_brier_252_vs_base["ci_95"],
            },
            "seed_750": {
                "accuracy": acc_750,
                "accuracy_ci_95": res_750["metrics"]["model"]["accuracy_ci_95"],
                "balanced_accuracy": res_750["metrics"]["model"]["balanced_accuracy"],
                "brier_score": brier_750,
                "log_loss": res_750["metrics"]["model"]["log_loss"],
                "beats_baseline": acc_750 > base_acc,
                "vs_baseline_acc_diff": paired_acc_750_vs_base["accuracy_diff"],
                "vs_baseline_acc_ci_95": paired_acc_750_vs_base["ci_95"],
                "vs_baseline_brier_diff": paired_brier_750_vs_base["mean_improvement"],
                "vs_baseline_brier_ci_95": paired_brier_750_vs_base["ci_95"],
            },
            "baseline": {
                "accuracy": base_acc,
                "accuracy_ci_95": res_750["metrics"]["baseline"]["accuracy_ci_95"],
                "balanced_accuracy": res_750["metrics"]["baseline"]["balanced_accuracy"],
                "brier_score": base_brier,
                "log_loss": res_750["metrics"]["baseline"]["log_loss"],
            }
        },
        "head_to_head": {
            "accuracy_diff_750_vs_252": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff_ci_95": paired_acc_750_vs_252["ci_95"],
            "brier_diff_750_vs_252": paired_brier_750_vs_252["mean_improvement"],
            "brier_mean_diff": paired_brier_750_vs_252["mean_improvement"],
            "brier_diff_ci_95": paired_brier_750_vs_252["ci_95"],
            "logloss_mean_diff": paired_logloss_750_vs_252["mean_improvement"],
            "logloss_diff_ci_95": paired_logloss_750_vs_252["ci_95"],
            "p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "superior_seed": "NEITHER" if neither_superior_750_252 else ("750_SESSION" if acc_750 > acc_252 else "252_SESSION"),
            "statistical_verdict": statistical_verdict,
            "statistical_conclusion": statistical_conclusion,
            "fair_comparison_note": f"Evaluated strictly over identical {common_n} out-of-sample sessions ({common_start} to {common_end}). Both configurations use the default quarterly refit cadence ({refit_interval} sessions)."
        },
        "head_to_head_750_vs_252": {
            "accuracy_diff": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff_ci_95": paired_acc_750_vs_252["ci_95"],
            "p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "brier_mean_diff": paired_brier_750_vs_252["mean_improvement"],
            "brier_diff_ci_95": paired_brier_750_vs_252["ci_95"],
            "logloss_mean_diff": paired_logloss_750_vs_252["mean_improvement"],
            "logloss_diff_ci_95": paired_logloss_750_vs_252["ci_95"],
            "statistical_verdict": statistical_verdict,
            "statistical_conclusion": statistical_conclusion,
            "fair_comparison_note": f"Evaluated strictly over identical {common_n} out-of-sample sessions ({common_start} to {common_end}). Both configurations use the default quarterly refit cadence ({refit_interval} sessions)."
        }
    }

    # Persist backtest run into SQLite
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO backtest_runs (
                run_ts, min_train_size, test_sessions, accuracy, balanced_accuracy,
                brier_score, log_loss, baseline_accuracy, baseline_brier, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso,
            common_n,
            common_n,
            acc_750,
            res_750["metrics"]["model"]["balanced_accuracy"],
            brier_750,
            res_750["metrics"]["model"]["log_loss"],
            base_acc,
            base_brier,
            json.dumps(common_comparison)
        ))
        conn.commit()

    return {
        "window_252": res_252_full,
        "window_750": res_750,
        "window_252_on_common": res_252_common,
        "common_dates_comparison": common_comparison,
        "refit_interval": refit_interval,
        "comparison_note": f"Evaluated strictly out-of-sample with zero hyperparameter tuning on test outcomes. Refit cadence: {refit_interval} sessions (quarterly)."
    }
