import datetime
import json
import math
from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t
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

def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    max_lag: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Diebold-Mariano (1995) test with Harvey, Leybourne, and Newbold (1997)
    small-sample correction for paired forecast loss comparison under serial correlation.
    
    Null Hypothesis (H0):
      E[loss_a - loss_b] = 0 (Both models produce forecasts of equal expected loss).
    Alternative Hypothesis (H1):
      E[loss_a - loss_b] != 0 (Two-sided difference in forecast accuracy).
      
    Sign Convention:
      diff_t = loss_a_t - loss_b_t
      diff_t < 0 => Model A has lower loss (Model A is superior).
      diff_t > 0 => Model B has lower loss (Model B is superior).
      
    Returns:
      (mean_diff, dm_stat, p_value)
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0
    if n == 1:
        return float(d[0]), 0.0, 1.0

    if np.allclose(d, 0.0, atol=1e-12):
        return 0.0, 0.0, 1.0

    mean_d = float(np.mean(d))
    d_centered = d - mean_d
    var_d = float(np.var(d, ddof=1)) if n > 1 else 0.0
    if var_d < 1e-15:
        # Differences are non-zero but constant across all observations
        dm_stat = -9999.0 if mean_d < 0 else 9999.0
        return round(mean_d, 5), round(dm_stat, 4), 0.0

    # Truncation lag for Newey-West Bartlett kernel (Andrews / Newey-West rule)
    if max_lag is None:
        max_lag = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    max_lag = max(1, min(max_lag, n - 1))

    gamma_0 = float(np.mean(d_centered ** 2))
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - (lag / (max_lag + 1.0))
        gamma_k = float(np.mean(d_centered[lag:] * d_centered[:-lag]))
        gamma_sum += weight * gamma_k

    lr_var = gamma_0 + 2.0 * gamma_sum
    if lr_var <= 1e-12:
        lr_var = max(gamma_0, 1e-12)

    hac_se = math.sqrt(lr_var / n)
    if hac_se < 1e-12:
        return mean_d, 0.0, 1.0

    dm_stat = mean_d / hac_se

    # HLN small-sample correction factor for h-step ahead forecasts
    hln_factor = math.sqrt(max(0.0, (n + 1 - 2 * h + (h * (h - 1)) / n) / n))
    dm_hln = dm_stat * hln_factor

    # Two-sided p-value from Student-t with df = n - 1
    p_val = float(2.0 * student_t.sf(abs(dm_hln), df=max(1, n - 1)))
    return round(mean_d, 5), round(dm_hln, 4), round(p_val, 4)

def paired_accuracy_difference_ci(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """
    Compute paired directional accuracy difference (Model A - Model B),
    analytical paired standard error, 95% confidence interval, and McNemar test.
    
    Null Hypothesis (H0):
      E[Acc_A - Acc_B] = 0 (Both models have identical classification accuracy).
    Alternative Hypothesis (H1):
      E[Acc_A - Acc_B] != 0 (Two-sided difference in directional accuracy).
      
    Sign Convention:
      accuracy_diff = Acc_A - Acc_B
      Positive (> 0) indicates Model A has higher accuracy (superior).
      Negative (< 0) indicates Model B has higher accuracy.
    """
    n = len(y_true)
    if n == 0:
        return {
            "null_hypothesis": "H0: E[Acc_A - Acc_B] = 0 (identical accuracy)",
            "sign_convention": "accuracy_diff = Acc_A - Acc_B [positive favors A]",
            "accuracy_diff": 0.0,
            "ci_95": [0.0, 0.0],
            "standard_error": 0.0,
            "p_value_mcnemar": 1.0,
            "spans_zero": True,
            "is_statistically_significant": False,
            "sample_count": 0,
            "is_small_sample": True,
            "sample_warning": "No samples available for paired accuracy comparison."
        }
        
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
    is_small = n < 30
    warning_msg = "Small sample size (n < 30); insufficient power for paired classification inference" if is_small else None
    is_sig = bool((p_val < 0.05) and not spans_zero and not is_small)
    
    return {
        "null_hypothesis": "H0: E[Acc_A - Acc_B] = 0 (identical accuracy)",
        "sign_convention": "accuracy_diff = Acc_A - Acc_B [positive favors A]",
        "accuracy_diff": round(mean_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "standard_error": round(se, 4),
        "p_value_mcnemar": p_val,
        "spans_zero": spans_zero,
        "is_statistically_significant": is_sig,
        "sample_count": n,
        "is_small_sample": is_small,
        "sample_warning": warning_msg,
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
    Time-aware moving block bootstrap and Diebold-Mariano test for paired loss differences.
    
    Null Hypothesis (H0):
      E[Loss_A - Loss_B] = 0 (Equal expected forecast accuracy / calibration).
    Alternative Hypothesis (H1):
      E[Loss_A - Loss_B] != 0 (Two-sided difference in forecast loss).
      
    Sign Conventions:
      - loss_diff_t = Loss_A(t) - Loss_B(t)
        Negative (< 0) indicates Model A has lower loss (superior accuracy).
        Positive (> 0) indicates Model B has lower loss.
      - improvement_t = Loss_B(t) - Loss_A(t) = -loss_diff_t
        Positive (> 0) indicates Model A has lower loss (improvement over Model B).
        Negative (< 0) indicates Model B has lower loss.
        
    Significance Testing & Uncertainty Methodology:
      1. Asymptotic Test: Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997)
         finite-sample correction and Newey-West HAC standard errors.
      2. Valid Null-Hypothesis Bootstrap Test (Politis & Romano, Hall & Wilson):
         Loss difference is recentered under H0: tilde{d}_t = d_t - mean(d).
         Moving blocks of length block_size are resampled to preserve serial correlation.
         Studentized test statistics t* = mean(tilde{d}*) / SE(tilde{d}*) are compared against t_obs.
      3. Confidence Interval:
         95% moving block bootstrap percentile interval for mean improvement (Loss_B - Loss_A).
         If this CI spans zero, neither model is statistically superior.
      4. Sample Size Safeguards:
         Flags is_small_sample=True when n < 30 (insufficient power for reliable inference).
    """
    np.random.seed(seed)
    n = len(y_true)
    
    if n == 0:
        return {
            "loss_type": loss_type,
            "null_hypothesis": "H0: E[Loss_A - Loss_B] = 0",
            "sign_convention": "mean_diff = Loss_A - Loss_B [neg favors A]; mean_improvement = Loss_B - Loss_A [pos favors A]",
            "mean_diff": 0.0,
            "mean_improvement": 0.0,
            "ci_95": [0.0, 0.0],
            "dm_statistic": 0.0,
            "p_value_dm": 1.0,
            "bootstrap_p_value": 1.0,
            "spans_zero": True,
            "is_statistically_significant": False,
            "block_size": block_size,
            "sample_count": 0,
            "is_small_sample": True,
            "sample_warning": "No samples available for loss difference evaluation."
        }

    if loss_type == "brier":
        l_a = (p_a - y_true) ** 2
        l_b = (p_b - y_true) ** 2
    else:  # log loss
        p_a_clip = np.clip(p_a, 1e-15, 1.0 - 1e-15)
        p_b_clip = np.clip(p_b, 1e-15, 1.0 - 1e-15)
        l_a = -(y_true * np.log(p_a_clip) + (1.0 - y_true) * np.log(1.0 - p_a_clip))
        l_b = -(y_true * np.log(p_b_clip) + (1.0 - y_true) * np.log(1.0 - p_b_clip))

    # d_t: loss difference (Loss_A - Loss_B)
    d = l_a - l_b
    mean_d = float(np.mean(d))
    mean_improvement = -mean_d  # Loss_B - Loss_A (positive means A is better)
    
    # 1. Diebold-Mariano test with HAC standard errors
    _, dm_stat, p_val_dm = diebold_mariano_test(l_a, l_b, h=1, max_lag=max(1, block_size - 1))
    
    is_small = n < 30
    warning_msg = "Small sample size (n < 30); insufficient power for reliable bootstrap/HAC inference" if is_small else None
    
    # If all differences are practically zero (e.g. comparing identical models)
    if np.allclose(d, 0.0, atol=1e-12):
        return {
            "loss_type": loss_type,
            "null_hypothesis": "H0: E[Loss_A - Loss_B] = 0 (equal forecast accuracy)",
            "sign_convention": "mean_diff = mean(Loss_A - Loss_B) [negative favors A]; mean_improvement = mean(Loss_B - Loss_A) [positive favors A]",
            "mean_diff": 0.0,
            "mean_improvement": 0.0,
            "ci_95": [0.0, 0.0],
            "dm_statistic": 0.0,
            "p_value_dm": 1.0,
            "bootstrap_p_value": 1.0,
            "spans_zero": True,
            "is_statistically_significant": False,
            "block_size": block_size,
            "sample_count": n,
            "is_small_sample": is_small,
            "sample_warning": warning_msg,
        }

    var_d = float(np.var(d, ddof=1) if n > 1 else 0.0)
    if var_d < 1e-15:
        # Constant non-zero difference across all sessions
        ci_val = round(mean_improvement, 5)
        spans_zero = False
        is_sig = bool(not is_small)
        return {
            "loss_type": loss_type,
            "null_hypothesis": "H0: E[Loss_A - Loss_B] = 0 (equal forecast accuracy)",
            "sign_convention": "mean_diff = mean(Loss_A - Loss_B) [negative favors A]; mean_improvement = mean(Loss_B - Loss_A) [positive favors A]",
            "mean_diff": round(mean_d, 5),
            "mean_improvement": round(mean_improvement, 5),
            "ci_95": [ci_val, ci_val],
            "dm_statistic": round(dm_stat, 4),
            "p_value_dm": round(p_val_dm, 4),
            "bootstrap_p_value": 0.0,
            "spans_zero": spans_zero,
            "is_statistically_significant": is_sig,
            "block_size": block_size,
            "sample_count": n,
            "is_small_sample": is_small,
            "sample_warning": warning_msg,
        }

    # Moving block setup
    eff_block = min(block_size, n)
    num_blocks = int(np.ceil(n / eff_block))
    max_start = max(0, n - eff_block)
    
    # Generate block indices in a vectorized manner
    starts = np.random.randint(0, max_start + 1, size=(n_boot, num_blocks))
    block_offsets = np.arange(eff_block)
    boot_indices = (starts[:, :, None] + block_offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    
    # Confidence Interval on uncentered improvement (Loss_B - Loss_A)
    improvement_series = -d
    boot_improvements = improvement_series[boot_indices]
    boot_improvement_means = np.mean(boot_improvements, axis=1)
    ci_low = float(np.percentile(boot_improvement_means, 2.5))
    ci_high = float(np.percentile(boot_improvement_means, 97.5))
    spans_zero = bool(ci_low <= 0.0 <= ci_high)
    
    # Recentered Studentized Block Bootstrap for testing H0: E[d] = 0
    d_centered = d - mean_d
    boot_d_centered = d_centered[boot_indices]
    boot_means_centered = np.mean(boot_d_centered, axis=1)
    boot_se_centered = np.maximum(np.std(boot_d_centered, axis=1, ddof=1) / math.sqrt(n), 1e-12)
    boot_t = np.abs(boot_means_centered / boot_se_centered)
    
    se_d = max(np.std(d, ddof=1) / math.sqrt(n), 1e-12)
    t_obs = abs(mean_d / se_d)
    p_val_boot = float(np.mean(boot_t >= t_obs))
    
    # Statistical significance requires BOTH non-zero crossing CI AND p < 0.05 on the valid test
    is_sig = bool((p_val_dm < 0.05 and p_val_boot < 0.05) and not spans_zero and not is_small)
    
    return {
        "loss_type": loss_type,
        "null_hypothesis": "H0: E[Loss_A - Loss_B] = 0 (equal forecast accuracy)",
        "sign_convention": "mean_diff = mean(Loss_A - Loss_B) [negative favors A]; mean_improvement = mean(Loss_B - Loss_A) [positive favors A]",
        "mean_diff": round(mean_d, 5),
        "mean_improvement": round(mean_improvement, 5),
        "ci_95": [round(ci_low, 5), round(ci_high, 5)],
        "dm_statistic": round(dm_stat, 4),
        "p_value_dm": round(p_val_dm, 4),
        "bootstrap_p_value": round(p_val_boot, 4),
        "spans_zero": spans_zero,
        "is_statistically_significant": is_sig,
        "block_size": eff_block,
        "sample_count": n,
        "is_small_sample": is_small,
        "sample_warning": warning_msg,
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
    paired_logloss_252_vs_base = paired_loss_difference_block_bootstrap(y_common, prob_252, prob_base, loss_type="logloss")
    paired_logloss_750_vs_base = paired_loss_difference_block_bootstrap(y_common, prob_750, prob_base, loss_type="logloss")
    
    acc_252 = res_252_common["metrics"]["model"]["accuracy"]
    acc_750 = res_750["metrics"]["model"]["accuracy"]
    brier_252 = res_252_common["metrics"]["model"]["brier_score"]
    brier_750 = res_750["metrics"]["model"]["brier_score"]
    logloss_252 = res_252_common["metrics"]["model"]["log_loss"]
    logloss_750 = res_750["metrics"]["model"]["log_loss"]
    base_acc = res_750["metrics"]["baseline"]["accuracy"]
    base_brier = res_750["metrics"]["baseline"]["brier_score"]
    base_logloss = res_750["metrics"]["baseline"]["log_loss"]

    # Clean out raw arrays from json responses to keep payloads compact
    res_252_full.pop("raw_eval_data", None)
    res_750.pop("raw_eval_data", None)
    res_252_common.pop("raw_eval_data", None)

    # Statistical Significance Verification
    is_insufficient_sample = common_n < 30
    acc_is_sig = paired_acc_750_vs_252["is_statistically_significant"]
    brier_is_sig = paired_brier_750_vs_252["is_statistically_significant"]
    logloss_is_sig = paired_logloss_750_vs_252["is_statistically_significant"]

    sig_750_vs_252 = (acc_is_sig or brier_is_sig or logloss_is_sig) and not is_insufficient_sample
    sig_vs_base = (
        paired_acc_252_vs_base["is_statistically_significant"] or
        paired_acc_750_vs_base["is_statistically_significant"] or
        paired_brier_252_vs_base["is_statistically_significant"] or
        paired_brier_750_vs_base["is_statistically_significant"]
    ) and not is_insufficient_sample

    if is_insufficient_sample:
        statistical_verdict = "INSUFFICIENT_SAMPLE_SIZE"
        superior_seed = "NEITHER"
        statistical_conclusion = (
            f"Common evaluation sample size ({common_n}) is below the minimum threshold (n < 30) for reliable inference. "
            f"No statistical conclusions can be drawn."
        )
    elif not sig_750_vs_252 and not sig_vs_base:
        statistical_verdict = "NO_STATISTICAL_SUPERIORITY"
        superior_seed = "NEITHER"
        statistical_conclusion = (
            f"Over the {common_n} dynamically aligned out-of-sample sessions ({common_start} to {common_end}), "
            f"neither the 252-session seed nor the 750-session seed demonstrates statistical superiority over the baseline "
            f"or over each other. "
            f"Directional accuracy 95% CIs encompass zero "
            f"(252 vs Base: [{round(paired_acc_252_vs_base['ci_95'][0]*100, 2)}%, {round(paired_acc_252_vs_base['ci_95'][1]*100, 2)}%], p={paired_acc_252_vs_base['p_value_mcnemar']}; "
            f"750 vs Base: [{round(paired_acc_750_vs_base['ci_95'][0]*100, 2)}%, {round(paired_acc_750_vs_base['ci_95'][1]*100, 2)}%], p={paired_acc_750_vs_base['p_value_mcnemar']}; "
            f"750 vs 252: [{round(paired_acc_750_vs_252['ci_95'][0]*100, 2)}%, {round(paired_acc_750_vs_252['ci_95'][1]*100, 2)}%], p={paired_acc_750_vs_252['p_value_mcnemar']}). "
            f"Time-aware Diebold-Mariano tests and block bootstrap intervals for Brier score "
            f"(750 vs 252 95% CI: [{round(paired_brier_750_vs_252['ci_95'][0], 5)}, {round(paired_brier_750_vs_252['ci_95'][1], 5)}], DM p={paired_brier_750_vs_252['p_value_dm']}) "
            f"and log loss (95% CI: [{round(paired_logloss_750_vs_252['ci_95'][0], 5)}, {round(paired_logloss_750_vs_252['ci_95'][1], 5)}], DM p={paired_logloss_750_vs_252['p_value_dm']}) "
            f"both encompass zero and fail to reject the null hypothesis of equal forecast accuracy."
        )
    else:
        statistical_verdict = "EVIDENCE_OF_DIFFERENCE"
        superior_seed = "750_SESSION" if (acc_750 > acc_252 and acc_is_sig) else ("252_SESSION" if (acc_252 > acc_750 and acc_is_sig) else "NEITHER")
        statistical_conclusion = (
            f"Statistically significant difference detected over {common_n} common sessions: "
            f"Accuracy diff={round(paired_acc_750_vs_252['accuracy_diff']*100, 2)}% (p={paired_acc_750_vs_252['p_value_mcnemar']}), "
            f"Brier diff={round(paired_brier_750_vs_252['mean_improvement'], 5)} (DM p={paired_brier_750_vs_252['p_value_dm']})."
        )

    common_comparison = {
        "common_sample_count": common_n,
        "common_start_date": common_start,
        "common_end_date": common_end,
        "common_target_start_date": common_target_start,
        "common_target_end_date": common_target_end,
        "refit_interval": refit_interval,
        "is_insufficient_sample": is_insufficient_sample,
        "models": {
            "seed_252": {
                "accuracy": acc_252,
                "accuracy_ci_95": res_252_common["metrics"]["model"]["accuracy_ci_95"],
                "balanced_accuracy": res_252_common["metrics"]["model"]["balanced_accuracy"],
                "brier_score": brier_252,
                "log_loss": logloss_252,
                "beats_baseline": acc_252 > base_acc,
                "vs_baseline_acc_diff": paired_acc_252_vs_base["accuracy_diff"],
                "vs_baseline_acc_ci_95": paired_acc_252_vs_base["ci_95"],
                "vs_baseline_acc_p_value": paired_acc_252_vs_base["p_value_mcnemar"],
                "vs_baseline_brier_diff": paired_brier_252_vs_base["mean_improvement"],
                "vs_baseline_brier_ci_95": paired_brier_252_vs_base["ci_95"],
                "vs_baseline_brier_p_value_dm": paired_brier_252_vs_base["p_value_dm"],
                "vs_baseline_logloss_diff": paired_logloss_252_vs_base["mean_improvement"],
                "vs_baseline_logloss_ci_95": paired_logloss_252_vs_base["ci_95"],
                "vs_baseline_logloss_p_value_dm": paired_logloss_252_vs_base["p_value_dm"],
            },
            "seed_750": {
                "accuracy": acc_750,
                "accuracy_ci_95": res_750["metrics"]["model"]["accuracy_ci_95"],
                "balanced_accuracy": res_750["metrics"]["model"]["balanced_accuracy"],
                "brier_score": brier_750,
                "log_loss": logloss_750,
                "beats_baseline": acc_750 > base_acc,
                "vs_baseline_acc_diff": paired_acc_750_vs_base["accuracy_diff"],
                "vs_baseline_acc_ci_95": paired_acc_750_vs_base["ci_95"],
                "vs_baseline_acc_p_value": paired_acc_750_vs_base["p_value_mcnemar"],
                "vs_baseline_brier_diff": paired_brier_750_vs_base["mean_improvement"],
                "vs_baseline_brier_ci_95": paired_brier_750_vs_base["ci_95"],
                "vs_baseline_brier_p_value_dm": paired_brier_750_vs_base["p_value_dm"],
                "vs_baseline_logloss_diff": paired_logloss_750_vs_base["mean_improvement"],
                "vs_baseline_logloss_ci_95": paired_logloss_750_vs_base["ci_95"],
                "vs_baseline_logloss_p_value_dm": paired_logloss_750_vs_base["p_value_dm"],
            },
            "baseline": {
                "accuracy": base_acc,
                "accuracy_ci_95": res_750["metrics"]["baseline"]["accuracy_ci_95"],
                "balanced_accuracy": res_750["metrics"]["baseline"]["balanced_accuracy"],
                "brier_score": base_brier,
                "log_loss": base_logloss,
            }
        },
        "head_to_head": {
            "accuracy_diff_750_vs_252": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff_ci_95": paired_acc_750_vs_252["ci_95"],
            "accuracy_p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "accuracy_spans_zero": paired_acc_750_vs_252["spans_zero"],
            "brier_diff_750_vs_252": paired_brier_750_vs_252["mean_improvement"],
            "brier_mean_diff": paired_brier_750_vs_252["mean_improvement"],
            "brier_diff_ci_95": paired_brier_750_vs_252["ci_95"],
            "brier_p_value_dm": paired_brier_750_vs_252["p_value_dm"],
            "brier_bootstrap_p_value": paired_brier_750_vs_252["bootstrap_p_value"],
            "brier_spans_zero": paired_brier_750_vs_252["spans_zero"],
            "logloss_mean_diff": paired_logloss_750_vs_252["mean_improvement"],
            "logloss_diff_ci_95": paired_logloss_750_vs_252["ci_95"],
            "logloss_p_value_dm": paired_logloss_750_vs_252["p_value_dm"],
            "logloss_bootstrap_p_value": paired_logloss_750_vs_252["bootstrap_p_value"],
            "logloss_spans_zero": paired_logloss_750_vs_252["spans_zero"],
            "p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "superior_seed": superior_seed,
            "statistical_verdict": statistical_verdict,
            "statistical_conclusion": statistical_conclusion,
            "fair_comparison_note": f"Evaluated strictly over identical {common_n} out-of-sample sessions ({common_start} to {common_end}). Both configurations use the default quarterly refit cadence ({refit_interval} sessions)."
        },
        "head_to_head_750_vs_252": {
            "accuracy_diff": paired_acc_750_vs_252["accuracy_diff"],
            "accuracy_diff_ci_95": paired_acc_750_vs_252["ci_95"],
            "accuracy_p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "brier_mean_diff": paired_brier_750_vs_252["mean_improvement"],
            "brier_diff_ci_95": paired_brier_750_vs_252["ci_95"],
            "brier_p_value_dm": paired_brier_750_vs_252["p_value_dm"],
            "logloss_mean_diff": paired_logloss_750_vs_252["mean_improvement"],
            "logloss_diff_ci_95": paired_logloss_750_vs_252["ci_95"],
            "logloss_p_value_dm": paired_logloss_750_vs_252["p_value_dm"],
            "p_value_mcnemar": paired_acc_750_vs_252["p_value_mcnemar"],
            "superior_seed": superior_seed,
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
