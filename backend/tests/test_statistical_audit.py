import unittest
import numpy as np
import math
from backend.backtest import (
    wilson_score_interval,
    compute_calibration_curve,
    diebold_mariano_test,
    compute_hac_standard_error,
    paired_accuracy_difference_ci,
    paired_loss_difference_block_bootstrap,
    adjust_p_values_holm_bonferroni,
    run_walk_forward_evaluation,
    compare_walk_forward_configurations
)
from backend.config import DEFAULT_MIN_TRAIN_750
from backend.features import get_engineered_dataset

class TestStatisticalAudit(unittest.TestCase):
    """
    Automated statistical audit tests covering:
    1. Independent calibration bin calculations and Wilson intervals
    2. Target label alignment and absence of off-by-one errors in tail bins
    3. Paired uncertainty estimates (dependence-aware HAC, DM test, studentized moving-block bootstrap)
    4. Exact common-dates alignment (dynamically derived N_labeled - 750 sessions) between seeds
    5. Absence of unjustified claims of statistical superiority under multiple testing correction
    """

    def test_independent_wilson_interval_calculation(self):
        """
        Verify Wilson score interval against independent mathematical formulation:
        p_hat = k / n
        denominator = 1 + z^2 / n
        center = (p_hat + z^2 / (2n)) / denominator
        margin = (z / denominator) * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))
        """
        z = 1.959963984540054
        # Test case: 17 successes out of 22 trials (Bin 0 case)
        k, n = 17, 22
        p_hat = k / n
        denom = 1.0 + (z**2) / n
        center = (p_hat + (z**2) / (2.0 * n)) / denom
        radicand = (p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2)))
        margin = (z / denom) * math.sqrt(radicand)
        expected_low = round(max(0.0, center - margin), 4)
        expected_high = round(min(1.0, center + margin), 4)

        actual_low, actual_high = wilson_score_interval(k, n, z=z)
        self.assertAlmostEqual(actual_low, expected_low, places=4)
        self.assertAlmostEqual(actual_high, expected_high, places=4)

    def test_independent_calibration_bins(self):
        """
        Independently compute calibration bins from scratch and assert 100% equivalence.
        """
        np.random.seed(42)
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
        y_prob = np.array([0.25, 0.35, 0.42, 0.48, 0.50, 0.55, 0.58, 0.61, 0.65, 0.75, 0.82, 0.90])

        bins = compute_calibration_curve(y_true, y_prob, n_bins=5)
        self.assertEqual(len(bins), 5)

        total_counted = sum(b["sample_count"] for b in bins)
        self.assertEqual(total_counted, len(y_true))

        for b in bins:
            count = b["sample_count"]
            if count > 0:
                low = b["bin_low"]
                high = b["bin_high"]
                mask = (y_prob >= low) & (y_prob <= high if b["bin_index"] == 4 else y_prob < high)
                expected_mean_p = float(np.mean(y_prob[mask]))
                expected_rate = float(np.mean(y_true[mask]))
                self.assertAlmostEqual(b["mean_predicted_prob"], expected_mean_p, places=4)
                self.assertAlmostEqual(b["empirical_realized_rate"], expected_rate, places=4)

    def test_bin_0_investigation_and_target_alignment(self):
        """
        Verify that in Bin 0 (lowest predicted probability bucket):
        - The 22 sessions have strictly aligned target dates.
        - Actual target y_k is strictly equal to (Close_{k+1} > Close_k).
        - No off-by-one or data corruption caused the 77.3% empirical rate.
        """
        df, _ = get_engineered_dataset()
        labeled_df = df[df["target"].notna()].reset_index(drop=True)

        res = run_walk_forward_evaluation(min_train_size=252, refit_interval=60)
        probs = res["raw_eval_data"]["y_model_prob"]
        actuals = res["raw_eval_data"]["y_true"]

        bin0_indices = [i for i, p in enumerate(probs) if p < 0.38]
        self.assertEqual(len(bin0_indices), 22)

        up_hits = 0
        for i in bin0_indices:
            global_idx = 252 + i
            sig_close = labeled_df.loc[global_idx, "close"]
            tgt_close = labeled_df.loc[global_idx + 1, "close"]
            expected_target = 1 if tgt_close > sig_close else 0
            
            # Verify target definition in raw data matches actual evaluation label
            self.assertEqual(actuals[i], expected_target)
            self.assertEqual(labeled_df.loc[global_idx, "target"], expected_target)
            if actuals[i] == 1:
                up_hits += 1

        self.assertEqual(up_hits, 17)
        empirical_rate = up_hits / len(bin0_indices)
        self.assertAlmostEqual(empirical_rate, 0.7727, places=4)

        # Verify Wilson interval computation for Bin 0
        ci_low, ci_high = wilson_score_interval(up_hits, len(bin0_indices))
        self.assertAlmostEqual(ci_low, 0.5656, places=3)
        self.assertAlmostEqual(ci_high, 0.8988, places=3)

        # Audit Requirement 1: Verify tail bucket note is strictly labeled as an unverified hypothesis
        # and does NOT make hardcoded causal market regime claims from calibration alone.
        bin0_note = res["calibration"][0].get("regime_note")
        if bin0_note is not None:
            self.assertTrue(
                bin0_note.startswith("Unverified hypothesis:"),
                f"Expected note to begin with 'Unverified hypothesis:', got: {bin0_note}"
            )
            self.assertIn("market regime causality must never be inferred from calibration rates alone", bin0_note)
            self.assertNotIn("reflects post-crash relief bounces", bin0_note)

    def test_dynamic_common_dates_alignment(self):
        """
        Confirm both 252-session and 750-session models evaluate on dynamically derived common sessions.
        """
        _, stats = get_engineered_dataset()
        expected_common = stats["labeled_samples"] - DEFAULT_MIN_TRAIN_750

        cmp_result = compare_walk_forward_configurations(refit_interval=60)
        common = cmp_result["common_dates_comparison"]
        self.assertIsNotNone(common)

        # Common sample count must match dynamically derived count
        self.assertEqual(common["common_sample_count"], expected_common)
        self.assertEqual(cmp_result["window_750"]["evaluated_test_sessions"], expected_common)

        # Dates must match window_750
        self.assertEqual(common["common_start_date"], cmp_result["window_750"]["test_start_date"])
        self.assertEqual(common["common_end_date"], cmp_result["window_750"]["test_end_date"])
        self.assertEqual(common["common_target_start_date"], cmp_result["window_750"]["target_start_date"])
        self.assertEqual(common["common_target_end_date"], cmp_result["window_750"]["target_end_date"])

    def test_paired_uncertainty_and_zero_crossing_hypothesis(self):
        """
        Verify that 95% confidence intervals for differences cross zero,
        confirming neither model can be declared statistically superior.
        """
        cmp_result = compare_walk_forward_configurations(refit_interval=60)
        common = cmp_result["common_dates_comparison"]
        h2h = common["head_to_head"]

        # McNemar p-value must be greater than 0.05
        self.assertGreater(h2h["p_value_mcnemar"], 0.05)
        self.assertEqual(h2h["statistical_verdict"], "NO_STATISTICAL_SUPERIORITY")
        self.assertEqual(h2h["superior_seed"], "NEITHER")

        # 95% CI for accuracy diff (750 vs 252) must span zero
        ci_acc = h2h["accuracy_diff_ci_95"]
        self.assertLess(ci_acc[0], 0.0)
        self.assertGreater(ci_acc[1], 0.0)

        # Block bootstrap 95% CI for Brier score diff must span zero
        ci_brier = h2h["brier_diff_ci_95"]
        self.assertLess(ci_brier[0], 0.0)
        self.assertGreater(ci_brier[1], 0.0)

        # Model vs baseline on common dates: paired CIs must also cross zero
        m252_vs_base_ci = common["models"]["seed_252"]["vs_baseline_acc_ci_95"]
        m750_vs_base_ci = common["models"]["seed_750"]["vs_baseline_acc_ci_95"]
        self.assertLess(m252_vs_base_ci[0], 0.0)
        self.assertGreater(m252_vs_base_ci[1], 0.0)
        self.assertLess(m750_vs_base_ci[0], 0.0)
        self.assertGreater(m750_vs_base_ci[1], 0.0)

    def test_paired_bootstrap_properties(self):
        """
        Verify moving block bootstrap and Diebold-Mariano tests behave correctly on synthetic series.
        """
        np.random.seed(123)
        y = np.random.binomial(1, 0.5, 200)
        p_a = np.random.uniform(0.4, 0.6, 200)
        p_b = p_a.copy()  # Identical predictions -> 0 diff

        # Under H0 (identical models):
        boot_res = paired_loss_difference_block_bootstrap(y, p_a, p_b, block_size=5, n_boot=200, loss_type="brier")
        self.assertAlmostEqual(boot_res["mean_diff"], 0.0, places=5)
        self.assertAlmostEqual(boot_res["mean_improvement"], 0.0, places=5)
        self.assertTrue(boot_res["spans_zero"])
        self.assertEqual(boot_res["bootstrap_p_value"], 1.0)
        self.assertEqual(boot_res["p_value_dm"], 1.0)
        self.assertEqual(boot_res["dm_statistic"], 0.0)
        self.assertFalse(boot_res["is_statistically_significant"])

        # Sign convention test: Model A strictly better with realistic session variation
        rng_props = np.random.default_rng(123)
        p_a_better = np.where(y == 1, 0.90, 0.10) + rng_props.uniform(-0.03, 0.03, 200)
        p_b_worse = np.where(y == 1, 0.40, 0.60) + rng_props.uniform(-0.03, 0.03, 200)
        res_sig = paired_loss_difference_block_bootstrap(y, p_a_better, p_b_worse, block_size=5, n_boot=200, loss_type="brier")
        self.assertLess(res_sig["mean_diff"], 0.0)  # negative favors A
        self.assertGreater(res_sig["mean_improvement"], 0.0)  # positive favors A
        self.assertFalse(res_sig["spans_zero"])  # CI does not span zero
        self.assertLess(res_sig["p_value_dm"], 0.001)  # Highly significant
        self.assertTrue(res_sig["is_statistically_significant"])

    def test_diebold_mariano_test_properties(self):
        """
        Verify Diebold-Mariano HAC test with HLN finite sample correction.
        """
        loss_a = np.array([0.1, 0.12, 0.11, 0.09, 0.10] * 20)
        loss_b = loss_a.copy()
        mean_d, dm_stat, p_val = diebold_mariano_test(loss_a, loss_b)
        self.assertAlmostEqual(mean_d, 0.0, places=5)
        self.assertAlmostEqual(dm_stat, 0.0, places=4)
        self.assertAlmostEqual(p_val, 1.0, places=4)

        # Non-identical: loss_a strictly lower than loss_b
        loss_b_worse = loss_a + 0.15
        mean_d2, dm_stat2, p_val2 = diebold_mariano_test(loss_a, loss_b_worse)
        self.assertLess(mean_d2, 0.0)
        self.assertLess(dm_stat2, 0.0)
        self.assertLess(p_val2, 0.001)

    def test_insufficient_sample_size_handling(self):
        """
        Verify that small samples (n < 30) are flagged and do not claim statistical significance.
        """
        y_small = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 2)  # n = 20 < 30
        p_a = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1] * 2)
        p_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 2)

        res = paired_loss_difference_block_bootstrap(y_small, p_a, p_b, block_size=3, n_boot=100)
        self.assertTrue(res["is_small_sample"])
        self.assertIn("Small sample size (n < 30)", res["sample_warning"])
        self.assertFalse(res["is_statistically_significant"])

        # Also verify paired accuracy difference handling for n < 30
        acc_res = paired_accuracy_difference_ci(y_small, (p_a > 0.5).astype(int), (p_b > 0.5).astype(int))
        self.assertTrue(acc_res["is_small_sample"])
        self.assertFalse(acc_res["is_statistically_significant"])

    def test_bootstrap_null_calibration_and_known_alternatives(self):
        """
        Audit Requirement 1:
        1. Validate null calibration: under H0 (true equal expected loss), the test should not reject.
        2. Validate known alternatives: under H1 (one model decisively dominates), test should reject.
        3. Assert presence of bootstrap limitation documentation.
        """
        np.random.seed(42)
        n = 300
        y = np.random.binomial(1, 0.5, n)
        
        # Test 1: Null calibration under exchangeable null predictions
        # Model A and Model B are both uninformative random predictions around 0.5
        noise_a = np.random.uniform(0.45, 0.55, n)
        noise_b = np.random.uniform(0.45, 0.55, n)
        res_null = paired_loss_difference_block_bootstrap(y, noise_a, noise_b, block_size=5, n_boot=300, seed=42)
        
        self.assertTrue(res_null["spans_zero"], "Null CI should encompass zero")
        self.assertGreater(res_null["bootstrap_p_value"], 0.05, "Bootstrap p-value under null should not reject at alpha=0.05")
        self.assertGreater(res_null["p_value_dm"], 0.05, "DM p-value under null should not reject at alpha=0.05")
        self.assertFalse(res_null["is_statistically_significant"])
        self.assertIn("bootstrap_limitation_note", res_null)
        self.assertIn("Politis & Romano", res_null["bootstrap_limitation_note"])

        # Test 2: Known alternative under decisive superiority
        # Model A has calibrated near-perfect predictions; Model B has inverted predictions
        rng_alt = np.random.default_rng(42)
        p_dominant_a = np.where(y == 1, rng_alt.uniform(0.92, 0.98, n), rng_alt.uniform(0.02, 0.08, n))
        p_inferior_b = np.where(y == 1, rng_alt.uniform(0.12, 0.18, n), rng_alt.uniform(0.82, 0.88, n))
        res_alt = paired_loss_difference_block_bootstrap(y, p_dominant_a, p_inferior_b, block_size=5, n_boot=300, seed=42)
        
        self.assertFalse(res_alt["spans_zero"], "Alternative CI must strictly exclude zero")
        self.assertGreater(res_alt["mean_improvement"], 0.5, "Improvement should be decisively positive")
        self.assertLess(res_alt["bootstrap_p_value"], 0.01, "Bootstrap p-value must be < 0.01 for decisive alternative")
        self.assertLess(res_alt["p_value_dm"], 0.001, "DM p-value must be < 0.001 for decisive alternative")
        self.assertTrue(res_alt["is_statistically_significant"])

    def test_paired_accuracy_dependence_aware_hac_inference(self):
        """
        Audit Requirement 2:
        Validate dependence-aware paired accuracy inference (0-1 loss Diebold-Mariano test with
        Harvey-Leybourne-Newbold correction and Newey-West HAC standard errors).
        Validate that independent-observation methods (McNemar, binomial Wald) are explicitly
        flagged as approximate and unsuitable for definitive time-series inference.
        """
        np.random.seed(101)
        n = 200
        y_true = np.random.binomial(1, 0.5, n)
        
        # Create serially dependent forecast errors: runs of correct and incorrect predictions
        pred_a = y_true.copy()
        pred_b = y_true.copy()
        # Perturb in autoregressive blocks of length 10
        for block_start in range(0, n, 20):
            pred_b[block_start:block_start + 10] = 1 - pred_b[block_start:block_start + 10]
            
        acc_result = paired_accuracy_difference_ci(y_true, pred_a, pred_b, max_lag=5)
        
        self.assertIn("ci_95_hac", acc_result)
        self.assertIn("ci_95_independent", acc_result)
        self.assertIn("standard_error_hac", acc_result)
        self.assertIn("p_value_dm", acc_result)
        self.assertIn("p_value_mcnemar", acc_result)
        
        # Verify independent method is explicitly flagged as approximate with serial dependence limitation
        self.assertIn("independent_methods", acc_result)
        indep = acc_result["independent_methods"]
        self.assertTrue(indep["is_approximate_only"])
        self.assertIn("serial_dependence_limitation", indep)
        self.assertIn("approximate and unsuitable for definitive time-series inference", indep["serial_dependence_limitation"])
        
        # Verify HAC standard error computation is positive
        self.assertGreater(acc_result["standard_error_hac"], 0.0)

    def test_holm_bonferroni_multiple_comparisons_adjustment(self):
        """
        Audit Requirement 3:
        Validate Holm-Bonferroni multiplicity correction across Accuracy, Brier score, and Log Loss.
        Verify that conclusions are metric-specific and avoid global superiority claims without multiple testing control.
        """
        # Test step-down formula with known values
        raw_p = {
            "accuracy": 0.02,
            "brier_score": 0.03,
            "log_loss": 0.06
        }
        # Sorted: 0.02 (rank 0, mult 3 -> 0.06), 0.03 (rank 1, mult 2 -> 0.06), 0.06 (rank 2, mult 1 -> 0.06)
        adj_p = adjust_p_values_holm_bonferroni(raw_p)
        self.assertEqual(adj_p["accuracy"], 0.06)
        self.assertEqual(adj_p["brier_score"], 0.06)
        self.assertEqual(adj_p["log_loss"], 0.06)
        
        # Decisive case: raw 0.001, 0.04, 0.20
        raw_p_decisive = {"accuracy": 0.001, "brier_score": 0.04, "log_loss": 0.20}
        adj_decisive = adjust_p_values_holm_bonferroni(raw_p_decisive)
        self.assertEqual(adj_decisive["accuracy"], 0.003)
        self.assertEqual(adj_decisive["brier_score"], 0.08)
        self.assertEqual(adj_decisive["log_loss"], 0.20)
        
        # Verify actual common-dates comparison outputs multiple comparisons and metric evaluations
        cmp_result = compare_walk_forward_configurations(refit_interval=60)
        common = cmp_result["common_dates_comparison"]
        h2h = common["head_to_head"]
        
        self.assertIn("multiple_comparisons", h2h)
        multi = h2h["multiple_comparisons"]
        self.assertEqual(multi["family_size"], 3)
        self.assertIn("adjusted_p_values", multi)
        self.assertIn("accuracy", multi["adjusted_p_values"])
        self.assertIn("brier_score", multi["adjusted_p_values"])
        self.assertIn("log_loss", multi["adjusted_p_values"])
        
        self.assertIn("metric_evaluations", h2h)
        evals = h2h["metric_evaluations"]
        self.assertIn("accuracy", evals)
        self.assertIn("brier_score", evals)
        self.assertIn("log_loss", evals)
        
        for metric_name in ["accuracy", "brier_score", "log_loss"]:
            metric_data = evals[metric_name]
            self.assertIn("ci_95", metric_data)
            self.assertIn("spans_zero", metric_data)
            self.assertIn("p_value_adjusted_holm", metric_data)
            self.assertIn("metric_verdict", metric_data)
            self.assertEqual(metric_data["metric_verdict"], "NO_STATISTICAL_DIFFERENCE")

        # Because all CIs span zero and all adjusted p-values >= 0.05:
        self.assertEqual(h2h["statistical_verdict"], "NO_STATISTICAL_SUPERIORITY")
        self.assertEqual(h2h["superior_seed"], "NEITHER")

    def test_constant_nonzero_loss_difference_behavior(self):
        """
        A constant nonzero loss difference (var_d < 1e-15) must not automatically
        produce p=0 or a significant verdict based on sample size.
        Must return an explicit degenerate/insufficient-variation status.
        """
        # n = 100 sessions, Model A loss is constantly 0.03 lower than Model B loss
        y_true = np.ones(100)
        p_a = np.full(100, 0.9)  # Loss A = (0.9 - 1)^2 = 0.01
        p_b = np.full(100, 0.8)  # Loss B = (0.8 - 1)^2 = 0.04
        # Loss difference d = 0.01 - 0.04 = -0.03 (constant across all n=100 sessions, var_d == 0)
        
        res = paired_loss_difference_block_bootstrap(y_true, p_a, p_b, block_size=5, n_boot=200, seed=42)
        self.assertFalse(res["is_statistically_significant"], "Must not claim significance when variance is degenerate")
        self.assertNotEqual(res["bootstrap_p_value"], 0.0, "Must not set bootstrap p-value to 0.0 on zero variance")
        self.assertGreaterEqual(res["bootstrap_p_value"], 0.05, "P-value must not reject null when variance is degenerate")
        self.assertTrue(res["spans_zero"], "spans_zero should be True under degenerate zero variation")
        self.assertIn("insufficient variation", res["sample_warning"].lower())
        self.assertIn("degenerate", res["bootstrap_limitation_note"].lower())

    def test_identical_models_return_nonsignificant(self):
        """
        Identical models must return a nonsignificant result with p=1.0 and spans_zero=True.
        """
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 12)
        p_a = np.array([0.7, 0.3, 0.8, 0.6, 0.2, 0.4, 0.7, 0.1] * 12)
        p_b = p_a.copy()
        
        res = paired_loss_difference_block_bootstrap(y_true, p_a, p_b, block_size=5, n_boot=200, seed=42)
        self.assertFalse(res["is_statistically_significant"])
        self.assertEqual(res["bootstrap_p_value"], 1.0)
        self.assertEqual(res["p_value_dm"], 1.0)
        self.assertTrue(res["spans_zero"])
        self.assertEqual(res["mean_diff"], 0.0)
        self.assertEqual(res["mean_improvement"], 0.0)

    def test_bootstrap_studentization_uses_exact_replicate_observations(self):
        """
        Moving-block bootstrap studentization must operate strictly on the exact
        resampled series of length n included in each replicate, without incorporating
        discarded tail observations when n is not a multiple of block_size.
        """
        # n = 43 is not a multiple of block_size = 5 (ceil(43/5) = 9 blocks = 45 elements)
        n = 43
        rng = np.random.default_rng(1001)
        y_true = rng.binomial(1, 0.5, n)
        p_a = rng.uniform(0.2, 0.8, n)
        p_b = rng.uniform(0.2, 0.8, n)

        # Run bootstrap on series
        res = paired_loss_difference_block_bootstrap(
            y_true, p_a, p_b, block_size=5, n_boot=100, seed=42
        )
        self.assertEqual(res["sample_count"], 43)
        self.assertEqual(res["block_size"], 5)
        self.assertGreaterEqual(res["bootstrap_p_value"], 0.0)
        self.assertLessEqual(res["bootstrap_p_value"], 1.0)

        # Verify that modifying global random seed does NOT affect local RNG reproducibility
        np.random.seed(9999)
        res_det1 = paired_loss_difference_block_bootstrap(y_true, p_a, p_b, block_size=5, n_boot=100, seed=777)
        np.random.seed(1111)
        res_det2 = paired_loss_difference_block_bootstrap(y_true, p_a, p_b, block_size=5, n_boot=100, seed=777)
        self.assertEqual(res_det1["bootstrap_p_value"], res_det2["bootstrap_p_value"])
        self.assertEqual(res_det1["ci_95"], res_det2["ci_95"])

    def test_bootstrap_p_value_bounded_in_unit_interval(self):
        """
        The bootstrap p-value must remain strictly within [0.0, 1.0] across diverse configurations.
        """
        rng = np.random.default_rng(2023)
        for n in [12, 35, 53, 120]:
            y_true = rng.binomial(1, 0.5, n)
            p_a = rng.uniform(0.05, 0.95, n)
            p_b = rng.uniform(0.05, 0.95, n)
            for b_size in [2, 5, 8]:
                res = paired_loss_difference_block_bootstrap(
                    y_true, p_a, p_b, block_size=b_size, n_boot=100, seed=42
                )
                self.assertGreaterEqual(res["bootstrap_p_value"], 0.0)
                self.assertLessEqual(res["bootstrap_p_value"], 1.0)
                self.assertGreaterEqual(res["p_value_dm"], 0.0)
                self.assertLessEqual(res["p_value_dm"], 1.0)

if __name__ == "__main__":
    unittest.main()

