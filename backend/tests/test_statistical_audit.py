import unittest
import numpy as np
import math
from backend.backtest import (
    wilson_score_interval,
    compute_calibration_curve,
    paired_accuracy_difference_ci,
    paired_loss_difference_block_bootstrap,
    run_walk_forward_evaluation,
    compare_walk_forward_configurations
)
from backend.features import get_engineered_dataset

class TestStatisticalAudit(unittest.TestCase):
    """
    Automated statistical audit tests covering:
    1. Independent calibration bin calculations and Wilson intervals
    2. Target label alignment and absence of off-by-one errors in tail bins
    3. Paired uncertainty estimates (McNemar, Wald difference CI, moving block bootstrap)
    4. Exact common-dates alignment (434 sessions) between 252 and 750 configurations
    5. Absence of unjustified claims of statistical superiority
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

    def test_exact_434_common_dates_alignment(self):
        """
        Confirm both 252-session and 750-session models evaluate on the EXACT SAME 434 sessions.
        """
        cmp_result = compare_walk_forward_configurations(refit_interval=60)
        common = cmp_result["common_dates_comparison"]
        self.assertIsNotNone(common)

        # Common sample count must be exactly 434
        self.assertEqual(common["common_sample_count"], 434)
        self.assertEqual(cmp_result["window_750"]["evaluated_test_sessions"], 434)

        # Dates must match
        self.assertEqual(common["common_start_date"], "2025-01-16")
        self.assertEqual(common["common_end_date"], "2026-10-19")
        self.assertEqual(common["common_target_start_date"], "2025-01-17")
        self.assertEqual(common["common_target_end_date"], "2026-10-22")

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
        Verify moving block bootstrap function behaves correctly on known synthetic series.
        """
        np.random.seed(123)
        y = np.random.binomial(1, 0.5, 200)
        p_a = np.random.uniform(0.4, 0.6, 200)
        p_b = p_a.copy() # Identical predictions -> 0 diff

        boot_res = paired_loss_difference_block_bootstrap(y, p_a, p_b, block_size=5, n_boot=200, loss_type="brier")
        self.assertAlmostEqual(boot_res["mean_improvement"], 0.0, places=5)
        self.assertTrue(boot_res["spans_zero"])

if __name__ == "__main__":
    unittest.main()
