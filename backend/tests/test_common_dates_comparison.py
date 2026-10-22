import unittest
from backend.backtest import (
    compare_walk_forward_configurations,
    assert_paired_comparison_alignment,
    run_walk_forward_evaluation
)

from backend.config import DEFAULT_MIN_TRAIN_750
from backend.features import get_engineered_dataset

class TestCommonDatesComparison(unittest.TestCase):
    """
    Audit Requirements 2 & 4:
    - Before every paired model comparison, assert exact equality of signal dates, target dates, and actual labels. Fail loudly if any mismatch exists.
    - Evaluate 252 and 750 seeds over the exact same common out-of-sample dates, derived dynamically from the dataset.
    """

    def test_fair_common_dates_evaluation(self):
        result = compare_walk_forward_configurations(refit_interval=60)
        self.assertIn("common_dates_comparison", result)
        common = result["common_dates_comparison"]
        self.assertIsNotNone(common)

        # Common sample count must be derived dynamically from the actual dataset:
        _, stats = get_engineered_dataset()
        expected_common = stats["labeled_samples"] - DEFAULT_MIN_TRAIN_750
        n_common = common["common_sample_count"]
        self.assertEqual(n_common, expected_common)
        
        # 750 window sessions must match common sample count
        self.assertEqual(result["window_750"]["evaluated_test_sessions"], n_common)
        
        # Common comparison start and end dates must match window_750
        self.assertEqual(common["common_start_date"], result["window_750"]["test_start_date"])
        self.assertEqual(common["common_end_date"], result["window_750"]["test_end_date"])
        
        # Baseline must be present and evaluated on these exact dates
        self.assertIn("baseline", common["models"])
        self.assertIn("accuracy", common["models"]["baseline"])
        self.assertIn("brier_score", common["models"]["baseline"])

        # Head to head differences must be explicitly reported separately
        h2h = common["head_to_head"]
        self.assertIn("accuracy_diff_750_vs_252", h2h)
        self.assertIn("accuracy_diff_ci_95", h2h)
        self.assertIn("accuracy_p_value_mcnemar", h2h)
        self.assertIn("accuracy_spans_zero", h2h)

        self.assertIn("brier_mean_diff", h2h)
        self.assertIn("brier_diff_ci_95", h2h)
        self.assertIn("brier_p_value_dm", h2h)
        self.assertIn("brier_bootstrap_p_value", h2h)
        self.assertIn("brier_spans_zero", h2h)

        self.assertIn("logloss_mean_diff", h2h)
        self.assertIn("logloss_diff_ci_95", h2h)
        self.assertIn("logloss_p_value_dm", h2h)
        self.assertIn("logloss_bootstrap_p_value", h2h)
        self.assertIn("logloss_spans_zero", h2h)

        self.assertIn("superior_seed", h2h)
        self.assertIn("statistical_verdict", h2h)
        self.assertIn("statistical_conclusion", h2h)

        # Baseline comparisons must also report separate p-values and CIs
        self.assertIn("vs_baseline_acc_ci_95", common["models"]["seed_252"])
        self.assertIn("vs_baseline_brier_ci_95", common["models"]["seed_252"])
        self.assertIn("vs_baseline_logloss_ci_95", common["models"]["seed_252"])
        self.assertIn("vs_baseline_brier_p_value_dm", common["models"]["seed_252"])

    def test_paired_alignment_assertion_passes_on_identical_series(self):
        """
        assert_paired_comparison_alignment must succeed without error when series match identically.
        """
        sig = ["2025-01-01", "2025-01-02", "2025-01-03"]
        tgt = ["2025-01-02", "2025-01-03", "2025-01-06"]
        lbl = [1, 0, 1]

        # Should complete cleanly without raising
        assert_paired_comparison_alignment(sig, sig.copy(), tgt, tgt.copy(), lbl, lbl.copy())

    def test_paired_alignment_fails_loudly_on_signal_date_mismatch(self):
        """
        Fail loudly (AssertionError) when signal dates differ at any index.
        """
        sig_a = ["2025-01-01", "2025-01-02", "2025-01-03"]
        sig_b = ["2025-01-01", "2025-01-04", "2025-01-03"]  # Mismatch at index 1
        tgt = ["2025-01-02", "2025-01-03", "2025-01-06"]
        lbl = [1, 0, 1]

        with self.assertRaises(AssertionError) as ctx:
            assert_paired_comparison_alignment(sig_a, sig_b, tgt, tgt, lbl, lbl)
        self.assertIn("signal date mismatch at index 1", str(ctx.exception))

    def test_paired_alignment_fails_loudly_on_target_date_mismatch(self):
        """
        Fail loudly (AssertionError) when target dates differ at any index.
        """
        sig = ["2025-01-01", "2025-01-02", "2025-01-03"]
        tgt_a = ["2025-01-02", "2025-01-03", "2025-01-06"]
        tgt_b = ["2025-01-02", "2025-01-05", "2025-01-06"]  # Mismatch at index 1
        lbl = [1, 0, 1]

        with self.assertRaises(AssertionError) as ctx:
            assert_paired_comparison_alignment(sig, sig, tgt_a, tgt_b, lbl, lbl)
        self.assertIn("target date mismatch at index 1", str(ctx.exception))

    def test_paired_alignment_fails_loudly_on_label_mismatch(self):
        """
        Fail loudly (AssertionError) when ground-truth labels differ at any index.
        """
        sig = ["2025-01-01", "2025-01-02", "2025-01-03"]
        tgt = ["2025-01-02", "2025-01-03", "2025-01-06"]
        lbl_a = [1, 0, 1]
        lbl_b = [1, 1, 1]  # Mismatch at index 1

        with self.assertRaises(AssertionError) as ctx:
            assert_paired_comparison_alignment(sig, sig, tgt, tgt, lbl_a, lbl_b)
        self.assertIn("label mismatch at index 1", str(ctx.exception))

    def test_paired_alignment_fails_loudly_on_length_mismatch(self):
        """
        Fail loudly (AssertionError) when array lengths differ.
        """
        sig_a = ["2025-01-01", "2025-01-02"]
        sig_b = ["2025-01-01", "2025-01-02", "2025-01-03"]
        tgt_a = ["2025-01-02", "2025-01-03"]
        tgt_b = ["2025-01-02", "2025-01-03", "2025-01-06"]
        lbl_a = [1, 0]
        lbl_b = [1, 0, 1]

        with self.assertRaises(AssertionError) as ctx:
            assert_paired_comparison_alignment(sig_a, sig_b, tgt_a, tgt_b, lbl_a, lbl_b)
        self.assertIn("length mismatch", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
