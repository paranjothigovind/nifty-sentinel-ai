import unittest
import numpy as np
from backend.backtest import wilson_score_interval, balanced_accuracy_ci, compute_calibration_curve, mcnemar_test_p_value

class TestConfidenceIntervalsAndInference(unittest.TestCase):
    """
    Audit Requirements 7 & 8: Verify Wilson score intervals, balanced accuracy CIs,
    calibration small-sample flags, and McNemar test statistical inference.
    """

    def test_wilson_interval_bounds(self):
        # 500 correct out of 1000
        low, high = wilson_score_interval(500, 1000)
        self.assertAlmostEqual((low + high) / 2.0, 0.50, places=2)
        self.assertLess(low, 0.50)
        self.assertGreater(high, 0.50)
        self.assertGreater(low, 0.0)
        self.assertLess(high, 1.0)

    def test_wilson_interval_widens_for_small_n(self):
        # 5 out of 10 vs 500 out of 1000
        low_10, high_10 = wilson_score_interval(5, 10)
        low_1000, high_1000 = wilson_score_interval(500, 1000)
        spread_10 = high_10 - low_10
        spread_1000 = high_1000 - low_1000
        self.assertGreater(spread_10, spread_1000)

    def test_calibration_small_sample_warning(self):
        # Only 15 samples in a bin
        y_true = np.ones(15)
        y_prob = np.full(15, 0.60)
        bins = compute_calibration_curve(y_true, y_prob, n_bins=5)
        
        # At least one bin has 15 samples and must flag small sample
        active_bins = [b for b in bins if b["sample_count"] > 0]
        for b in active_bins:
            if b["sample_count"] < 30:
                self.assertTrue(b["is_small_sample"])
                self.assertIsNotNone(b["sample_warning"])

    def test_mcnemar_identical_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_model = np.array([1, 0, 1, 0, 1])
        y_base = np.array([1, 0, 1, 0, 1])
        p_val = mcnemar_test_p_value(y_true, y_model, y_base)
        self.assertEqual(p_val, 1.0)

    def test_balanced_accuracy_ci_calculation(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        bal_acc, low, high = balanced_accuracy_ci(y_true, y_pred)
        self.assertGreater(bal_acc, 0.0)
        self.assertLessEqual(low, bal_acc)
        self.assertGreaterEqual(high, bal_acc)

if __name__ == "__main__":
    unittest.main()
