import unittest
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, balanced_accuracy_score
from backend.backtest import compute_calibration_curve

class TestProbabilityMetrics(unittest.TestCase):
    """
    Test Brier score, Log Loss, and reliability calibration calculations.
    """

    def test_brier_score_perfect_prediction(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        score = brier_score_loss(y_true, y_prob)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_brier_score_uninformative_baseline(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        # (0.5-1)^2 * 2 + (0.5-0)^2 * 2 / 4 = 0.25
        score = brier_score_loss(y_true, y_prob)
        self.assertAlmostEqual(score, 0.25, places=5)

    def test_log_loss_finite_on_extremes(self):
        y_true = np.array([1, 0])
        y_prob = np.array([0.999, 0.001])
        loss = log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15))
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.05)

    def test_calibration_curve_structure(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
        y_prob = np.linspace(0.35, 0.65, 10)
        bins = compute_calibration_curve(y_true, y_prob, n_bins=5)
        self.assertEqual(len(bins), 5)
        total_counted = sum(b["sample_count"] for b in bins)
        self.assertEqual(total_counted, 10)

if __name__ == "__main__":
    unittest.main()
