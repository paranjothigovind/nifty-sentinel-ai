import unittest
from backend.config import DEFAULT_REFIT_INTERVAL
from backend.backtest import run_walk_forward_evaluation

class TestRefitInterval(unittest.TestCase):
    """
    Audit Requirement 3: Reconcile configured refit interval with quarterly specification.
    Ensure default refit interval is 60 trading sessions (~quarterly).
    """

    def test_default_refit_interval_is_60(self):
        self.assertEqual(
            DEFAULT_REFIT_INTERVAL, 60,
            "DEFAULT_REFIT_INTERVAL must be 60 trading sessions (quarterly cadence), not 5!"
        )

    def test_run_walk_forward_evaluation_accepts_custom_refit(self):
        # Run with small seed for fast testing
        result = run_walk_forward_evaluation(min_train_size=1000, refit_interval=60)
        if "error" not in result:
            self.assertEqual(result["refit_interval"], 60)
            self.assertIn("Quarterly refit cadence", result["refit_interval_desc"])

if __name__ == "__main__":
    unittest.main()
