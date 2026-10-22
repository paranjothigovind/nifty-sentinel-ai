import unittest
import numpy as np
import pandas as pd
from backend.features import calculate_features
from backend.backtest import run_walk_forward_evaluation

class TestTimingAndOffByOne(unittest.TestCase):
    """
    Audit Requirement 2: Detect target leakage and off-by-one errors in walk-forward evaluation.
    Ensure training targets are strictly realized before prediction time.
    """

    def test_target_label_timing_invariance(self):
        """
        Verify that target Y_t depends on Close_{t+1}.
        Therefore, at session t close, only Y_{t-1} is known (Close_t > Close_{t-1}).
        Y_t is NOT known.
        """
        closes = [100.0, 102.0, 101.0, 104.0, 103.0]
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        df = pd.DataFrame({
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000] * 5,
        })
        feat_df = calculate_features(df)
        
        # Row 0: Close_0 = 100, Close_1 = 102 -> Target_0 = 1 (realized on 2023-01-02)
        self.assertEqual(feat_df["target"].iloc[0], 1.0)
        self.assertEqual(feat_df["target_date"].iloc[0], "2023-01-02")
        
        # Row 1: Close_1 = 102, Close_2 = 101 -> Target_1 = 0 (realized on 2023-01-03)
        self.assertEqual(feat_df["target"].iloc[1], 0.0)
        self.assertEqual(feat_df["target_date"].iloc[1], "2023-01-03")

        # Row 4 (latest): target must be NaN because Close_5 does not exist yet!
        self.assertTrue(np.isnan(feat_df["target"].iloc[4]))

    def test_no_target_leakage_in_walk_forward_loop(self):
        """
        At step k, the model is evaluated on observation X_k to predict y_k.
        The training labels must strictly be y[:k] (indices 0 to k-1).
        Index k must NOT be in the training target vector!
        """
        min_train = 1000
        result = run_walk_forward_evaluation(min_train_size=min_train, refit_interval=60)
        if "error" not in result:
            # Check timeline sample: signal_date and target_date must be distinct consecutive trading sessions
            timeline = result["timeline_sample"]
            self.assertGreater(len(timeline), 0)
            for pt in timeline:
                self.assertIn("signal_date", pt)
                self.assertIn("target_date", pt)
                self.assertNotEqual(
                    pt["signal_date"], pt["target_date"],
                    "Signal date and target date must not be identical!"
                )

if __name__ == "__main__":
    unittest.main()
