import unittest
import numpy as np
import pandas as pd
from backend.features import calculate_features, FEATURE_COLUMNS

class TestFeatureLeakage(unittest.TestCase):
    """
    Ensure strict causal feature construction with zero lookahead leakage.
    Features at session t must only depend on prices at or before session t.
    """
    
    def setUp(self):
        # Create deterministic synthetic test fixture clearly labeled as simulation
        np.random.seed(42)
        n = 150
        dates = pd.date_range("2022-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
        
        # Random walk prices
        returns = np.random.normal(0.0005, 0.012, n)
        prices = 18000.0 * np.cumprod(1.0 + returns)
        
        self.fixture_df = pd.DataFrame({
            "date": dates,
            "open": prices * 0.998,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.randint(100000, 500000, n),
            "is_simulated": 1  # Labeled simulated fixture
        })

    def test_causal_invariance_when_future_data_appended(self):
        """
        Features computed on session t MUST NOT change when future sessions t+1...t+k are appended.
        """
        # Calculate features on first 80 sessions
        df_80 = calculate_features(self.fixture_df.iloc[:80].copy())
        
        # Calculate features on first 120 sessions
        df_120 = calculate_features(self.fixture_df.iloc[:120].copy())
        
        # Verify that for all rows 0..79, every feature value is bitwise identical
        for col in FEATURE_COLUMNS:
            series_80 = df_80[col].iloc[:80]
            series_120 = df_120[col].iloc[:80]
            valid_mask = series_80.notna() & series_120.notna()
            self.assertTrue(valid_mask.sum() > 0, f"No valid values for feature {col}")
            np.testing.assert_allclose(
                series_80[valid_mask].values,
                series_120[valid_mask].values,
                rtol=1e-7, atol=1e-7,
                err_msg=f"Lookahead leakage detected: feature {col} changed when future data was added!"
            )

    def test_target_not_present_in_features(self):
        """
        Target variable (Y_t = sign(Close_{t+1} - Close_t)) must not be in the feature set.
        """
        df_feat = calculate_features(self.fixture_df.copy())
        for col in FEATURE_COLUMNS:
            self.assertNotEqual(col, "target")
            self.assertNotEqual(col, "next_close")
            self.assertNotEqual(col, "next_return")

if __name__ == "__main__":
    unittest.main()
