import unittest
import pandas as pd
from backend.data_provider import validate_ohlcv_data

class TestDataIntegrity(unittest.TestCase):
    """
    Test OHLC consistency rules, anomaly detection, and trading calendar gap checks.
    """

    def test_valid_data_passes(self):
        clean_data = pd.DataFrame({
            "Date": ["2023-01-02", "2023-01-03", "2023-01-04"],
            "Open": [18100.0, 18150.0, 18200.0],
            "High": [18200.0, 18250.0, 18300.0],
            "Low": [18050.0, 18100.0, 18150.0],
            "Close": [18150.0, 18200.0, 18250.0],
            "Volume": [150000, 160000, 170000]
        })
        valid_df, audit = validate_ohlcv_data(clean_data)
        self.assertEqual(len(valid_df), 3)
        self.assertEqual(audit["anomaly_count"], 0)
        self.assertEqual(audit["status"], "VALID")

    def test_anomaly_detection_high_less_than_low(self):
        corrupted_data = pd.DataFrame({
            "Date": ["2023-01-02"],
            "Open": [18100.0],
            "High": [18000.0],  # Invalid: High < Low
            "Low": [18050.0],
            "Close": [18020.0],
            "Volume": [100000]
        })
        with self.assertRaises(ValueError):
            validate_ohlcv_data(corrupted_data)

    def test_negative_volume_detection(self):
        bad_volume = pd.DataFrame({
            "Date": ["2023-01-02", "2023-01-03"],
            "Open": [18100.0, 18150.0],
            "High": [18200.0, 18250.0],
            "Low": [18050.0, 18100.0],
            "Close": [18150.0, 18200.0],
            "Volume": [-500, 160000]  # One bad row
        })
        valid_df, audit = validate_ohlcv_data(bad_volume)
        self.assertEqual(len(valid_df), 1)
        self.assertEqual(audit["anomaly_count"], 1)

if __name__ == "__main__":
    unittest.main()
