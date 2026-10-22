import unittest
import numpy as np
from backend.models import QuantLogisticRegression

class TestPreprocessingIsolation(unittest.TestCase):
    """
    Audit Requirement 6: Ensure that StandardScaler in QuantLogisticRegression
    is fitted on training data only, and never mutates or refits on test observations.
    """

    def test_scaler_mean_and_scale_unaffected_by_predict(self):
        np.random.seed(42)
        # Training set with mean ~ 10, std ~ 2
        X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 3))
        y_train = (X_train[:, 0] > 10.0).astype(int)

        model = QuantLogisticRegression(C=1.0)
        model.fit(X_train, y_train, feature_names=["f1", "f2", "f3"])

        # Capture fitted scaler parameters
        fitted_mean = np.copy(model.scaler.mean_)
        fitted_scale = np.copy(model.scaler.scale_)
        fitted_var = np.copy(model.scaler.var_)

        # Verify scaler captured training distribution
        np.testing.assert_allclose(fitted_mean, np.mean(X_train, axis=0), rtol=1e-5)

        # Extreme out-of-sample test batch with radically different mean ~ 500, std ~ 50
        X_test_extreme = np.random.normal(loc=500.0, scale=50.0, size=(50, 3))

        # Perform predictions multiple times
        _ = model.predict_proba(X_test_extreme)
        _ = model.predict(X_test_extreme)

        # Verify that scaler parameters remain bitwise identical to the training fit
        np.testing.assert_array_equal(
            model.scaler.mean_, fitted_mean,
            err_msg="Data leakage! Scaler mean was modified during predict_proba on test data."
        )
        np.testing.assert_array_equal(
            model.scaler.scale_, fitted_scale,
            err_msg="Data leakage! Scaler scale was modified during predict_proba on test data."
        )
        np.testing.assert_array_equal(
            model.scaler.var_, fitted_var,
            err_msg="Data leakage! Scaler var was modified during predict_proba on test data."
        )

if __name__ == "__main__":
    unittest.main()
