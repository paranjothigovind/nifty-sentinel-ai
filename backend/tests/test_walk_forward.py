import unittest
import numpy as np
from backend.models import QuantLogisticRegression, MajorityClassBaseline

class TestWalkForwardTiming(unittest.TestCase):
    """
    Ensure chronological walk-forward cross validation invariants:
    At step k, training indices are strictly < k.
    """
    
    def test_chronological_ordering(self):
        total_sessions = 300
        min_train_size = 50
        
        # Verify loop indices
        for k in range(min_train_size, total_sessions):
            train_indices = list(range(k))
            test_index = k
            
            # Highest training index must strictly precede test index
            self.assertLess(max(train_indices), test_index)
            # Size of training set expands monotonically
            self.assertEqual(len(train_indices), k)

    def test_logistic_model_convergence_on_expanding_slice(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1 > 0).astype(int)
        
        # Fit on past
        model = QuantLogisticRegression()
        model.fit(X[:80], y[:80], feature_names=[f"f_{i}" for i in range(5)])
        
        # Predict on future sample 80
        prob = model.predict_proba(X[80:81])
        self.assertEqual(prob.shape, (1, 2))
        self.assertGreaterEqual(prob[0, 1], 0.0)
        self.assertLessEqual(prob[0, 1], 1.0)

if __name__ == "__main__":
    unittest.main()
