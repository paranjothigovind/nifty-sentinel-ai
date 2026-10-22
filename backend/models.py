from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class MajorityClassBaseline:
    """
    Unconditional majority-class baseline.
    Predicts the historical empirical prior probability P(Up) of the training sample.
    """
    def __init__(self):
        self.prior_p: float = 0.5
        self.majority_class: int = 1
        
    def fit(self, y_train: np.ndarray) -> "MajorityClassBaseline":
        if len(y_train) == 0:
            self.prior_p = 0.5
            self.majority_class = 1
            return self
        self.prior_p = float(np.mean(y_train))
        # Clip slightly away from 0 and 1 for numerical stability in log loss
        self.prior_p = float(np.clip(self.prior_p, 0.001, 0.999))
        self.majority_class = 1 if self.prior_p >= 0.5 else 0
        return self
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Returns array of shape (N, 2) where col 1 is P(Y=1)."""
        n = len(X_test)
        p1 = np.full(n, self.prior_p)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(len(X_test), self.majority_class)

class QuantLogisticRegression:
    """
    L2-regularized Logistic Regression model with strict out-of-sample feature standardization.
    """
    def __init__(self, C: float = 1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=self.C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None) -> "QuantLogisticRegression":
        if feature_names:
            self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
        
    def get_feature_importances(self) -> List[Dict[str, Any]]:
        if not self.is_fitted or not self.feature_names:
            return []
        coefs = self.model.coef_[0]
        results = []
        for name, coef in zip(self.feature_names, coefs):
            results.append({
                "feature": name,
                "coefficient": round(float(coef), 4),
                "odds_ratio": round(float(np.exp(coef)), 4),
                "direction": "BULLISH" if coef > 0 else "BEARISH",
                "abs_importance": round(float(abs(coef)), 4)
            })
        return sorted(results, key=lambda x: x["abs_importance"], reverse=True)
