# logistic_regression_model.py
import numpy as np
from .model_interface import ModelInterface
from templates.logistic_regression_template import LogisticRegressionTemplate  # Import the full logistic regression class

class LogisticRegressionModel(ModelInterface):
    """API wrapper for Logistic Regression with sync and async support."""

    def __init__(self, **kwargs):
        self.model = LogisticRegressionTemplate(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return self.model.evaluate(X, y)
