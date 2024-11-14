# random_forest_model.py
import numpy as np
from .model_interface import ModelInterface
from templates.random_forest_template import RandomForestTemplate  # Import the full random forest class

class RandomForestModel(ModelInterface):
    """API wrapper for Random Forest with sync and async support."""

    def __init__(self, **kwargs):
        self.model = RandomForestTemplate(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return self.model.evaluate(X, y)
