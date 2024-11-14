# logistic_regression_model.py
import numpy as np
from .model_interface import ModelInterface
from templates.logistic_regression_template import LogisticRegressionTemplate  # Import the full logistic regression class

class LogisticRegressionModel(ModelInterface):
    """
    API wrapper for Logistic Regression with synchronous and asynchronous support.
    
    Provides both sync and async methods for model training, prediction, and evaluation.
    """

    def __init__(self, **kwargs):
        self.model = LogisticRegressionTemplate(**kwargs)

    # Synchronous methods
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Synchronously train the model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        """Synchronously predict class labels."""
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Synchronously evaluate the model's accuracy."""
        return self.model.evaluate(X, y)

    # Asynchronous methods
    async def async_fit(self, X_batches: list, y_batches: list):
        """
        Asynchronously train the model on multiple batches.

        Parameters:
        - X_batches: List of feature batches.
        - y_batches: List of target batches.
        """
        await self.model.async_fit(X_batches, y_batches)

    async def async_predict(self, X_batches: list):
        """
        Asynchronously predict class labels on multiple batches.

        Parameters:
        - X_batches: List of feature batches.
        
        Returns:
        - List of predictions for each batch.
        """
        return await self.model.async_predict(X_batches)

    async def async_evaluate(self, X_batches: list, y_batches: list):
        """
        Asynchronously evaluate the model's accuracy on multiple batches.

        Parameters:
        - X_batches: List of feature batches.
        - y_batches: List of target batches.
        
        Returns:
        - List of accuracy scores for each batch.
        """
        return await self.model.async_evaluate(X_batches, y_batches)
