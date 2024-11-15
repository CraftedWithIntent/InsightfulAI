# random_forest_model.py
import numpy as np
from .model_interface import ModelInterface
from templates.random_forest_template import RandomForestTemplate  # Import the full random forest class
from operation_result import OperationResult

class RandomForestModel(ModelInterface):
    """
    API wrapper for Random Forest with synchronous and asynchronous support using ROP.
    
    Provides sync and async methods for model training, prediction, and evaluation,
    returning OperationResult objects for consistent outcome handling.
    """

    def __init__(self, **kwargs):
        self.model = RandomForestTemplate(**kwargs)

    # Synchronous methods
    def fit(self, X: np.ndarray, y: np.ndarray) -> OperationResult[None]:
        """Synchronously train the model and return an OperationResult."""
        try:
            self.model.fit(X, y)
            return OperationResult.success(None)
        except Exception as e:
            return OperationResult.failure(e)

    def predict(self, X: np.ndarray) -> OperationResult[np.ndarray]:
        """Synchronously predict class labels and return an OperationResult."""
        try:
            predictions = self.model.predict(X)
            return OperationResult.success(predictions)
        except Exception as e:
            return OperationResult.failure(e)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> OperationResult[float]:
        """Synchronously evaluate the model's accuracy and return an OperationResult."""
        try:
            accuracy = self.model.evaluate(X, y)
            return OperationResult.success(accuracy)
        except Exception as e:
            return OperationResult.failure(e)

    # Asynchronous methods
    async def async_fit(self, X_batches: list, y_batches: list) -> OperationResult[None]:
        """Asynchronously train the model on multiple batches and return an OperationResult."""
        try:
            await self.model.async_fit(X_batches, y_batches)
            return OperationResult.success(None)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_predict(self, X_batches: list) -> OperationResult[list]:
        """Asynchronously predict class labels on multiple batches and return an OperationResult."""
        try:
            predictions = await self.model.async_predict(X_batches)
            return OperationResult.success(predictions)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_evaluate(self, X_batches: list, y_batches: list) -> OperationResult[list]:
        """Asynchronously evaluate the model's accuracy on multiple batches and return an OperationResult."""
        try:
            accuracies = await self.model.async_evaluate(X_batches, y_batches)
            return OperationResult.success(accuracies)
        except Exception as e:
            return OperationResult.failure(e)