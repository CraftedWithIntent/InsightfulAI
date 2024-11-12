"""
InsightfulAI - Random Forest Template with Sync and Async Support
=================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Your Name
Date: YYYY-MM-DD
Description: This module provides a customizable Random Forest template for binary and multi-class 
             classification tasks, with both sync and async support, retry logic, and batch processing.

Dependencies:
- scikit-learn
- numpy
- asyncio

"""

import numpy as np
import logging
import asyncio
from .retry_decorator import retry_with_backoff

class RandomForestTemplate:
    """
    Encapsulated Random Forest Template with sync and async batch support, retry logic, and customizable
    parameters for classification tasks.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, max_retries: int = 3) -> None:
        """
        Initializes the Random Forest Template.

        Parameters:
        - n_estimators (int): The number of trees in the forest. Default is 100.
        - max_depth (int): The maximum depth of each tree. Default is None (unrestricted).
        - max_retries (int): Number of retries for failed operations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.model = None
        self.scaler = None

    def initialize_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.scaler = StandardScaler()

    @retry_with_backoff
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Synchronously trains the Random Forest model with retry logic.
        """
        if self.model is None:
            self.initialize_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    @retry_with_backoff
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Synchronously predicts class labels for input data with retry logic.
        """
        if self.model is None:
            self.initialize_model()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @retry_with_backoff
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Synchronously evaluates the model on input data and true labels with retry logic.
        """
        predictions = self.predict(X)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, predictions)

    # Synchronous Batch Processing
    def fit_batch(self, X_batches: list, y_batches: list) -> None:
        """
        Synchronously trains the model on multiple data batches with retry logic.
        """
        for X, y in zip(X_batches, y_batches):
            self.fit(X, y)

    def predict_batch(self, X_batches: list) -> list:
        """
        Synchronously predicts for multiple data batches with retry logic.
        """
        predictions = []
        for X in X_batches:
            predictions.append(self.predict(X))
        return predictions

    def evaluate_batch(self, X_batches: list, y_batches: list) -> list:
        """
        Synchronously evaluates the model for multiple data batches with retry logic.
        """
        accuracies = []
        for X, y in zip(X_batches, y_batches):
            accuracies.append(self.evaluate(X, y))
        return accuracies

    # Asynchronous Batch Processing
    async def async_fit_batch(self, X_batches: list, y_batches: list) -> None:
        """
        Asynchronously trains the model on multiple data batches with retry logic.
        """
        tasks = [self._async_fit(X, y) for X, y in zip(X_batches, y_batches)]
        await asyncio.gather(*tasks)

    async def async_predict_batch(self, X_batches: list) -> list:
        """
        Asynchronously predicts for multiple data batches with retry logic.
        """
        tasks = [self._async_predict(X) for X in X_batches]
        return await asyncio.gather(*tasks)

    async def async_evaluate_batch(self, X_batches: list, y_batches: list) -> list:
        """
        Asynchronously evaluates the model for multiple data batches with retry logic.
        """
        tasks = [self._async_evaluate(X, y) for X, y in zip(X_batches, y_batches)]
        return await asyncio.gather(*tasks)

    # Helper methods for async operations with retry
    @retry_with_backoff
    async def _async_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Helper function for asynchronous training with retry logic.
        """
        if self.model is None:
            self.initialize_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    @retry_with_backoff
    async def _async_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Helper function for asynchronous prediction with retry logic.
        """
        if self.model is None:
            self.initialize_model()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @retry_with_backoff
    async def _async_evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Helper function for asynchronous evaluation with retry logic.
        """
        y_pred = await self._async_predict(X)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, y_pred)
