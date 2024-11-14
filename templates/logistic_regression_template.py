"""
InsightfulAI - Logistic Regression Template with Sync and Async Retry Logic
===========================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13
Description: This module provides a customizable Logistic Regression template for binary and multi-class 
             classification tasks, with sync and async support, enhanced error handling, and configurable 
             retry logic to avoid duplicate processing.

Dependencies:
- scikit-learn
- numpy
- asyncio
- opentelemetry-api
- opentelemetry-sdk
- opentelemetry-instrumentation

"""

import asyncio
import logging
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from retry.retry_decorator import retry_exponential_backoff

# Configure logging
logging.basicConfig(level=logging.INFO)

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class LogisticRegressionTemplate:
    """
    Logistic Regression Template with sync and async support, enhanced error handling,
    and retry logic for failed batches with exponential backoff.
    """

    def __init__(self, C: float = 1.0, solver: str = 'lbfgs', max_iter: int = 100) -> None:
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        if solver not in ["lbfgs", "liblinear", "saga"]:
            raise ValueError(f"Unsupported solver: {solver}. Choose from 'lbfgs', 'liblinear', 'saga'.")

        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.model = LogisticRegression(C=self.C, solver=self.solver, max_iter=self.max_iter)
        self.scaler = StandardScaler()

    def validate_data(self, X: np.ndarray, y: np.ndarray = None) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array.")
            if X.shape[0] != len(y):
                raise ValueError("The number of samples in X and y must be the same.")
        if X.size == 0 or (y is not None and y.size == 0):
            raise ValueError("X and y cannot be empty.")
        if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
            raise ValueError("X and y cannot contain NaN values.")

    # Apply the retry decorator directly to each method that needs retry logic
    @retry_exponential_backoff
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Synchronously train the Logistic Regression model with feature scaling and retry logic.
        """
        self.validate_data(X, y)
        with tracer.start_as_current_span("LogisticRegressionTemplate.fit") as span:
            logging.info("Starting sync training with feature scaling...")
            self._process_single_batch_fit(X, y)
            logging.info("Training completed.")
            span.set_attribute("custom.operation", "fit")

    def _process_single_batch_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Helper function to process a single sync batch fit.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    @retry_exponential_backoff
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Synchronously predict class labels with retry logic.
        """
        self.validate_data(X)
        with tracer.start_as_current_span("LogisticRegressionTemplate.predict") as span:
            logging.info("Predicting labels (sync)...")
            result = self._process_single_batch_predict(X)
            span.set_attribute("custom.operation", "predict")
            return result

    def _process_single_batch_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Helper function to process a single sync batch prediction.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @retry_exponential_backoff
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Synchronously evaluate accuracy with retry logic.
        """
        self.validate_data(X, y_true)
        with tracer.start_as_current_span("LogisticRegressionTemplate.evaluate") as span:
            result = self._process_single_batch_evaluate(X, y_true)
            span.set_attribute("custom.operation", "evaluate")
            return result

    def _process_single_batch_evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Helper function to process a single sync batch evaluation.
        """
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    # Asynchronous Methods with Retry Logic
    @retry_exponential_backoff
    async def async_fit(self, X_batches: List[np.ndarray], y_batches: List[np.ndarray]) -> None:
        """
        Asynchronously train the Logistic Regression model in batches with retry logic.
        """
        processed_batches = set()
        for batch_idx, (X, y) in enumerate(zip(X_batches, y_batches)):
            if batch_idx in processed_batches:
                logging.info(f"Skipping already processed batch {batch_idx}.")
                continue
            self.validate_data(X, y)
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_fit") as span:
                await self._process_single_batch_fit(X, y)
                processed_batches.add(batch_idx)
                logging.info(f"Batch {batch_idx} processed successfully.")
                span.set_attribute("custom.operation", "async_fit")

    @retry_exponential_backoff
    async def async_predict(self, X_batches: List[np.ndarray]) -> List[np.ndarray]:
        """
        Asynchronously predict class labels for multiple batches with retry logic.
        """
        predictions = []
        processed_batches = set()
        for batch_idx, X in enumerate(X_batches):
            if batch_idx in processed_batches:
                logging.info(f"Skipping already processed prediction for batch {batch_idx}.")
                continue
            self.validate_data(X)
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_predict") as span:
                predictions.append(await self._process_single_batch_predict(X))
                processed_batches.add(batch_idx)
                span.set_attribute("custom.operation", "async_predict")
        return predictions

    @retry_exponential_backoff
    async def async_evaluate(self, X_batches: List[np.ndarray], y_true_batches: List[np.ndarray]) -> List[float]:
        """
        Asynchronously evaluate the accuracy on multiple batches with retry logic.
        """
        accuracies = []
        processed_batches = set()
        for batch_idx, (X, y_true) in enumerate(zip(X_batches, y_true_batches)):
            if batch_idx in processed_batches:
                logging.info(f"Skipping already processed evaluation for batch {batch_idx}.")
                continue
            self.validate_data(X, y_true)
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_evaluate") as span:
                accuracies.append(await self._process_single_batch_evaluate(X, y_true))
                processed_batches.add(batch_idx)
                span.set_attribute("custom.operation", "async_evaluate")
        return accuracies