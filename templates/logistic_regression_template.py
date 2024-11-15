"""
InsightfulAI - Logistic Regression Template with Sync and Async Retry Logic
===========================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13
Description: This module provides a customizable Logistic Regression template for binary and multi-class 
             classification tasks, with sync and async support, enhanced error handling, and configurable 
             retry logic to avoid duplicate processing using Railway Oriented Programming (ROP) principles.

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
from operation_result import OperationResult  # Ensure OperationResult type is defined in your project

# Configure logging
logging.basicConfig(level=logging.INFO)

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class LogisticRegressionTemplate:
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

    def validate_data(self, X: np.ndarray, y: np.ndarray = None) -> bool:
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
        return True

    @retry_exponential_backoff
    def fit(self, X: np.ndarray, y: np.ndarray) -> OperationResult[None]:
        """Train the Logistic Regression model with feature scaling and retry logic."""
        try:
            self.validate_data(X, y)
            with tracer.start_as_current_span("LogisticRegressionTemplate.fit") as span:
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                logging.info("Training completed.")
                return OperationResult.success(None, span)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return OperationResult.failure(e, span)

    @retry_exponential_backoff
    def predict(self, X: np.ndarray) -> OperationResult[np.ndarray]:
        """Predict class labels with retry logic."""
        try:
            self.validate_data(X)
            with tracer.start_as_current_span("LogisticRegressionTemplate.predict") as span:
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)
                return OperationResult.success(predictions, span)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return OperationResult.failure(e, span)

    @retry_exponential_backoff
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> OperationResult[float]:
        """Evaluate accuracy with retry logic."""
        try:
            self.validate_data(X, y_true)
            with tracer.start_as_current_span("LogisticRegressionTemplate.evaluate") as span:
                y_pred = self.predict(X).result  # Assuming result here is OperationResult
                accuracy = accuracy_score(y_true, y_pred)
                return OperationResult.success(accuracy, span)
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return OperationResult.failure(e, span)

    # Asynchronous Methods
    @retry_exponential_backoff
    async def async_fit(self, X_batches: List[np.ndarray], y_batches: List[np.ndarray]) -> OperationResult[None]:
        """Async batch training with retry logic."""
        try:
            processed_batches = set()
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_fit") as span:
                for batch_idx, (X, y) in enumerate(zip(X_batches, y_batches)):
                    if batch_idx in processed_batches:
                        logging.info(f"Skipping batch {batch_idx}.")
                        continue
                    self.validate_data(X, y)
                    X_scaled = self.scaler.fit_transform(X)
                    self.model.fit(X_scaled, y)
                    processed_batches.add(batch_idx)
                logging.info("Async training completed.")
                return OperationResult.success(None, span)
        except Exception as e:
            logging.error(f"Async training failed: {e}")
            return OperationResult.failure(e, span)

    @retry_exponential_backoff
    async def async_predict(self, X_batches: List[np.ndarray]) -> OperationResult[List[np.ndarray]]:
        """Async batch prediction with retry logic."""
        try:
            predictions = []
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_predict") as span:
                for X in X_batches:
                    self.validate_data(X)
                    X_scaled = self.scaler.transform(X)
                    predictions.append(self.model.predict(X_scaled))
                return OperationResult.success(predictions, span)
        except Exception as e:
            logging.error(f"Async prediction failed: {e}")
            return OperationResult.failure(e, span)

    @retry_exponential_backoff
    async def async_evaluate(self, X_batches: List[np.ndarray], y_batches: List[np.ndarray]) -> OperationResult[List[float]]:
        """Async batch evaluation with retry logic."""
        try:
            accuracies = []
            with tracer.start_as_current_span("LogisticRegressionTemplate.async_evaluate") as span:
                for X, y_true in zip(X_batches, y_batches):
                    self.validate_data(X, y_true)
                    y_pred = self.model.predict(self.scaler.transform(X))
                    accuracies.append(accuracy_score(y_true, y_pred))
                return OperationResult.success(accuracies, span)
        except Exception as e:
            logging.error(f"Async evaluation failed: {e}")
            return OperationResult.failure(e, span)