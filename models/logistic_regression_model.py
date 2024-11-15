"""
Logistic Regression Model Wrapper with ROP and Async Support
============================================================

Provides both synchronous and asynchronous methods for training, prediction, and evaluation
of a logistic regression model, using OperationResult for unified result handling and enhanced error reporting.
"""

import numpy as np
from .model_interface import ModelInterface
from templates.logistic_regression_template import LogisticRegressionTemplate
from operation_result import OperationResult
from opentelemetry import trace

class LogisticRegressionModel(ModelInterface):
    """
    API wrapper for Logistic Regression with synchronous and asynchronous support,
    using Railway Oriented Programming principles and OpenTelemetry for observability.
    """

    def __init__(self, **kwargs):
        self.model = LogisticRegressionTemplate(**kwargs)

    # Synchronous methods
    def fit(self, X: np.ndarray, y: np.ndarray) -> OperationResult[None]:
        """Synchronously train the model."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.fit") as span:
            result = self.model.fit(X, y)
            if result.is_success:
                return OperationResult.success("Sync fit completed", span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)

    def predict(self, X: np.ndarray) -> OperationResult[np.ndarray]:
        """Synchronously predict class labels."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.predict") as span:
            result = self.model.predict(X)
            if result.is_success:
                return OperationResult.success(result.result, span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> OperationResult[float]:
        """Synchronously evaluate the model's accuracy."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.evaluate") as span:
            result = self.model.evaluate(X, y)
            if result.is_success:
                return OperationResult.success(result.result, span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)

    # Asynchronous methods
    async def async_fit(self, X_batches: list, y_batches: list) -> OperationResult[None]:
        """Asynchronously train the model on multiple batches."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.async_fit") as span:
            result = await self.model.async_fit(X_batches, y_batches)
            if result.is_success:
                return OperationResult.success("Async fit completed", span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)

    async def async_predict(self, X_batches: list) -> OperationResult[list]:
        """Asynchronously predict class labels on multiple batches."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.async_predict") as span:
            result = await self.model.async_predict(X_batches)
            if result.is_success:
                return OperationResult.success(result.result, span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)

    async def async_evaluate(self, X_batches: list, y_batches: list) -> OperationResult[list]:
        """Asynchronously evaluate the model's accuracy on multiple batches."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("LogisticRegressionModel.async_evaluate") as span:
            result = await self.model.async_evaluate(X_batches, y_batches)
            if result.is_success:
                return OperationResult.success(result.result, span)
            span.record_exception(result.error)
            return OperationResult.failure(result.error, span)