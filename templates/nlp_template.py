"""
InsightfulAI - NLP Template with Sync, Async, and OpenTelemetry Support
=======================================================================

Project: InsightfulAI
Description: This template provides an NLP model for text classification tasks, with batch async processing,
             retry logic, OpenTelemetry tracing, and Railway Oriented Programming (ROP) support.
Dependencies:
- scikit-learn
- numpy
- asyncio
- opentelemetry-api
- opentelemetry-sdk
"""

import asyncio
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from retry.retry_decorator import retry_exponential_backoff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from operation_result import OperationResult  # Ensure OperationResult type is defined in your project
from typing import List

# Configure OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class NLPTemplate:
    """NLP Template for text classification tasks, with async and sync batch processing."""

    def __init__(self, max_features=1000, C=1.0, solver="lbfgs", max_retries=3):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(C=C, solver=solver)
        self.max_retries = max_retries

    def fit(self, texts: List[str], labels: List[int]) -> OperationResult[None]:
        """Train the model synchronously."""
        try:
            with tracer.start_as_current_span("NLPTemplate.fit") as span:
                X = self.vectorizer.fit_transform(texts)
                self.model.fit(X, labels)
                logging.info("Model trained successfully.")
                span.set_attribute("custom.operation", "fit")
                return OperationResult.success(None, span)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return OperationResult.failure(e, span)

    def predict(self, texts: List[str]) -> OperationResult[List[int]]:
        """Synchronously predict labels for text."""
        try:
            with tracer.start_as_current_span("NLPTemplate.predict") as span:
                X = self.vectorizer.transform(texts)
                predictions = self.model.predict(X).tolist()
                span.set_attribute("custom.operation", "predict")
                return OperationResult.success(predictions, span)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return OperationResult.failure(e, span)

    def evaluate(self, texts: List[str], labels: List[int]) -> OperationResult[float]:
        """Synchronously evaluate model accuracy."""

        try:
            with tracer.start_as_current_span("NLPTemplate.evaluate") as span:
                predictions_result = self.predict(texts)
                if not predictions_result.is_success:
                    return predictions_result
                accuracy = accuracy_score(labels, predictions_result.result)
                span.set_attribute("custom.operation", "evaluate")
                logging.info(f"Model evaluation accuracy: {accuracy:.2f}")
                return OperationResult.success(accuracy, span)
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return OperationResult.failure(e, span)

    @retry_exponential_backoff
    async def async_fit(self, text_batches: List[List[str]], label_batches: List[List[int]]) -> OperationResult[None]:
        """Asynchronously fit the model with multiple batches."""
        try:
            with tracer.start_as_current_span("NLPTemplate.async_fit") as span:
                tasks = [self._async_fit(texts, labels) for texts, labels in zip(text_batches, label_batches)]
                await asyncio.gather(*tasks)
                span.set_attribute("custom.operation", "async_fit")
                return OperationResult.success(None, span)
        except Exception as e:
            logging.error(f"Async training failed: {e}")
            return OperationResult.failure(e, span)

    async def _async_fit(self, texts: List[str], labels: List[int]) -> None:
        """Helper async function to fit a single batch."""
        with tracer.start_as_current_span("NLPTemplate._async_fit"):
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)

    @retry_exponential_backoff
    async def async_predict(self, text_batches: List[List[str]]) -> OperationResult[List[List[int]]]:
        """Asynchronously predict on multiple batches."""
        try:
            with tracer.start_as_current_span("NLPTemplate.async_predict") as span:
                tasks = [self._async_predict(texts) for texts in text_batches]
                results = await asyncio.gather(*tasks)
                span.set_attribute("custom.operation", "async_predict")
                return OperationResult.success(results, span)
        except Exception as e:
            logging.error(f"Async prediction failed: {e}")
            return OperationResult.failure(e, span)

    async def _async_predict(self, texts: List[str]) -> List[int]:
        """Helper async function to predict a single batch."""
        with tracer.start_as_current_span("NLPTemplate._async_predict"):
            X = self.vectorizer.transform(texts)
            return self.model.predict(X).tolist()

    @retry_exponential_backoff
    async def async_evaluate(self, text_batches: List[List[str]], label_batches: List[List[int]]) -> OperationResult[List[float]]:
        """Asynchronously evaluate model accuracy on multiple batches."""

        try:
            with tracer.start_as_current_span("NLPTemplate.async_evaluate") as span:
                tasks = [self._async_evaluate(texts, labels) for texts, labels in zip(text_batches, label_batches)]
                accuracies = await asyncio.gather(*tasks)
                span.set_attribute("custom.operation", "async_evaluate")
                return OperationResult.success(accuracies, span)
        except Exception as e:
            logging.error(f"Async evaluation failed: {e}")
            return OperationResult.failure(e, span)

    async def _async_evaluate(self, texts: List[str], labels: List[int]) -> float:
        """Helper async function to evaluate a single batch."""
        with tracer.start_as_current_span("NLPTemplate._async_evaluate"):
            predictions_result = await self._async_predict(texts)
            return accuracy_score(labels, predictions_result)