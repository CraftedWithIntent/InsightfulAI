"""
InsightfulAI - NLP Template with Sync, Async, and OpenTelemetry Support
=======================================================================

Project: InsightfulAI
Description: This template provides an NLP model for text classification tasks, with batch async processing,
             retry logic, and OpenTelemetry tracing.
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

    def fit(self, texts: List[str], labels: List[int]):
        """Train the model synchronously."""
        with tracer.start_as_current_span("NLPTemplate.fit") as span:
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
            logging.info("Model trained successfully.")
            span.set_attribute("custom.operation", "fit")

    def predict(self, texts: List[str]) -> List[int]:
        """Synchronously predict labels for text."""
        with tracer.start_as_current_span("NLPTemplate.predict") as span:
            X = self.vectorizer.transform(texts)
            predictions = self.model.predict(X)
            span.set_attribute("custom.operation", "predict")
            return predictions

    @retry_exponential_backoff
    async def async_fit(self, text_batches: List[List[str]], label_batches: List[List[int]]):
        """Asynchronously fit the model with multiple batches."""
        with tracer.start_as_current_span("NLPTemplate.async_fit") as span:
            tasks = [self._async_fit(texts, labels) for texts, labels in zip(text_batches, label_batches)]
            await asyncio.gather(*tasks)
            span.set_attribute("custom.operation", "async_fit")

    async def _async_fit(self, texts: List[str], labels: List[int]):
        """Helper async function to fit a single batch."""
        with tracer.start_as_current_span("NLPTemplate._async_fit"):
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)

    @retry_exponential_backoff
    async def async_predict_batch(self, text_batches: List[List[str]]) -> List[List[int]]:
        """Asynchronously predict on multiple batches."""
        with tracer.start_as_current_span("NLPTemplate.async_predict_batch") as span:
            tasks = [self._async_predict(texts) for texts in text_batches]
            results = await asyncio.gather(*tasks)
            span.set_attribute("custom.operation", "async_predict_batch")
            return results

    async def _async_predict(self, texts: List[str]) -> List[int]:
        """Helper async function to predict a single batch."""
        with tracer.start_as_current_span("NLPTemplate._async_predict"):
            X = self.vectorizer.transform(texts)
            return self.model.predict(X).tolist()
