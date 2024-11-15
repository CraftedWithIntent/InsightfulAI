"""
InsightfulAI - NLP Model Test Suite with Sync and Async Support
===============================================================

Tests the NLPModel functionality using example data for text classification tasks.
This suite covers both synchronous and asynchronous methods.
"""

import unittest
import pandas as pd
import numpy as np
import asyncio
from insightful_ai_api import InsightfulAI  # Import the main API class
from operation_result import OperationResult  # Assuming OperationResult is defined in operation_result module
from opentelemetry import trace

class TestNLPModel(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the InsightfulAI NLP model for text classification tasks.
    Includes both synchronous and asynchronous tests.
    """

    def setUp(self):
        """Set up data and model for NLP testing, with OpenTelemetry tracing."""
        # Start the tracer
        tracer = trace.get_tracer(__name__)
        
        # Start a new span for the setup phase
        self.setup_span = tracer.start_as_current_span("test_setup_span")
        
        """Set up data and model for NLP testing."""
        self.texts = ["This is a good product.", "I had a terrible experience.", "Average quality."]
        self.labels = [1, 0, 1]
        self.text_batches = [self.texts, self.texts]
        self.label_batches = [self.labels, self.labels]

        # Initialize model with the InsightfulAI API
        self.model = InsightfulAI(model_type="nlp", max_features=500, C=1.0, solver="lbfgs")

    def test_fit(self):
        """Test synchronous fitting of the model."""
        self._check_assert_success(self.model.fit(self.texts, self.labels), "Synchronous fit")

    def test_predict(self):
        """Test synchronous prediction."""
        self._check_assert_failure(self.model.fit(self.texts, self.labels), "training");
        self._check_assert_success2(self.model.predict(self.texts), "Synchronous prediction")

    def test_evaluate(self):
        """Test synchronous evaluation."""
        self._check_assert_failure(self.model.fit(self.texts, self.labels), "training");

        evaluate_result = self.model.evaluate(self.texts, self.labels)
        if evaluate_result.is_success:
            accuracy = evaluate_result.result
            print("Accuracy:", accuracy.result)
            self.assertTrue(0 <= accuracy.result.result <= 1, "Accuracy should be between 0 and 1")
        else:
            print(f"Evaluation failed with error: {evaluate_result.error}")
            self.fail("Evaluation operation raised an error.")

    async def test_async_fit(self):
        """Test async batch fitting of the model."""
        self._check_assert_success(await self.model.async_fit(self.text_batches, self.label_batches), "Asynchronous fit")

    async def test_async_predict(self):
        """Test async batch prediction."""
        self._check_assert_failure(await self.model.async_fit(self.text_batches, self.label_batches), "Async training");
        self._check_assert_success2(await self.model.async_predict(self.text_batches), "Asynchronous prediction")

    async def test_async_evaluate_batch(self):
        """Test async batch evaluation."""
        self._check_assert_failure(await self.model.async_fit(self.text_batches, self.label_batches), "Async training");

        evaluate_result = await self.model.async_evaluate(self.text_batches, self.label_batches)
        if evaluate_result.is_success:
            accuracies = evaluate_result.result

            # Check that predictions is a list and not another OperationResult
            if isinstance(accuracies, OperationResult):
                accuracies = accuracies.result.result

            print("Async Batch Accuracies:", accuracies)
            for accuracy in accuracies:
                self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1")
        else:
            print(f"Async evaluation failed with error: {evaluate_result.error}")
            self.fail("Async evaluation operation raised an error.")

    def test_e2e_sync_with_rop_bind(self):
        """End-to-end test for synchronous fit, predict, and evaluate calls using ROP and bind."""
        
        result = (
            self.model.fit(self.texts, self.labels)
            .bind(lambda _: self.model.predict(self.texts))
            .bind(lambda predictions: self._validate_predictions(predictions, self.texts))
            .bind(lambda _: self.model.evaluate(self.texts, self.labels))
            .bind(self._validate_accuracy)
        )

        print("result: ", result.span)
        if not result.is_success:
            self.fail(f"Test failed with error: {result.error}")

    def _validate_predictions(self, predictions, texts):
        """Helper to validate predictions."""
        print("Sync Predictions:", predictions)
        assert len(predictions.result.result) == len(texts), "Prediction length mismatch with test data."
        return OperationResult.success(predictions)

    def _validate_accuracy(self, accuracy):
        """Helper to validate accuracy."""
        print("Sync Accuracy:", accuracy)
        assert 0 <= accuracy.result.result <= 1, "Accuracy should be between 0 and 1."
        return OperationResult.success(accuracy)


    def _check_assert_failure(self, result: OperationResult, operation_name: str = "optional"):
        if not result.is_success:
            self.fail(f"{operation_name.capitalize()} failed: {result.error}")

    def _check_assert_success(self, result: OperationResult, operation_name: str = "optional"):
        if result.is_success:
            print(f"{operation_name.capitalize()} completed successfully.")
        else:
            print(f"{operation_name.capitalize()} failed with error: {result.error}")
            self.fail(f"{operation_name.capitalize()} operation raised an error.")

    def _check_assert_success2(self, result: OperationResult, operation_name: str = "optional"):
        if result.is_success:
            results = result.result
            print(f"{operation_name.capitalize()}:", results)
        else:
            print(f"{operation_name.capitalize()} operation failed with error: {result.error}")
            self.fail(f"{operation_name.capitalize()} operation raised an error.")

if __name__ == '__main__':
    unittest.main(verbosity=2)