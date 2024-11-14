"""
InsightfulAI - NLP Model Test Suite
===================================

Tests the NLPModel functionality using example data for text classification tasks.
"""

import unittest
import pandas as pd
import numpy as np
import asyncio
from insightful_ai_api import InsightfulAI  # Import the main API class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestNLPModel(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up data and model for NLP testing."""
        self.texts = ["This is a good product.", "I had a terrible experience.", "Average quality."]
        self.labels = [1, 0, 1]
        self.text_batches = [self.texts, self.texts]
        self.label_batches = [self.labels, self.labels]
        
        # Initialize model with the InsightfulAI API
        self.model = InsightfulAI(model_type="nlp", max_features=500, C=1.0, solver="lbfgs")

    def test_fit(self):
        """Test synchronous fitting of the model."""
        self.model.fit(self.texts, self.labels)

    def test_predict(self):
        """Test synchronous prediction."""
        self.model.fit(self.texts, self.labels)
        predictions = self.model.predict(self.texts)
        print("Predictions:", predictions)

    def test_evaluate(self):
        """Test synchronous evaluation."""
        self.model.fit(self.texts, self.labels)
        accuracy = self.model.evaluate(self.texts, self.labels)
        print("Accuracy:", accuracy)
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1")

    async def test_async_fit(self):
        """Test async batch fitting of the model."""
        await self.model.async_fit(self.text_batches, self.label_batches)

    async def test_async_predict(self):
        """Test async batch prediction."""
        await self.model.async_fit(self.text_batches, self.label_batches)
        predictions = await self.model.async_predict(self.text_batches)
        print("Async Predictions:", predictions)

    async def test_async_evaluate_batch(self):
        """Test async batch evaluation."""
        await self.model.async_fit(self.text_batches, self.label_batches)
        accuracies = await self.model.async_evaluate(self.text_batches, self.label_batches)
        print("Async Batch Accuracies:", accuracies)
        for accuracy in accuracies:
            self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1")

if __name__ == '__main__':
    unittest.main(verbosity=2)
