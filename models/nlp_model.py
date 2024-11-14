"""
NLP Model - Wrapper for NLPTemplate
===================================

Provides a simple interface for NLP operations using the NLPTemplate.
"""

import numpy as np
from .model_interface import ModelInterface
from templates.nlp_template import NLPTemplate

class NLPModel(ModelInterface):
    """Wrapper class to manage NLP operations using the NLPTemplate."""

    def __init__(self, **kwargs):
        self.template = NLPTemplate(**kwargs)

    def fit(self, texts, labels):
        """Fit the model to data."""
        return self.template.fit(texts, labels)

    def predict(self, texts):
        """Predict using the model."""
        return self.template.predict(texts)

    def evaluate(self, texts, labels):
        """Evaluate the model on the provided test data."""
        return self.template.evaluate(texts, labels)

    async def async_fit(self, text_batches, label_batches):
        """Asynchronously fit the model in batches."""
        return await self.template.async_fit(text_batches, label_batches)

    async def async_predict(self, text_batches):
        """Asynchronously predict in batches using the model."""
        return await self.template.async_predict(text_batches)

    async def async_evaluate(self, text_batches, label_batches):
        """Asynchronously evaluate the model in batches."""
        return await self.template.async_evaluate(text_batches, label_batches)
