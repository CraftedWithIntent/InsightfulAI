"""
NLP Model - Wrapper for NLPTemplate
===================================

Provides a simple interface for NLP operations using the NLPTemplate.
"""
import numpy as np
from .model_interface import ModelInterface
from templates.nlp_template import NLPTemplate

class NLPModel:
    """Wrapper class to manage NLP operations using the NLPTemplate."""

    def __init__(self, **kwargs):
        self.template = NLPTemplate(**kwargs)

    def fit(self, texts, labels):
        """Fit the model to data."""
        return self.template.fit(texts, labels)

    def predict(self, texts):
        """Predict using the model."""
        return self.template.predict(texts)

    async def async_fit(self, text_batches, label_batches):
        """Async batch fitting of model."""
        return await self.template.async_fit(text_batches, label_batches)

    async def async_predict_batch(self, text_batches):
        """Async batch prediction using the model."""
        return await self.template.async_predict_batch(text_batches)
