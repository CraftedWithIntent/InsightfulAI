"""
NLP Model - Wrapper for NLPTemplate
===================================

Provides a simple interface for NLP operations using the NLPTemplate.
"""

import numpy as np
from .model_interface import ModelInterface
from templates.nlp_template import NLPTemplate
from operation_result import OperationResult

class NLPModel(ModelInterface):
    """Wrapper class to manage NLP operations using the NLPTemplate with ROP support."""

    def __init__(self, **kwargs):
        self.template = NLPTemplate(**kwargs)

    def fit(self, texts, labels) -> OperationResult[None]: 
        """Synchronously fit the model to data and return an OperationResult."""
        try:
            result = self.template.fit(texts, labels)
            return OperationResult.success(result)
        except Exception as e:
            return OperationResult.failure(e)

    def predict(self, texts) -> OperationResult:
        """Synchronously predict using the model and return an OperationResult."""
        try:
            predictions = self.template.predict(texts)
            return OperationResult.success(predictions)
        except Exception as e:
            return OperationResult.failure(e)

    def evaluate(self, texts, labels) -> OperationResult[float]:
        """Synchronously evaluate the model on the provided test data and return an OperationResult."""
        try:
            accuracy = self.template.evaluate(texts, labels)
            return OperationResult.success(accuracy)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_fit(self, text_batches, label_batches) -> OperationResult[None]:
        """Asynchronously fit the model in batches and return an OperationResult."""
        try:
            await self.template.async_fit(text_batches, label_batches)
            return OperationResult.success(None)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_predict(self, text_batches) -> OperationResult:
        """Asynchronously predict in batches using the model and return an OperationResult."""
        try:
            predictions = await self.template.async_predict(text_batches)
            return OperationResult.success(predictions)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_evaluate(self, text_batches, label_batches) -> OperationResult[list]:
        """Asynchronously evaluate the model in batches and return an OperationResult."""
        try:
            accuracies = await self.template.async_evaluate(text_batches, label_batches)
            return OperationResult.success(accuracies)
        except Exception as e:
            return OperationResult.failure(e)