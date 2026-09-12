"""
NLP Model - Wrapper for NLPTemplate
===================================

Provides a simple interface for NLP operations using the NLPTemplate.
"""

from typing import Any, List
import numpy as np
from .model_interface import ModelInterface
from templates.nlp_template import NLPTemplate
from operation_result import OperationResult

class NLPModel(ModelInterface):
    """Wrapper class to manage NLP operations using the NLPTemplate with ROP support."""

    template: NLPTemplate

    def __init__(self, **kwargs: Any) -> None:
        """Initialize NLP model.
        
        Args:
            **kwargs: Model configuration parameters
        """
        self.template = NLPTemplate(**kwargs)

    def fit(self, texts: List[str], labels: np.ndarray) -> OperationResult[None]: 
        """Synchronously fit the model to data and return an OperationResult.
        
        Args:
            texts: List of text samples
            labels: Target labels array
            
        Returns:
            OperationResult[None]: Success or exception
        """
        try:
            result = self.template.fit(texts, labels)
            return OperationResult.success(result)
        except Exception as e:
            return OperationResult.failure(e)

    def predict(self, texts: List[str]) -> OperationResult[np.ndarray]:
        """Synchronously predict using the model and return an OperationResult.
        
        Args:
            texts: List of text samples
            
        Returns:
            OperationResult[np.ndarray]: Predictions or exception
        """
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

    async def async_fit(self, text_batches: List[List[str]], label_batches: List[np.ndarray]) -> OperationResult[str]:
        """Asynchronously fit the model in batches and return an OperationResult.
        
        Args:
            text_batches: List of text batch lists
            label_batches: List of label arrays
            
        Returns:
            OperationResult[str]: Success message or exception
        """
        try:
            await self.template.async_fit(text_batches, label_batches)
            return OperationResult.success(None)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_predict(self, text_batches: List[List[str]]) -> OperationResult[List[np.ndarray]]:
        """Asynchronously predict in batches using the model and return an OperationResult.
        
        Args:
            text_batches: List of text batch lists
            
        Returns:
            OperationResult[List[np.ndarray]]: List of predictions or exception
        """
        try:
            predictions = await self.template.async_predict(text_batches)
            return OperationResult.success(predictions)
        except Exception as e:
            return OperationResult.failure(e)

    async def async_evaluate(self, text_batches: List[List[str]], label_batches: List[np.ndarray]) -> OperationResult[List[float]]:
        """Asynchronously evaluate the model in batches and return an OperationResult.
        
        Args:
            text_batches: List of text batch lists
            label_batches: List of label arrays
            
        Returns:
            OperationResult[List[float]]: List of evaluation scores or exception
        """
        try:
            accuracies = await self.template.async_evaluate(text_batches, label_batches)
            return OperationResult.success(accuracies)
        except Exception as e:
            return OperationResult.failure(e)