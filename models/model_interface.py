from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from operation_result import OperationResult
from opentelemetry import trace


class ModelInterface(ABC):
    """
    Defines a standard interface for InsightfulAI models, supporting synchronous and asynchronous operations
    with Railway Oriented Programming (ROP) principles using OperationResult for unified outcome handling.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> OperationResult[None]:
        """Synchronously trains the model on the provided data, returning an OperationResult.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            
        Returns:
            OperationResult[None]: Success or failure result
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> OperationResult[np.ndarray]:
        """Synchronously predicts labels for the provided input data, returning an OperationResult.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            OperationResult[np.ndarray]: Predictions or failure result
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> OperationResult[float]:
        """Synchronously evaluates the model on the provided test data, returning an OperationResult.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            
        Returns:
            OperationResult[float]: Evaluation score or failure result
        """
        pass

    @abstractmethod
    async def async_fit(self, X_batches: List[np.ndarray], y_batches: List[np.ndarray]) -> OperationResult[None]:
        """Asynchronously trains the model on the provided data batches, returning an OperationResult.
        
        Args:
            X_batches: List of feature matrices
            y_batches: List of target vectors
            
        Returns:
            OperationResult[None]: Success or failure result
        """
        pass

    @abstractmethod
    async def async_predict(self, X_batches: List[np.ndarray]) -> OperationResult[List[np.ndarray]]:
        """Asynchronously predicts labels for the provided input data batches, returning an OperationResult.
        
        Args:
            X_batches: List of feature matrices
            
        Returns:
            OperationResult[List[np.ndarray]]: List of predictions or failure result
        """
        pass

    @abstractmethod
    async def async_evaluate(self, X_batches: List[np.ndarray], y_batches: List[np.ndarray]) -> OperationResult[List[float]]:
        """Asynchronously evaluates the model on the provided test data batches, returning an OperationResult.
        
        Args:
            X_batches: List of feature matrices
            y_batches: List of target vectors
            
        Returns:
            OperationResult[List[float]]: List of evaluation scores or failure result
        """
        pass
