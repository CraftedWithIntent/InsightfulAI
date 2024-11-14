# model_interface.py
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Defines a standard interface for InsightfulAI models, supporting synchronous and asynchronous operations."""
    
    @abstractmethod
    def fit(self, X, y):
        """Synchronously trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Synchronously predicts labels for the provided input data."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Synchronously evaluates the model on the provided test data."""
        pass

    @abstractmethod
    async def async_fit(self, X, y):
        """Asynchronously trains the model on the provided data."""
        pass

    @abstractmethod
    async def async_predict(self, X):
        """Asynchronously predicts labels for the provided input data."""
        pass

    @abstractmethod
    async def async_evaluate(self, X, y):
        """Asynchronously evaluates the model on the provided test data."""
        pass
