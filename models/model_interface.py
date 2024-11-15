from abc import ABC, abstractmethod
from operation_result import OperationResult
from opentelemetry import trace

class ModelInterface(ABC):
    """
    Defines a standard interface for InsightfulAI models, supporting synchronous and asynchronous operations
    with Railway Oriented Programming (ROP) principles using OperationResult for unified outcome handling.
    """
    
    @abstractmethod
    def fit(self, X, y) -> OperationResult[None]:
        """Synchronously trains the model on the provided data, returning an OperationResult."""
        pass

    @abstractmethod
    def predict(self, X) -> OperationResult:
        """Synchronously predicts labels for the provided input data, returning an OperationResult."""
        pass

    @abstractmethod
    def evaluate(self, X, y) -> OperationResult:
        """Synchronously evaluates the model on the provided test data, returning an OperationResult."""
        pass

    @abstractmethod
    async def async_fit(self, X, y) -> OperationResult:
        """Asynchronously trains the model on the provided data, returning an OperationResult."""
        pass

    @abstractmethod
    async def async_predict(self, X) -> OperationResult:
        """Asynchronously predicts labels for the provided input data, returning an OperationResult."""
        pass

    @abstractmethod
    async def async_evaluate(self, X, y) -> OperationResult:
        """Asynchronously evaluates the model on the provided test data, returning an OperationResult."""
        pass