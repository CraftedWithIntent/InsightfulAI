# model_interface.py
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Defines a standard interface for InsightfulAI models."""
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        pass
