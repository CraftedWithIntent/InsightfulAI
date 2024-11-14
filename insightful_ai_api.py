# insightful_ai_api.py
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel

class InsightfulAI:
    """
    Public API class for InsightfulAI, providing a unified interface for model training, prediction, and evaluation.

    Parameters:
    - model_type (str): The type of model to initialize, either "logistic_regression" or "random_forest".
    - **kwargs: Additional keyword arguments to customize model parameters.
    
    Usage:
    >>> api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')
    >>> api.fit(X_train, y_train)
    >>> predictions = api.predict(X_test)
    >>> accuracy = api.evaluate(X_test, y_test)
    """

    def __init__(self, model_type="logistic_regression", **kwargs):
        # Initialize the appropriate model based on model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegressionModel(**kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestModel(**kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported. Use 'logistic_regression' or 'random_forest'.")

    def fit(self, X, y):
        """
        Train the model on the given data.

        Parameters:
        - X: The feature matrix.
        - y: The target vector.

        Returns:
        - self: Allows for method chaining.
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        Predict labels for new data.

        Parameters:
        - X: The feature matrix for which predictions are to be made.

        Returns:
        - predictions: Predicted labels for the data in X.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on test data.

        Parameters:
        - X: The feature matrix for evaluation.
        - y: The true labels for the test data.

        Returns:
        - score: The evaluation score (e.g., accuracy) of the model on the test data.
        """
        return self.model.evaluate(X, y)