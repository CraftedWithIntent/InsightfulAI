# insightful_ai_api.py
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.nlp_model import NLPModel

class InsightfulAI:
    """
    Public API class for InsightfulAI, providing a unified interface for model training, prediction, evaluation,
    and asynchronous batch processing with OpenTelemetry support.

    Parameters:
    - model_type (str): Specifies the type of model to initialize. Supported values include:
      - "logistic_regression" for logistic regression models
      - "random_forest" for random forest models
      - "nlp" for natural language processing tasks
    - **kwargs: Additional keyword arguments for model configuration, such as:
      - For logistic regression: C (regularization strength), solver (optimization algorithm)
      - For random forest: n_estimators (number of trees), max_depth (maximum depth of trees)
      - For NLP: max_features (max vocabulary size), solver (optimization algorithm)

    Usage:
    ```
    # Initialize API with a logistic regression model
    api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')
    api.fit(X_train, y_train)
    predictions = api.predict(X_test)
    accuracy = api.evaluate(X_test, y_test)

    # Initialize API with an NLP model for text classification
    api_nlp = InsightfulAI(model_type="nlp", max_features=500, solver='lbfgs')
    api_nlp.fit(text_batches, label_batches)
    async_predictions = await api_nlp.async_predict(text_batches)
    async_accuracy = await api_nlp.async_evaluate(text_batches, label_batches)
    ```
    """

    def __init__(self, model_type="logistic_regression", **kwargs):
        # Initialize the appropriate model based on model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegressionModel(**kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestModel(**kwargs)
        elif model_type == "nlp":
            self.model = NLPModel(**kwargs)
        else:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                "Choose from 'logistic_regression', 'random_forest', or 'nlp'."
            )

    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        - X: The feature matrix (e.g., a NumPy array or list of lists).
        - y: The target labels or values (e.g., a list or array of labels).

        Returns:
        - self: Returns the instance for method chaining.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict labels or values for new data.

        Parameters:
        - X: The feature matrix for which predictions are required.

        Returns:
        - predictions: Predicted labels or values for the data in X.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on test data to return an accuracy score or other metric.

        Parameters:
        - X: The feature matrix for evaluation.
        - y: The true labels for the test data.

        Returns:
        - score: The evaluation score (e.g., accuracy for classification) of the model on the test data.
        """
        return self.model.evaluate(X, y)

    async def async_fit(self, X_batches, y_batches):
        """
        Asynchronously train the model on multiple batches of data, supporting NLP and other batch models.

        Parameters:
        - X_batches: List of feature matrices, each representing a batch of data.
        - y_batches: List of target vectors, each representing a batch of labels.

        Returns:
        - self: Returns the instance for method chaining.
        """
        await self.model.async_fit(X_batches, y_batches)
        return self

    async def async_predict(self, X_batches):
        """
        Asynchronously predict labels or values for multiple batches of data.

        Parameters:
        - X_batches: List of feature matrices, each representing a batch of data.

        Returns:
        - List of predictions for each batch of data in X_batches.
        """
        return await self.model.async_predict(X_batches)

    async def async_evaluate(self, X_batches, y_batches):
        """
        Asynchronously evaluate the model on multiple batches of data to return accuracy or another metric.

        Parameters:
        - X_batches: List of feature matrices for evaluation.
        - y_batches: List of true label vectors for each batch of data.

        Returns:
        - List of evaluation scores (e.g., accuracy) for each batch.
        """
        return await self.model.async_evaluate(X_batches, y_batches)
