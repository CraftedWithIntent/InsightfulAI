"""
InsightfulAI - Public API with ROP, Telemetry, and Async Support
=================================================================

Project: InsightfulAI
Description: A unified API for machine learning operations, including model training, prediction, and evaluation, 
             using Railway Oriented Programming, OpenTelemetry, and both sync and async support.

Dependencies:
- scikit-learn
- numpy
- asyncio
- opentelemetry-api
- opentelemetry-sdk
"""

from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.nlp_model import NLPModel
from operation_result import OperationResult
from opentelemetry import trace

class InsightfulAI:
    """
    Public API class for InsightfulAI, providing a unified interface for model training, prediction, evaluation,
    and asynchronous batch processing with OpenTelemetry support.

    Parameters:
    - model_type (str): Specifies the type of model to initialize. Supported values include:
      - "logistic_regression" for logistic regression models
      - "random_forest" for random forest models
      - "nlp" for natural language processing tasks
    - **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self, model_type="logistic_regression", **kwargs):
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

    def fit(self, X, y) -> OperationResult[None]:
        """Train the model on the provided data and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("fit") as span:
            try:
                self.model.fit(X, y)
                return OperationResult.success(None, span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)

    def predict(self, X) -> OperationResult:
        """Predict labels or values for new data and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("predict") as span:
            try:
                result = self.model.predict(X)
                return OperationResult.success(result, span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)

    def evaluate(self, X, y) -> OperationResult[float]:
        """Evaluate the model on test data and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("evaluate") as span:
            try:
                score = self.model.evaluate(X, y)
                return OperationResult.success(score, span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)

    async def async_fit(self, X_batches, y_batches) -> OperationResult[None]:
        """Asynchronously train the model on multiple batches and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("async_fit") as span:
            try:
                await self.model.async_fit(X_batches, y_batches)
                return OperationResult.success("Async fit completed", span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)

    async def async_predict(self, X_batches) -> OperationResult:
        """Asynchronously predict labels or values for multiple batches and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("async_predict") as span:
            try:
                predictions = await self.model.async_predict(X_batches)
                return OperationResult.success(predictions, span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)

    async def async_evaluate(self, X_batches, y_batches) -> OperationResult:
        """Asynchronously evaluate the model on multiple batches and return an OperationResult with telemetry."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("async_evaluate") as span:
            try:
                evaluations = await self.model.async_evaluate(X_batches, y_batches)
                return OperationResult.success(evaluations, span)
            except Exception as e:
                span.record_exception(e)
                return OperationResult.failure(e, span)