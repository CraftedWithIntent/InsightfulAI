"""
InsightfulAI - Classification Package with Railway Oriented Programming (ROP) and OpenTelemetry Support
=======================================================================================================

Project: InsightfulAI
Description:
This package provides templates for classification models, including Logistic Regression, Random Forest,
and NLP models, with synchronous and asynchronous operations, batch processing, customizable parameters, 
and retry logic. Each model method is enhanced with Railway Oriented Programming (ROP) principles using 
OperationResult to encapsulate the outcomes and improve error handling.

Features:
- Synchronous and asynchronous support for training, prediction, and evaluation.
- Batch processing for handling large datasets efficiently.
- Customizable hyperparameters for each model.
- Consistent outcome handling with OperationResult for Railway Oriented Programming.
- Integrated OpenTelemetry support for enhanced observability and tracing.

Modules:
- logistic_regression_model.py: Encapsulated Logistic Regression model for binary and multi-class classification.
- random_forest_model.py: Flexible Random Forest model with hyperparameters and batch support.
- nlp_model.py: NLP model template for text classification with ROP and OpenTelemetry.

Usage:
- LogisticRegressionModel: A customizable logistic regression model.
- RandomForestModel: A flexible random forest model for classification tasks.
- NLPModel: An NLP model for text classification tasks.

Add additional classification templates as they are developed to expand the InsightfulAI library.
"""

from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .nlp_model import NLPModel
from .model_interface import ModelInterface

__all__ = [
    "LogisticRegressionModel",
    "RandomForestModel",
    "NLPModel",
    "ModelInterface",
]