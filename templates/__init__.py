"""
InsightfulAI - Classification and NLP Package
=============================================

Project: InsightfulAI
Description:
This package offers powerful templates for classification models, such as Logistic Regression and Random Forest, 
alongside an NLP (Natural Language Processing) template for text classification. These templates support both 
synchronous and asynchronous operations, batch processing, customizable parameters, and retry logic, making model 
training and inference highly reliable and flexible.

Modules:
- logistic_regression_template.py: Implements a Logistic Regression model for binary and multi-class classification.
- random_forest_template.py: Implements a Random Forest model with flexible hyperparameters and batch processing.
- nlp_template.py: Implements an NLP model template using logistic regression for text classification tasks, 
  supporting batch async processing and OpenTelemetry tracing.

Usage:
- LogisticRegressionTemplate: A customizable logistic regression model template.
- RandomForestTemplate: A random forest model template suited for various classification tasks.
- NLPTemplate: A template for NLP tasks, specifically designed for text classification.

Additional templates for other types of classification and NLP models can be added to expand the InsightfulAI library.
"""

from .logistic_regression_template import LogisticRegressionTemplate
from .random_forest_template import RandomForestTemplate
from .nlp_template import NLPTemplate

__all__ = [
    "LogisticRegressionTemplate",
    "RandomForestTemplate",
    "NLPTemplate",
]