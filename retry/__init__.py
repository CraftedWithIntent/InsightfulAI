"""
InsightfulAI - Classification Package
=====================================

Project: InsightfulAI
Description: This package provides templates for classification models, including Logistic Regression 
             and Random Forest, with support for synchronous and asynchronous operations, batch processing, 
             customizable parameters, and retry logic for robust model training.

Modules:
- logistic_regression.py: Encapsulated Logistic Regression model for binary and multi-class classification.
- random_forest.py: Encapsulated Random Forest model with flexible hyperparameters and batch support.

Usage:
- LogisticRegressionTemplate: A customizable logistic regression model.
- RandomForestTemplate: A flexible random forest model for classification tasks.

Add additional classification templates as they are developed to expand the InsightfulAI library.
"""

from .retry_decorator import retry_exponential_backoff

__all__ = [
    "retry_exponential_backoff",
]
