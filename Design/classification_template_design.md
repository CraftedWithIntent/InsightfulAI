# Classification Template Design

This document outlines the structure, key parameters, and customization options of the **Classification Templates** in the **InsightfulAI** library. It provides details for both the **Logistic Regression** and **Random Forest** templates, covering parameter options, core features, and example use cases for various classification tasks.

---

## 📦 Project Information

- **Project**: InsightfulAI
- **Repository**: [https://github.com/CraftedWithIntent/InsightfulAI](https://github.com/CraftedWithIntent/InsightfulAI)
- **Terminology Reference**: [Terminology Guide](../Tutorials/terminology_guide.md)

---

## 🔍 Logistic Regression Template

### Core Parameters
- **C (float)**: Controls the regularization strength. Smaller values specify stronger regularization. Default is `1.0`.
- **solver (str)**: Specifies the optimization algorithm for the logistic regression model. Common choices include `'lbfgs'` and `'liblinear'`.
- **max_iter (int)**: The maximum number of iterations for the solver to converge. Default is `100`.

### Customizable Features
1. **Feature Scaling**: Uses `StandardScaler` to normalize features, improving the model's performance and convergence.
2. **Regularization Options**: Adjust the `C` parameter to balance between underfitting and overfitting.
3. **Solver Choices**: Customize the `solver` parameter based on the dataset size and specific requirements.

### Code Outline
```python
class LogisticRegressionTemplate:
    def __init__(self, C=1.0, solver='lbfgs', max_iter=100):
        # Initialize model with specified parameters
        self.model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        self.scaler = StandardScaler()
```

### Example Use Cases

- **Customer Churn Prediction**: Binary classification task to predict if a customer is likely to leave.
- **Sentiment Analysis**: Classifies text data as positive, negative, or neutral based on content.

```gherkin
Feature: Customer Churn Prediction
  As a data analyst
  I want to use logistic regression to predict customer churn
  So that we can identify high-risk customers

  Scenario: Predict customer churn based on account data
    Given a dataset with customer account and activity data
    When I train a logistic regression model
    Then the model should accurately classify churned vs. active customers

Feature: Sentiment Analysis
  As a data scientist
  I want to classify text data into sentiment categories
  So that I can analyze customer feedback trends

  Scenario: Classify product reviews as positive or negative
    Given a dataset of customer reviews
    When I train the logistic regression model on labeled reviews
    Then the model should classify each review's sentiment
```

---

## 🌲 Random Forest Template

### Core Parameters
- **n_estimators (int)**: The number of trees in the forest. Default is `100`. Increasing this parameter typically improves accuracy but also increases computation.
- **max_depth (int)**: The maximum depth of each tree, which helps prevent overfitting. Default is `None`, meaning trees are expanded until all leaves are pure.
- **max_retries (int)**: Controls the number of retries for each operation, allowing retry on failures.

### Customizable Features
1. **Feature Importance**: The random forest model provides feature importance scores, helping interpret which features have the highest predictive power.
2. **Configurable Tree Depth**: Adjust the `max_depth` parameter to control the complexity and generalization of the model.
3. **Tree Count**: Set the `n_estimators` parameter to manage the number of trees, balancing accuracy and efficiency.

### Code Outline
```python
class RandomForestTemplate:
    def __init__(self, n_estimators=100, max_depth=None):
        # Initialize model with specified parameters
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.scaler = StandardScaler()
```

### Example Use Cases

- **Product Categorization**: Multi-class classification task to categorize products based on attributes.
- **Customer Segmentation**: Group customers into segments based on behavioral data for targeted marketing.

```gherkin
Feature: Product Categorization
  As a product manager
  I want to categorize products into specific groups
  So that I can organize the product catalog

  Scenario: Classify products based on their attributes
    Given a dataset of product attributes
    When I train a random forest model
    Then the model should classify each product into its respective category

Feature: Customer Segmentation
  As a marketing analyst
  I want to group customers into segments based on purchasing behavior
  So that I can target them with personalized promotions

  Scenario: Segment customers based on purchase data
    Given a dataset of customer purchasing history
    When I train a random forest model on the data
    Then the model should group customers into distinct segments
```

---

## 🔄 Batch Processing (Sync and Async)

Both templates support batch processing for handling larger datasets or processing multiple tasks in parallel. Batch processing is available in both synchronous and asynchronous formats.

### Synchronous Batch Example
```python
# Logistic Regression batch processing
from insightfulai import LogisticRegressionTemplate
import numpy as np

X_batches = [np.random.rand(20, 5) for _ in range(3)]
y_batches = [np.random.randint(0, 2, 20) for _ in range(3)]

model = LogisticRegressionTemplate(C=1.0)
model.fit_batch(X_batches, y_batches)
predictions = model.predict_batch(X_batches)
```

### Asynchronous Batch Example
```python
# Random Forest batch processing (async)
import asyncio
from insightfulai import RandomForestTemplate
import numpy as np

X_batches = [np.random.rand(20, 5) for _ in range(3)]
y_batches = [np.random.randint(0, 3, 20) for _ in range(3)]

model = RandomForestTemplate(n_estimators=100, max_depth=10)

async def async_example():
    await model.async_fit_batch(X_batches, y_batches)
    predictions = await model.async_predict_batch(X_batches)
    print("Async Batch Predictions:", predictions)

asyncio.run(async_example())
```

---

## 📋 Development and Testing

### Testing
- **Unit Tests**: The InsightfulAI library includes unit tests for each template, ensuring accuracy and stability under different configurations.
- **Expected Output**: Each template is tested for expected output accuracy, handling of edge cases, and performance across various datasets.

### Documentation and Usage
Each template includes:
- **In-code comments** for easy understanding of parameters and methods.
- **Usage documentation** to guide users in setting up and using each template with custom parameters.

---

This design document provides an overview of the features, customization options, and use cases for the **Classification Templates** in **InsightfulAI**. The library offers a flexible, customizable approach to implementing logistic regression and random forest models, with support for both sync and async batch processing.