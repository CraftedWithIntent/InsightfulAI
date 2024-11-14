# Introducing InsightfulAI: Public Alpha API for Simplified Machine Learning

We’re thrilled to launch **InsightfulAI**, a **Public Alpha API** designed to make classification and regression tasks easier for Python developers and data scientists. This alpha release is available on **[PyPI](https://pypi.org/project/InsightfulAI/)**, allowing you to quickly install and test it with `pip`!

InsightfulAI provides a streamlined, intuitive setup that lets you focus on solving problems rather than dealing with complex machine learning code. This is your chance to be an early adopter, giving valuable feedback to shape InsightfulAI's future.

---

## Key Features of the InsightfulAI Alpha API

- **Classification and Regression**: Includes ready-to-use logistic regression and random forest models.
- **Retry Logic**: Automatically retries failed operations to handle transient errors.
- **Customizable Parameters**: Configure hyperparameters like `C` and `solver` in logistic regression, or `n_estimators` and `max_depth` for random forests.
- **Solver Options**: Logistic regression supports popular solvers such as `'lbfgs'`, `'liblinear'`, and `'saga'`, allowing flexibility based on your dataset's size and characteristics.
- **Batch Asynchronous Processing**: Perform model training, predictions, and evaluations on batches asynchronously, which is especially useful for handling large datasets or real-time applications.
- **OpenTelemetry Support**: Track your model’s training and prediction performance with built-in OpenTelemetry tracing, simplifying monitoring and debugging.

This **Public Alpha API** provides essential tools to kickstart your machine learning projects and integrate basic monitoring.

---

## How to Install the InsightfulAI Public Alpha API

The alpha release of InsightfulAI is available on PyPI! Install it with the following command:

```bash
pip install InsightfulAI
```

This will install the alpha version of InsightfulAI, allowing you to experiment with its features and provide feedback to help us improve it.

---

## Getting Started with InsightfulAI

Here’s a quick tutorial on using InsightfulAI’s logistic regression model in your projects.

### Step 1: Import and Initialize

Import InsightfulAI from the API. Choose your model type (logistic regression or random forest), and initialize with your preferred settings:

```python
from insightful_ai_api import InsightfulAI

# Initialize the API for logistic regression with solver choice
model = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')  # Options: 'lbfgs', 'liblinear', 'saga'
```

### Step 2: Prepare Your Data

Load your dataset into numpy arrays or pandas data frames, then split it into training and test sets:

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([[...], ...])  # Features
y = np.array([...])          # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 3: Train the Model

Train your model using the `fit` method:

```python
model.fit(X_train, y_train)
print("Model training complete!")
```

### Step 4: Batch Asynchronous Predictions

Take advantage of batch asynchronous processing to make predictions on large batches efficiently:

```python
import asyncio

# Async batch prediction
async def async_predictes():
    predictions = await model.async_predict([X_test_batch_1, X_test_batch_2])
    print("Batch Predictions:", predictions)

# Run async batch prediction
asyncio.run(async_predictes())
```

### Step 5: Evaluate Model Performance

Evaluate your model accuracy using the `evaluate` function:

```python
accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

## Monitoring with OpenTelemetry

InsightfulAI includes **OpenTelemetry** for monitoring and tracking, allowing you to gain insights into your model’s performance and easily debug issues.

---

## Try the InsightfulAI Public Alpha API Today!

This **Public Alpha API** release is your chance to get hands-on with InsightfulAI and help influence its evolution. **Install InsightfulAI from PyPI**:

```bash
pip install InsightfulAI
```

Your feedback is essential—dive in, explore the features, and let us know what you think!