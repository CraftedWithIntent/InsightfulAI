# Classification Template Usage Guide

This guide provides instructions for setting up and using the **Logistic Regression** and **Random Forest** templates in the **InsightfulAI** library. These templates support various classification tasks, including binary and multi-class classification, with synchronous and asynchronous batch processing options.

---

## 🛠 Setup Instructions

1. **Install Dependencies**: Ensure that `scikit-learn` and `numpy` are installed. You can install them along with the InsightfulAI package.

   ```bash
   pip install insightfulai
   ```

2. **Import the Templates**:
   - The **LogisticRegressionTemplate** and **RandomForestTemplate** classes are available through the main InsightfulAI package.

   ```python
   from insightfulai import LogisticRegressionTemplate, RandomForestTemplate
   ```

---

## 🚀 Usage Examples and Common Use Cases

### 1. Logistic Regression Template

#### Use Case: Predicting Customer Churn (Binary Classification)

**Description**: Use logistic regression to predict customer churn based on customer account and activity data.

#### Example Code
```python
from insightfulai import LogisticRegressionTemplate
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression template
model = LogisticRegressionTemplate(C=1.0, solver='lbfgs')

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = model.evaluate(X_test, y_test)
print(f"Customer Churn Prediction Accuracy: {accuracy * 100:.2f}%")
```

#### Common Parameters and Setup Steps
- **Parameters**:
  - `C`: Controls regularization strength. Smaller values imply stronger regularization.
  - `solver`: Algorithm used for optimization, e.g., `'lbfgs'`, `'liblinear'`.
- **Setup**: Ensure data is preprocessed (e.g., scaling) and properly split into training and testing sets.

---

### 2. Random Forest Template

#### Use Case: Product Categorization (Multi-class Classification)

**Description**: Use a random forest to categorize products into different groups based on attributes such as price, size, and color.

#### Example Code
```python
from insightfulai import RandomForestTemplate
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = np.random.rand(100, 5), np.random.randint(0, 3, 100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest template
model = RandomForestTemplate(n_estimators=100, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = model.evaluate(X_test, y_test)
print(f"Product Categorization Accuracy: {accuracy * 100:.2f}%")
```

#### Common Parameters and Setup Steps
- **Parameters**:
  - `n_estimators`: The number of trees in the forest. Higher values increase accuracy but also computation time.
  - `max_depth`: Controls the maximum depth of each tree, preventing overfitting.
- **Setup**: Select relevant features based on the dataset. Use data preprocessing if necessary.

---

## 🔄 Batch Processing (Sync and Async)

Both templates support batch processing, which is useful for handling larger datasets or making multiple predictions at once. 

### Synchronous Batch Example
```python
from insightfulai import LogisticRegressionTemplate
import numpy as np

# Generate synthetic batch data
X_batches = [np.random.rand(20, 5) for _ in range(3)]
y_batches = [np.random.randint(0, 2, 20) for _ in range(3)]

# Initialize model
model = LogisticRegressionTemplate(C=1.0)

# Synchronous batch processing
model.fit_batch(X_batches, y_batches)
predictions = model.predict_batch(X_batches)
print("Batch Predictions:", predictions)
```

### Asynchronous Batch Example
```python
import asyncio
from insightfulai import RandomForestTemplate
import numpy as np

# Generate synthetic batch data
X_batches = [np.random.rand(20, 5) for _ in range(3)]
y_batches = [np.random.randint(0, 3, 20) for _ in range(3)]

# Initialize model
model = RandomForestTemplate(n_estimators=100, max_depth=10)

# Asynchronous batch processing
async def async_example():
    await model.async_fit_batch(X_batches, y_batches)
    predictions = await model.async_predict_batch(X_batches)
    print("Async Batch Predictions:", predictions)

# Run async batch processing
asyncio.run(async_example())
```

---

## Summary of Usage Guide

This guide covers:
- **Setup Steps**: Install dependencies, import templates, and initialize models.
- **Usage Examples**: Step-by-step examples for both **Logistic Regression** and **Random Forest** templates.
- **Batch Processing**: Instructions for both sync and async batch training and predictions.

The **InsightfulAI** library provides flexible classification templates for various real-world tasks, making it easier to implement binary and multi-class classification models with minimal setup.