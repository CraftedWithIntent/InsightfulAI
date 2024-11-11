# Classification Template Usage Guide

This guide provides setup instructions, example code, and common use cases for using the **Logistic Regression** and **Random Forest** classification templates in InsightfulAI. These templates support tasks like binary classification (e.g., customer churn prediction) and multi-class classification (e.g., product categorization).

---

## Setup Instructions

1. **Install Dependencies**: Ensure `scikit-learn` and any other required libraries are installed.
   ```bash
   pip install scikit-learn
   ```

2. **Import the Templates**:
   - Import the `LogisticRegressionTemplate` and `RandomForestTemplate` classes from the **Templates.Classification** module.

   ```python
   from Templates.Classification.logistic_regression import LogisticRegressionTemplate
   from Templates.Classification.random_forest import RandomForestTemplate
   ```

---

## Example Code and Use Cases

### 1. Logistic Regression Template

#### Use Case: Predicting Customer Churn (Binary Classification)

**Description**: Predicting whether a customer is likely to churn based on account and activity data.

#### Example Code
```python
from Templates.Classification.logistic_regression import LogisticRegressionTemplate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

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

#### Common Setup Steps
- **Data Preparation**: Ensure data is preprocessed appropriately (e.g., scaling).
- **Model Initialization**: Adjust parameters such as `C` (regularization) and `solver` based on the dataset.

---

### 2. Random Forest Template

#### Use Case: Product Categorization (Multi-class Classification)

**Description**: Categorizing products into different groups based on various product attributes.

#### Example Code
```python
from Templates.Classification.random_forest import RandomForestTemplate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data for multi-class classification
X, y = make_classification(n_samples=1000, n_features=15, n_classes=3, n_informative=10, random_state=42)

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

#### Common Setup Steps
- **Feature Selection**: Select relevant features based on the dataset.
- **Model Initialization**: Adjust `n_estimators` and `max_depth` to tune model complexity and performance.

---

### Summary of Usage Guide

This guide covers:
- **Setup Steps**: Including dependency installation, importing templates, and initializing models.
- **Example Code**: For both **Logistic Regression** and **Random Forest** templates.
- **Use Cases**: Binary classification for customer churn prediction and multi-class classification for product categorization.

Use this guide as a reference to get started with classification tasks in InsightfulAI, allowing flexible customization and easy setup.