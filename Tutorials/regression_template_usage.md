# Regression Template Usage Guide

This guide provides setup instructions, example code, and common applications for using the **Linear Regression** and **Ridge Regression** templates in InsightfulAI. These templates support various regression tasks, including predicting continuous values like sales forecasts and house prices.

---

## Setup Instructions

1. **Install Dependencies**: Ensure `scikit-learn` and any other required libraries are installed.
   ```bash
   pip install scikit-learn
   ```

2. **Import the Templates**:
   - Import the `LinearRegressionTemplate` and `RidgeRegressionTemplate` classes from the **Templates.Regression** module.

   ```python
   from Templates.Regression.linear_regression import LinearRegressionTemplate
   from Templates.Regression.ridge_regression import RidgeRegressionTemplate
   ```

---

## Example Code and Applications

### 1. Linear Regression Template

#### Application: Sales Forecasting

**Description**: Predicting future sales based on historical data.

#### Example Code
```python
from Templates.Regression.linear_regression import LinearRegressionTemplate
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data for regression
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression template
model = LinearRegressionTemplate(fit_intercept=True, normalize=True)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
r2_score = model.evaluate(X_test, y_test)
print(f"Sales Forecasting R^2 Score: {r2_score:.2f}")
```

#### Common Setup Steps
- **Data Preparation**: Ensure data is cleaned and relevant features are selected.
- **Model Initialization**: Adjust `fit_intercept` and `normalize` parameters based on dataset characteristics.

---

### 2. Ridge Regression Template

#### Application: House Price Prediction

**Description**: Estimating house prices using features like square footage, location, and number of rooms.

#### Example Code
```python
from Templates.Regression.ridge_regression import RidgeRegressionTemplate
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data for regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Ridge Regression template
model = RidgeRegressionTemplate(alpha=1.0, fit_intercept=True)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
r2_score = model.evaluate(X_test, y_test)
print(f"House Price Prediction R^2 Score: {r2_score:.2f}")
```

#### Common Setup Steps
- **Feature Scaling**: Normalize or scale features as needed for the dataset.
- **Regularization Tuning**: Adjust the `alpha` parameter for controlling regularization strength.

---

### Summary of Usage Guide

This guide covers:
- **Setup Steps**: Including dependency installation, importing templates, and initializing models.
- **Example Code**: For both **Linear Regression** and **Ridge Regression** templates.
- **Applications**: Sales forecasting and house price prediction as common regression use cases.

Use this guide to get started with regression tasks in InsightfulAI, leveraging customizable options to tailor models to specific datasets and objectives.