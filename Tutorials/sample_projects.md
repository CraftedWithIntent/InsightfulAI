# Sample Projects Using Multiple Templates in InsightfulAI

This guide provides sample projects that demonstrate how to combine multiple templates within InsightfulAI. These scenarios offer practical references for real-life applications using classification, regression, NLP, and anomaly detection templates together.

---

## Project Information

- **Project**: InsightfulAI
- **Repository**: [https://github.com/CraftedWithIntent/InsightfulAI](https://github.com/CraftedWithIntent/InsightfulAI)
- **Terminology Reference**: [Terminology Guide](../Tutorials/terminology_guide.md)

---

## Sample Project 1: Customer Insights and Churn Prediction

### Overview
This project combines **classification** and **NLP** templates to analyze customer feedback and predict churn. The **NLP template** performs sentiment analysis on customer reviews, while the **classification template** (Logistic Regression) uses customer behavior data to predict churn likelihood.

### Steps
1. **Sentiment Analysis** (NLP - Naive Bayes or SVM)
   - Process customer reviews and classify sentiments as positive, negative, or neutral.
2. **Churn Prediction** (Classification - Logistic Regression)
   - Use behavioral data (e.g., frequency of purchases, customer service interactions) to predict customer churn likelihood.

### Code Outline
```python
from Templates.NLP.naive_bayes import NaiveBayesTemplate
from Templates.Classification.logistic_regression import LogisticRegressionTemplate

# Sample data for sentiment analysis
reviews = ["Great service!", "Very disappointed with the product.", "Satisfied with the purchase."]
sentiment_labels = [1, 0, 1]  # 1 = Positive, 0 = Negative

# Initialize NLP template
nlp_model = NaiveBayesTemplate()
nlp_model.fit(reviews, sentiment_labels)
review_sentiments = nlp_model.predict(["Great value!", "Poor quality"])

# Sample data for churn prediction
customer_data = [[5, 2], [1, 10], [4, 0]]  # Example features: [purchase frequency, complaints]
churn_labels = [0, 1, 0]  # 1 = Churn, 0 = No Churn

# Initialize classification template
churn_model = LogisticRegressionTemplate()
churn_model.fit(customer_data, churn_labels)
churn_predictions = churn_model.predict([[3, 1]])

print("Review Sentiments:", review_sentiments)
print("Churn Predictions:", churn_predictions)
```

### Real-Life Application
- **Customer Retention**: By analyzing customer feedback and predicting churn, businesses can proactively address customer concerns and reduce churn rates.

---

## Sample Project 2: Sales Forecasting and Anomaly Detection

### Overview
This project combines **regression** and **anomaly detection** templates to forecast sales and detect unusual sales patterns. The **regression template** (Linear Regression) predicts future sales, while the **anomaly detection template** (Isolation Forest) identifies any unusual patterns in historical data.

### Steps
1. **Sales Forecasting** (Regression - Linear Regression)
   - Predict future sales based on historical data such as previous sales, seasonality, and promotions.
2. **Anomaly Detection** (Isolation Forest)
   - Detect any anomalies or spikes in sales that deviate from expected patterns.

### Code Outline
```python
from Templates.Regression.linear_regression import LinearRegressionTemplate
from Templates.AnomalyDetection.isolation_forest import IsolationForestTemplate

# Sample data for sales forecasting
sales_data = [[200, 10], [250, 12], [220, 11]]  # Example features: [past sales, marketing spend]
sales_labels = [210, 260, 230]  # Future sales

# Initialize regression template
regression_model = LinearRegressionTemplate()
regression_model.fit(sales_data, sales_labels)
future_sales = regression_model.predict([[230, 13]])

# Sample data for anomaly detection
historical_sales = [[200], [250], [300], [5000], [220]]  # Sales data with an outlier

# Initialize anomaly detection template
anomaly_model = IsolationForestTemplate()
anomaly_model.fit(historical_sales)
anomalies = anomaly_model.predict([[210], [5000]])

print("Future Sales Prediction:", future_sales)
print("Anomaly Detection:", anomalies)
```

### Real-Life Application
- **Sales Management**: Sales forecasting helps in inventory planning, while anomaly detection allows businesses to quickly identify unexpected sales trends or data entry errors.

---

## Sample Project 3: Fraud Detection and Financial Risk Assessment

### Overview
This project combines **classification** and **regression** templates to identify fraudulent transactions and assess financial risk. The **classification template** (Random Forest) is used for fraud detection, while the **regression template** (Ridge Regression) assesses financial risk based on customer portfolios.

### Steps
1. **Fraud Detection** (Classification - Random Forest)
   - Identify potentially fraudulent transactions based on customer transaction data.
2. **Financial Risk Assessment** (Regression - Ridge Regression)
   - Predict the risk score for financial portfolios based on various financial indicators.

### Code Outline
```python
from Templates.Classification.random_forest import RandomForestTemplate
from Templates.Regression.ridge_regression import RidgeRegressionTemplate

# Sample data for fraud detection
transaction_data = [[1000, 1], [500, 0], [7000, 1]]  # Example features: [transaction amount, location]
fraud_labels = [0, 0, 1]  # 1 = Fraud, 0 = Not Fraud

# Initialize classification template
fraud_model = RandomForestTemplate()
fraud_model.fit(transaction_data, fraud_labels)
fraud_predictions = fraud_model.predict([[2000, 0]])

# Sample data for financial risk assessment
portfolio_data = [[1.2, 0.5], [0.8, 0.7], [1.5, 0.6]]  # Example features: [volatility, beta]
risk_labels = [0.3, 0.6, 0.2]  # Risk scores

# Initialize regression template
risk_model = RidgeRegressionTemplate()
risk_model.fit(portfolio_data, risk_labels)
risk_score = risk_model.predict([[1.1, 0.6]])

print("Fraud Predictions:", fraud_predictions)
print("Risk Assessment Score:", risk_score)
```

### Real-Life Application
- **Financial Security**: Fraud detection helps in identifying unauthorized transactions, while risk assessment supports informed decision-making in portfolio management.

---

## Summary of Sample Projects

These sample projects demonstrate:
- **Template Integration**: How multiple templates can work together for real-world applications.
- **Practical Use Cases**: Scenarios like customer retention, sales management, and financial security.
- **Example Code**: Outlines for each project to guide you in combining templates for comprehensive analysis.

Use these examples as starting points for building multi-template applications with InsightfulAI.