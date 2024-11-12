# Customization Guide: Model Tuning and Preprocessing Adjustments

This guide provides insights and best practices for customizing models across all templates in InsightfulAI. Each model in InsightfulAI—classification, regression, NLP, and anomaly detection—includes parameters and preprocessing options that you can adjust to suit your specific dataset and goals.

---

## Project Information

- **Project**: InsightfulAI
- **Repository**: [https://github.com/CraftedWithIntent/InsightfulAI](https://github.com/CraftedWithIntent/InsightfulAI)
- **Terminology Reference**: [Terminology Guide](../Tutorials/terminology_guide.md)

---

## Customizing Classification Templates

### Logistic Regression Template

- **Regularization (`C`)**: Controls the strength of regularization. Lower values indicate stronger regularization, which can prevent overfitting.
- **Solver**: Choose between solvers like `'lbfgs'`, `'liblinear'`, and `'saga'` based on dataset size and complexity.

**Best Practice**: Start with `C=1.0` and `'lbfgs'` solver for standard cases. For large datasets, consider `'saga'` for faster convergence.

### Random Forest Template

- **Number of Trees (`n_estimators`)**: More trees increase accuracy but also computational cost. Default is `100`.
- **Maximum Depth (`max_depth`)**: Limits the depth of each tree, controlling model complexity.

**Best Practice**: Set `n_estimators=100` and experiment with `max_depth` values to find the optimal balance between performance and accuracy.

---

## Customizing Regression Templates

### Linear Regression Template

- **Fit Intercept**: Specifies whether to calculate the intercept term. Set to `False` if the data is already centered around zero.
- **Normalize**: If `True`, features are normalized before model fitting, helpful for features with different scales.

**Best Practice**: Use `fit_intercept=True` for most cases. Set `normalize=True` when feature scales vary widely.

### Ridge Regression Template

- **Alpha**: Controls the regularization strength, where larger values reduce complexity. Default is `1.0`.
  
**Best Practice**: For general usage, set `alpha=1.0`. Increase `alpha` to reduce overfitting on smaller datasets or noisy data.

---

## Customizing NLP Templates

### Naive Bayes Template

- **Alpha (Smoothing)**: Prevents zero probabilities for unseen words. A higher `alpha` reduces sensitivity to rare words.
- **Vectorization (TF-IDF vs. Count Vectorizer)**: TF-IDF accounts for word importance, while Count Vectorizer uses raw counts.

**Best Practice**: Use `alpha=1.0` for balanced smoothing. Choose TF-IDF for text data with varied vocabulary, and Count Vectorizer for simpler text datasets.

### SVM Template

- **Kernel**: Controls the type of decision boundary. Common options include `'linear'`, `'poly'`, and `'rbf'`.
- **Regularization Parameter (`C`)**: Balances misclassification tolerance. Lower values increase regularization.

**Best Practice**: Set `kernel='linear'` and `C=1.0` for text classification. Experiment with `C` to adjust the margin size and generalization.

---

## Customizing Anomaly Detection Templates

### Isolation Forest Template

- **Number of Estimators (`n_estimators`)**: Controls the number of trees in the forest.
- **Contamination**: Specifies the expected proportion of outliers.

**Best Practice**: Use `n_estimators=100` for general usage. Adjust contamination based on the expected frequency of anomalies.

### Z-Score Outlier Detection

- **Threshold**: Defines the Z-score beyond which data points are considered anomalies.
  
**Best Practice**: Set threshold values between 2 and 3 for moderate detection. Adjust based on dataset specifics.

---

## General Preprocessing Adjustments

### Scaling and Normalization

- **Standard Scaling**: Use `StandardScaler` to normalize numerical features across models, particularly for regression and SVM.
- **Text Vectorization**: Use TF-IDF for text data with varied vocabulary; otherwise, Count Vectorizer may suffice.

### Feature Selection

- **Classification and Regression**: Remove features with high correlation to prevent redundancy.
- **NLP**: Use stopword removal and lemmatization to reduce noise in text data.

---

## Tips for Model Tuning

1. **Start with Default Parameters**: Begin with default parameters to establish a baseline.
2. **Use Cross-Validation**: Apply cross-validation to evaluate parameter changes and prevent overfitting.
3. **Adjust One Parameter at a Time**: Make isolated changes to understand each parameter’s effect on model performance.
4. **Experiment with Regularization**: Regularization parameters like `C` (SVM, logistic regression) and `alpha` (ridge regression, Naive Bayes) help control model complexity and improve generalization.

---

This guide provides a comprehensive approach to customizing models within InsightfulAI. By following these best practices and tuning suggestions, you can enhance model accuracy, efficiency, and applicability to a wide range of data.