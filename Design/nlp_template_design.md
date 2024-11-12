# NLP Template Design

This document outlines the structure, core parameters, and features for customization of the **NLP Templates** in InsightfulAI. It includes details for both **Naive Bayes** and **SVM** templates.

---

## Project Information

- **Project**: InsightfulAI
- **Repository**: [https://github.com/CraftedWithIntent/InsightfulAI](https://github.com/CraftedWithIntent/InsightfulAI)
- **Terminology Reference**: [Terminology Guide](../Tutorials/terminology_guide.md)

---

## Naive Bayes Template

### Core Parameters
- **alpha (float)**: Smoothing parameter to prevent zero probabilities for unseen words. Default is `1.0`.

### Customizable Features
1. **Vectorization**: TF-IDF vectorization for transforming text data into numerical format.
2. **Stopword Removal**: Allows for customizable removal of common stopwords to improve classification accuracy.

### Code Outline
```python
from sklearn.naive_bayes import MultinomialNB
from Templates.NLP.preprocessing import TextPreprocessor

class NaiveBayesTemplate:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)
        self.preprocessor = TextPreprocessor(use_tfidf=True)

    def fit(self, documents, labels):
        vectors = self.preprocessor.vectorize(documents)
        self.model.fit(vectors, labels)

    def predict(self, documents):
        vectors = self.preprocessor.vectorize(documents)
        return self.model.predict(vectors)
```

### Example Use Cases
- **Spam Detection**: Classifying emails or messages as spam or not spam.
- **Sentiment Analysis**: Determining sentiment (positive, negative) in customer feedback.

### Example Scenarios

```gherkin
Feature: Spam Detection
  As an email security analyst
  I want to classify emails as spam or not spam
  So that I can filter out unwanted messages

  Scenario: Classify a list of emails
    Given a list of email messages
    When I apply the Naive Bayes template for spam detection
    Then the model should label each email as "Spam" or "Not Spam"

Feature: Sentiment Analysis
  As a product manager
  I want to identify the sentiment of customer feedback
  So that I can assess customer satisfaction

  Scenario: Analyze customer feedback
    Given a dataset of customer feedback messages
    When I apply the Naive Bayes template for sentiment analysis
    Then the model should label each message as "Positive" or "Negative"
```

---

## SVM Template

### Core Parameters
- **kernel (str)**: Type of kernel used in the algorithm. Default is `'linear'`.
- **C (float)**: Regularization parameter, where smaller values specify stronger regularization. Default is `1.0`.

### Customizable Features
1. **Kernel Options**: Can be customized to different kernels (`linear`, `rbf`, etc.) for varying decision boundaries.
2. **Feature Scaling**: Uses TF-IDF vectorization for transforming text data, which improves classification performance.

### Code Outline
```python
from sklearn.svm import SVC
from Templates.NLP.preprocessing import TextPreprocessor

class SVMTemplate:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C)
        self.preprocessor = TextPreprocessor(use_tfidf=True)

    def fit(self, documents, labels):
        vectors = self.preprocessor.vectorize(documents)
        self.model.fit(vectors, labels)

    def predict(self, documents):
        vectors = self.preprocessor.vectorize(documents)
        return self.model.predict(vectors)
```

### Example Use Cases
- **Topic Categorization**: Classifying news articles or documents by topic.
- **Customer Sentiment Analysis**: Identifying sentiment (positive, negative) in social media comments.

### Example Scenarios

```gherkin
Feature: Topic Categorization
  As a content manager
  I want to categorize news articles by topic
  So that I can organize articles by their subject matter

  Scenario: Classify articles by topic
    Given a dataset of news articles
    When I apply the SVM template for topic categorization
    Then the model should label each article with a topic category

Feature: Customer Sentiment Analysis
  As a social media manager
  I want to analyze customer comments for sentiment
  So that I can gauge public opinion

  Scenario: Classify comments by sentiment
    Given a list of customer comments from social media
    When I apply the SVM template for sentiment analysis
    Then the model should label each comment as "Positive" or "Negative"
```

---

## Development and Testing

### Testing
- **Unit Tests**: Each template includes unit tests to validate model accuracy and performance with different datasets.
- **Expected Output**: Tests check for model reliability in text classification tasks, such as spam detection and topic categorization.

### Documentation and Usage
Each template includes:
- **In-code comments** to explain parameters and methods.
- **Usage documentation** to guide users in setting up and using each template with custom parameters.

---

This document serves as a foundational guide to the NLP templates, detailing structure, key parameters, and customization options for users and contributors.