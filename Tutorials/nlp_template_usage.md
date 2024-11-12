# NLP Template Usage Guide

This guide provides setup instructions, example code, and common use cases for using the **Naive Bayes** and **SVM** NLP templates in InsightfulAI. These templates support tasks like text classification (e.g., spam detection) and sentiment analysis.

---

## Setup Instructions

1. **Install Dependencies**: Ensure `scikit-learn`, `nltk`, and any other required libraries are installed.
   ```bash
   pip install scikit-learn nltk
   ```

2. **Download NLTK Data**: Some preprocessing steps require additional NLTK data files.
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Import the Templates**:
   - Import the `NaiveBayesTemplate` and `SVMTemplate` classes from the **Templates.NLP** module.

   ```python
   from Templates.NLP.naive_bayes import NaiveBayesTemplate
   from Templates.NLP.svm import SVMTemplate
   ```

---

## Example Code and Use Cases

### 1. Naive Bayes Template

#### Use Case: Spam Detection (Text Classification)

**Description**: Classifying emails or messages as spam or not spam based on text content.

#### Example Code
```python
from Templates.NLP.naive_bayes import NaiveBayesTemplate

# Sample documents and labels (0 = not spam, 1 = spam)
documents = ["Free money now!", "Important: Your account update", "Buy now and save big", "Hello, how are you?"]
labels = [1, 0, 1, 0]

# Initialize the Naive Bayes template
model = NaiveBayesTemplate()

# Train the model
model.fit(documents, labels)

# Predict on new documents
new_documents = ["Get free rewards now!", "Meeting at 10 AM tomorrow"]
predictions = model.predict(new_documents)

# Output predictions
for doc, label in zip(new_documents, predictions):
    print(f"'{doc}' classified as: {'Spam' if label == 1 else 'Not Spam'}")
```

#### Common Setup Steps
- **Data Preparation**: Ensure documents are cleaned and tokenized as needed.
- **Model Initialization**: The Naive Bayes template can be initialized directly, using TF-IDF vectorization for optimal performance.

---

### 2. SVM Template

#### Use Case: Sentiment Analysis (Text Classification)

**Description**: Classifying the sentiment of customer reviews or social media posts as positive, neutral, or negative.

#### Example Code
```python
from Templates.NLP.svm import SVMTemplate

# Sample documents and labels (0 = negative, 1 = positive)
documents = ["I love this product!", "Terrible experience, would not recommend", "Absolutely fantastic!", "Not worth the price"]
labels = [1, 0, 1, 0]

# Initialize the SVM template
model = SVMTemplate(kernel='linear', C=1.0)

# Train the model
model.fit(documents, labels)

# Predict on new documents
new_documents = ["This is the best service I've used", "Awful support, very disappointed"]
predictions = model.predict(new_documents)

# Output predictions
for doc, label in zip(new_documents, predictions):
    print(f"'{doc}' sentiment classified as: {'Positive' if label == 1 else 'Negative'}")
```

#### Common Setup Steps
- **Feature Scaling**: Use vectorization (TF-IDF) for converting text to a numerical format.
- **Model Initialization**: Adjust the `kernel` and `C` parameters for different dataset complexities.

---

### Summary of Usage Guide

This guide covers:
- **Setup Steps**: Including dependency installation, NLTK data download, and importing templates.
- **Example Code**: For both **Naive Bayes** and **SVM** NLP templates.
- **Use Cases**: Text classification for spam detection and sentiment analysis.

Use this guide as a reference to get started with NLP tasks in InsightfulAI, allowing flexible customization and easy setup for various text-based applications.