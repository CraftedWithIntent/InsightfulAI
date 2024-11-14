"""
InsightfulAI - Logistic Regression Test for Binary Classification Datasets using Public API
===========================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This test suite validates the InsightfulAI public API for logistic regression models
on various binary classification datasets, ensuring the model's performance on prediction and evaluation.

"""

import unittest
import pandas as pd
import numpy as np
from insightful_ai_api import InsightfulAI  # Import the main API class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestLogisticRegressionDiabetesOutcome(unittest.TestCase):
    def setUp(self):
        """
        Set up the binary classification test scenario using a real dataset.
        Replace 'dataset_name' with your chosen dataset file path in the datasets folder.
        """
        print("\nSetting up binary classification test data and model instance...")

        # Specify the dataset file path
        dataset_path = "tests/datasets/kaggle/diabetes_outcome.csv"  # Update the file path as needed
        
        # Load and preprocess the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_data(dataset_path)

        # Initialize the Logistic Regression model using the public API
        self.model_api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset for binary classification.
        This function dynamically handles diabetes or any other CSV binary dataset with labeled target.
        """
        data = pd.read_csv(file_path)
        
        # Extract features (X) and target (y) assuming the target column is the last one in binary CSV
        X = data.iloc[:, :-1].values  # Convert features to numpy array
        y = data.iloc[:, -1].values   # Convert target to numpy array
        
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def test_fit_binary_classification(self):
        """Test model training for binary classification without errors."""
        print("Testing model training (fit) for binary classification...")
        
        # Act: Train the model using the public API
        try:
            self.model_api.fit(self.X_train, self.y_train)
            print("Model trained successfully for binary classification.")
        except Exception as e:
            self.fail(f"Model training raised an exception: {e}")
        
        # Assert: Confirm that training completes without errors
        self.assertTrue(True, "Model trained successfully.")

    def test_predict_binary_classification(self):
        """Test model prediction for binary classification."""
        print("Testing model prediction for binary classification...")
        
        # Arrange: Train the model
        self.model_api.fit(self.X_train, self.y_train)
        
        # Act: Predict outcomes for the test data using the public API
        predictions = self.model_api.predict(self.X_test)
        
        # Print each prediction result
        for i, (features, prediction) in enumerate(zip(self.X_test, predictions), 1):
            outcome = "Positive" if prediction == 1 else "Negative"
            print(f"Sample {i}: Features={features} | Prediction: {outcome}")

        # Assert: Check that predictions have the correct length
        self.assertEqual(len(predictions), len(self.X_test), "Prediction length mismatch with test data.")

    def test_evaluate_binary_classification(self):
        """Test model evaluation accuracy for binary classification."""
        print("Testing model evaluation (accuracy) for binary classification...")
        
        # Arrange: Train the model
        self.model_api.fit(self.X_train, self.y_train)
        
        # Act: Evaluate the model using the public API
        accuracy = self.model_api.evaluate(self.X_test, self.y_test)
        print(f"Binary classification model accuracy: {accuracy:.2f}")
        
        # Assert: Check that accuracy is within the valid range [0, 1]
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
