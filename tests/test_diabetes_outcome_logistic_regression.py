"""
InsightfulAI - Logistic Regression Test Suite for Binary Classification with Sync and Async Support
=================================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This test suite validates the InsightfulAI public API for logistic regression models
using a binary classification dataset, ensuring the model's performance on prediction 
and evaluation. Includes both synchronous and asynchronous tests.

"""

import unittest
import pandas as pd
import numpy as np
from insightful_ai_api import InsightfulAI  # Import the main API class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import asyncio

class TestLogisticRegressionBinaryClassification(unittest.TestCase):
    """
    Unit tests for the InsightfulAI Logistic Regression model for binary classification.
    Includes both synchronous and asynchronous tests.
    """

    def setUp(self):
        """
        Sets up the binary classification test scenario with a real dataset.
        Initializes the model and prepares training and testing datasets.
        """
        print("\nSetting up binary classification test data and model instance...")

        # Load and preprocess the dataset
        dataset_path = "tests/datasets/kaggle/diabetes_outcome.csv"  # Update the path as needed
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_data(dataset_path)

        # Initialize the Logistic Regression model using the public API
        self.model_api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')
        print("Setup complete.\n")

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset for binary classification.
        
        Parameters:
        - file_path: Path to the CSV file with dataset.
        
        Returns:
        - Tuple: (X_train, X_test, y_train, y_test) arrays ready for model training and testing.
        """
        data = pd.read_csv(file_path)
        
        # Extract features (X) and target (y), assuming target is the last column in CSV
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the numerical features for optimal model performance
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def test_fit_binary_classification(self):
        """Test synchronous model training for binary classification."""
        print("Running synchronous training test for binary classification...")
        try:
            self.model_api.fit(self.X_train, self.y_train)
            print("Synchronous model training completed successfully.")
        except Exception as e:
            self.fail(f"Model training raised an exception: {e}")

    def test_predict_binary_classification(self):
        """Test synchronous model predictions for binary classification."""
        print("Running synchronous prediction test for binary classification...")
        
        # Train the model
        self.model_api.fit(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model_api.predict(self.X_test)
        
        # Print predictions for sample verification
        for i, (features, prediction) in enumerate(zip(self.X_test, predictions), start=1):
            outcome = "Positive" if prediction == 1 else "Negative"
            print(f"Sample {i}: Features={features} | Prediction: {outcome}")

        self.assertEqual(len(predictions), len(self.X_test), "Prediction length does not match test data length.")

    def test_evaluate_binary_classification(self):
        """Test synchronous model evaluation for binary classification."""
        print("Running synchronous evaluation test for binary classification...")

        # Train and evaluate
        self.model_api.fit(self.X_train, self.y_train)
        accuracy = self.model_api.evaluate(self.X_test, self.y_test)
        print(f"Synchronous evaluation accuracy: {accuracy:.2f}")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")

class TestAsyncLogisticRegressionBinaryClassification(unittest.IsolatedAsyncioTestCase):
    """
    Asynchronous unit tests for the InsightfulAI Logistic Regression model.
    """

    async def asyncSetUp(self):
        """
        Asynchronously sets up the binary classification test scenario with real dataset.
        """
        print("\nSetting up async test data and model instance...")

        # Load and preprocess the dataset
        dataset_path = "tests/datasets/kaggle/diabetes_outcome.csv"
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_data(dataset_path)

        # Initialize the Logistic Regression model with asynchronous support
        self.model_api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')
        print("Async setup complete.\n")

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset for binary classification.
        """
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Split and scale the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    async def test_async_fit_binary_classification(self):
        """Test asynchronous model training for binary classification."""
        print("Running asynchronous training test for binary classification...")
        try:
            await self.model_api.async_fit(self.X_train, self.y_train)
            print("Asynchronous model training completed successfully.")
        except Exception as e:
            self.fail(f"Async model training raised an exception: {e}")

    async def test_async_predict_binary_classification(self):
        """Test asynchronous model predictions for binary classification."""
        print("Running asynchronous prediction test for binary classification...")
        
        # Train asynchronously
        await self.model_api.async_fit(self.X_train, self.y_train)
        
        # Make asynchronous predictions
        predictions = await self.model_api.async_predict(self.X_test)
        
        for i, (features, prediction) in enumerate(zip(self.X_test, predictions), start=1):
            outcome = "Positive" if prediction == 1 else "Negative"
            print(f"Sample {i}: Features={features} | Prediction: {outcome}")

        self.assertEqual(len(predictions), len(self.X_test), "Async prediction length mismatch.")

    async def test_async_evaluate_binary_classification(self):
        """Test asynchronous model evaluation for binary classification."""
        print("Running asynchronous evaluation test for binary classification...")
        
        # Train asynchronously and evaluate
        await self.model_api.async_fit(self.X_train, self.y_train)
        accuracy = await self.model_api.async_evaluate(self.X_test, self.y_test)
        print(f"Asynchronous evaluation accuracy: {accuracy:.2f}")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
