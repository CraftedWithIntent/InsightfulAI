"""
InsightfulAI - Random Forest Test Suite for Telco Customer Churn Prediction
===========================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This test suite evaluates the performance of the RandomForest model in predicting customer churn
using the Telco Customer Churn dataset via the InsightfulAI public API. Each test assesses the 
model's training, prediction, and evaluation accuracy.

"""

import unittest
import pandas as pd
import numpy as np
from insightful_ai_api import InsightfulAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestRandomForestTelcoCustomerChurn(unittest.TestCase):
    def setUp(self):
        """
        Sets up the Telco Customer Churn dataset and initializes the RandomForest model via
        the InsightfulAI public API for testing.
        """
        print("\nSetting up Telco Customer Churn data and Random Forest model...")

        # Load the Telco Customer Churn dataset
        dataset_path = 'tests/datasets/kaggle/telco_customer_churn.csv'  # Adjust path if necessary
        self.data = pd.read_csv(dataset_path)
        
        # Convert 'Churn' column to binary (1 for 'Yes', 0 for 'No')
        self.data['Churn'] = self.data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Drop unnecessary columns (e.g., customerID)
        self.data = self.data.drop(columns=['customerID'])
        
        # Encode categorical columns with one-hot encoding
        self.data = pd.get_dummies(self.data, drop_first=True)
        
        # Split dataset into features (X) and target (y)
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize feature values
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()
        
        # Initialize the RandomForest model with the public API
        self.model = InsightfulAI(model_type="random_forest", n_estimators=100, max_depth=10)
        print("Setup completed.\n")

    def test_fit(self):
        """Tests that the model trains on the data without errors."""
        print("Testing model training...")
        try:
            self.model.fit(self.X_train, self.y_train)
            print("Model training completed successfully.\n")
        except Exception as e:
            self.fail(f"Model training raised an exception: {e}")

    def test_predict(self):
        """Tests that the model makes predictions after training."""
        print("Testing model prediction...")
        self.model.fit(self.X_train, self.y_train)  # Ensure model is trained before predicting
        predictions = self.model.predict(self.X_test)
        
        # Output predictions for reference
        print(f"Predictions: {predictions}")
        
        # Assert that the prediction length matches the test set length
        self.assertEqual(len(predictions), len(self.y_test), "Prediction length mismatch with test data.")
        print("Prediction test completed successfully.\n")

    def test_evaluate(self):
        """Tests the model's evaluation accuracy on test data."""
        print("Testing model evaluation...")
        self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.evaluate(self.X_test, self.y_test)
        
        # Output the accuracy result
        print(f"Evaluation accuracy: {accuracy:.2f}")
        
        # Assert that accuracy falls within the valid range [0, 1]
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")
        print("Evaluation test completed successfully.\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)