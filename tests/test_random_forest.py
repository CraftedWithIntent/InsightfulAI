"""
InsightfulAI - Random Forest Test for Telco Customer Churn Prediction
======================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This test suite validates the RandomForestTemplate for predicting customer churn using
the Telco Customer Churn dataset.

"""

import unittest
import pandas as pd
import numpy as np
from Templates.Classification.random_forest import RandomForestTemplate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestRandomForestTelcoCustomerChurn(unittest.TestCase):
    def setUp(self):
        """
        Set up the Telco Customer Churn dataset and initialize the Random Forest model.
        """
        print("\nSetting up Telco Customer Churn data and Random Forest model...")

        # Load the Telco Customer Churn dataset
        self.data = pd.read_csv('tests/datasets/kaggle/Telco-Customer-Churn.csv')  # Replace with the actual path
        
        # Prepare feature and target variables
        self.data['Churn'] = self.data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Drop unnecessary columns
        self.data = self.data.drop(columns=['customerID'])
        
        # Encode categorical columns with one-hot encoding
        self.data = pd.get_dummies(self.data, drop_first=True)
        
        # Separate features and target
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()
        
        # Initialize RandomForestTemplate with the desired parameters
        self.model = RandomForestTemplate(n_estimators=100, max_depth=10)
        print("Setup completed.\n")

    def test_fit(self):
        """Test model training without errors."""
        print("Testing model training...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed successfully.\n")

    def test_predict(self):
        """Test model prediction after training."""
        print("Testing model prediction...")
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        print(f"Predictions: {predictions}")
        self.assertEqual(len(predictions), len(self.y_test), "Prediction length mismatch.")
        print("Prediction test completed successfully.\n")

    def test_evaluate(self):
        """Test model evaluation accuracy on test data."""
        print("Testing model evaluation...")
        self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Evaluation accuracy: {accuracy:.2f}")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")
        print("Evaluation test completed successfully.\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)
