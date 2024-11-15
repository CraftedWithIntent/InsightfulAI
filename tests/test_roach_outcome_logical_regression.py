"""
InsightfulAI - Logistic Regression Test for Roach Outcome Prediction using Public API
======================================================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This test suite validates the InsightfulAI public API for logistic regression models
on the roach outcome dataset, ensuring the model's performance on prediction and evaluation.

"""

import unittest
import pandas as pd
import numpy as np
from insightful_ai_api import InsightfulAI  # Import the main API class
from operation_result import OperationResult  # Assuming OperationResult is defined in the operation_result module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestLogisticRegressionRoachOutcome(unittest.TestCase):
    def setUp(self):
        """
        Set up the binary classification test scenario using the roach outcome dataset.
        """
        print("\nSetting up roach outcome test data and model instance...")

        # Specify the dataset file path
        dataset_path = "tests/datasets/kaggle/roach_outcome.csv"  # Update the file path as needed
        
        # Load and preprocess the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_data(dataset_path)

        # Initialize the Logistic Regression model using the public API
        self.model_api = InsightfulAI(model_type="logistic_regression", C=1.0, solver='lbfgs')

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the roach outcome dataset for binary classification.
        """
        data = pd.read_csv(file_path)
        
        # Convert 'Door' column to numerical values using one-hot encoding
        data = pd.get_dummies(data, columns=['Door'], drop_first=True)

        # Extract features (X) and target (y)
        X = data.drop(columns=['Outcome']).values  # Features excluding the target column
        y = data['Outcome'].values   # Target column
        
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

        # Act: Train the model using the public API and check the result
        result = self.model_api.fit(self.X_train, self.y_train)
        if result.is_success:
            print("Model trained successfully for binary classification.")
        else:
            self.fail(f"Model training failed with error: {result.error}")
        
    def test_predict_binary_classification(self):
        """Test model prediction for binary classification."""
        print("Testing model prediction for binary classification...")
        
        # Arrange: Train the model
        fit_result = self.model_api.fit(self.X_train, self.y_train)
        if not fit_result.is_success:
            self.fail(f"Training failed: {fit_result.error}")
        
        # Act: Predict outcomes for the test data using the public API
        predict_result = self.model_api.predict(self.X_test)

        if predict_result.is_success:
            # Access the actual predictions, ensuring predict_result.result is the expected list type
            predictions = predict_result.result

            # Check that predictions is a list and not another OperationResult
            if isinstance(predictions, OperationResult):
                predictions = predictions.result

            # Now proceed with iteration
            for i, (features, prediction) in enumerate(zip(self.X_test, predictions), 1):
                outcome = "Roach Present" if prediction == 1 else "No Roach"
                print(f"Sample {i}: Features={features} | Prediction: {outcome}")
    
            # Assert: Check that predictions have the correct length
            self.assertEqual(len(predictions), len(self.X_test), "Prediction length mismatch with test data.")
        else:
            # If prediction failed, fail the test and print the error
            self.fail(f"Prediction failed with error: {predict_result.error}")


    def test_evaluate_binary_classification(self):
        """Test model evaluation accuracy for binary classification."""
        print("Testing model evaluation (accuracy) for binary classification...")
        
        # Arrange: Train the model
        fit_result = self.model_api.fit(self.X_train, self.y_train)
        if not fit_result.is_success:
            self.fail(f"Training failed: {fit_result.error}")
        
        # Act: Evaluate the model using the public API
        evaluate_result = self.model_api.evaluate(self.X_test, self.y_test)
        if evaluate_result.is_success:
            accuracy = evaluate_result.result
            print(f"Binary classification model accuracy: {accuracy:.2f}")
            
            # Assert: Check that accuracy is within the valid range [0, 1]
            self.assertTrue(0 <= accuracy.result <= 1, "Accuracy should be between 0 and 1.")
        else:
            self.fail(f"Evaluation failed with error: {evaluate_result.error}")

if __name__ == '__main__':
    unittest.main(verbosity=2)