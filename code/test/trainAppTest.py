import unittest
import pandas as pd
from train import (
    handle_missing_customer_data, handle_missing_financial_data,
    handle_missing_enquiry_data, handle_missing_transaction_data
)

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        print("Setting up sample data...")
        self.sample_customer_data = pd.DataFrame({
            "customer_id": ["123", "456"],
            "name": ["Alice", "Bob"],
            "age": [30, 45],
            "gender": ["Female", "Male"],
            "marital_status": ["Single", "Married"],
            "education": ["Bachelor", "Master"],
            "occupation": ["Engineer", "Doctor"],
            "salary": [70000, 90000]
        })
        print("Sample DataFrame in setUp:")
        print(self.sample_customer_data)

        self.sample_financial_data = pd.DataFrame({
            "customer_id": ["123", "456"],
            "product_type": ["Credit Card", "Home Loan"],
            "loan_amount": [5000, 200000],
            "credit_limit": [15000, None],
            "credit_utilization": [0.5, None],
            "emi_paid": [10, 5],
            "tenure_months": [24, 60],
            "max_dpd": [30, 60],
            "default_status": [False, True]
        })

    def test_handle_missing_customer_data(self):
        """ Test customer data handling with missing values """
        print("Testing handle_missing_customer_data with sample data...")

        modified_df = handle_missing_customer_data(self.sample_customer_data.copy())

        print("Modified DataFrame:")
        print(modified_df)

        self.assertIsNotNone(modified_df, "Returned DataFrame should not be None")
        self.assertIn("customer_id", modified_df.columns, "customer_id should exist in DataFrame")
        self.assertEqual(len(modified_df), len(self.sample_customer_data), "Row count should match original")

    def test_handle_missing_financial_data(self):
        """ Test financial data handling """
        modified_df = handle_missing_financial_data(self.sample_financial_data.copy())

        self.assertIsNotNone(modified_df, "Returned DataFrame should not be None")
        self.assertIn("loan_amount", modified_df.columns, "loan_amount should exist in DataFrame")
        self.assertEqual(len(modified_df), len(self.sample_financial_data), "Row count should match original")

    def test_handle_missing_enquiry_data(self):
        """ Test enquiry data handling """
        sample_enquiry_data = pd.DataFrame({
            "customer_id": ["123", "456"],
            "product_type": ["Personal Loan", "Credit Card"],
            "enquiry_amount": ["500", "5000"]
        })

        modified_df = handle_missing_enquiry_data(sample_enquiry_data.copy())
        print(modified_df)
        self.assertIsNotNone(modified_df, "Returned DataFrame should not be None")
        self.assertIn("enquiry_date", modified_df.columns, "enquiry_date should exist in DataFrame")
        self.assertFalse(modified_df.isnull().values.any(), "No missing values should remain")

    def test_handle_missing_transaction_data(self):
        """ Test transaction data handling """
        sample_transaction_data = pd.DataFrame({
            "customer_id": ["123", "456"],
            "transaction_description": ["Amazon Purchase", "Travel Booking"]
        })

        modified_df = handle_missing_transaction_data(sample_transaction_data.copy())
        print(modified_df)
        self.assertIsNotNone(modified_df, "Returned DataFrame should not be None")
        self.assertIn("transaction_date", modified_df.columns, "transaction_date should exist in DataFrame")
        self.assertFalse(modified_df.isnull().values.any(), "No missing values should remain")

if __name__ == "__main__":
    unittest.main()
