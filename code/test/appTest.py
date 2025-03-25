import unittest
import pandas as pd
from app import (
    generate_customer_demographics,
    generate_financial_behavior,
    generate_customer_enquiries,
    generate_customer_transactions,
    generate_customer_sentiments,
    generate_customer_data
)

class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        """Setup test data."""
        self.num_customers = 10
        self.customers = generate_customer_demographics(self.num_customers)
        self.customer_ids = self.customers['customer_id'].tolist()

    def test_generate_customer_demographics(self):
        """Test customer demographics data generation."""
        self.assertIsInstance(self.customers, pd.DataFrame)
        self.assertEqual(len(self.customers), self.num_customers)
        self.assertIn('customer_id', self.customers.columns)
        self.assertIn('age', self.customers.columns)

    def test_generate_financial_behavior(self):
        """Test financial behavior data generation."""
        df = generate_financial_behavior(self.customer_ids, num_records=20)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('customer_id', df.columns)
        self.assertIn('loan_amount', df.columns)

    def test_generate_customer_enquiries(self):
        """Test customer enquiries data generation."""
        df = generate_customer_enquiries(self.customer_ids, num_records=15)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('customer_id', df.columns)
        self.assertIn('product_type', df.columns)

    def test_generate_customer_transactions(self):
        """Test customer transactions data generation."""
        df = generate_customer_transactions(self.customer_ids, num_records=25)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('customer_id', df.columns)
        self.assertIn('transaction_amount', df.columns)

    def test_generate_customer_sentiments(self):
        """Test customer sentiments data generation."""
        df = generate_customer_sentiments(self.customer_ids, num_records=30)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('customer_id', df.columns)
        self.assertIn('sentiment_text', df.columns)

    def test_generate_customer_data(self):
        """Test complete customer data aggregation."""
        df = generate_customer_data(num_customers=10, num_financial_records=20, num_enquiries=15, num_transactions=25, num_sentiments=30)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('customer_id', df.columns)
        self.assertIn('content', df.columns)

if __name__ == '__main__':
    unittest.main()
