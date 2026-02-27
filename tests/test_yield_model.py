import unittest
import pandas as pd
import numpy as np
from src.models.yield_model import YieldModel

class TestYieldModel(unittest.TestCase):
    def setUp(self):
        self.model = YieldModel()

    def test_train_and_predict(self):
        # Create synthetic training data
        years = [2010, 2011, 2012, 2013, 2014, 2015]
        # Create a dataframe with repeated years to simulate multiple weeks per year
        merged_data = pd.DataFrame({
            'year': np.repeat(years, 10), # 10 weeks per year
            'week_ending': pd.date_range(start='2010-01-01', periods=60, freq='W'),
            'condition_index': np.random.uniform(50, 90, 60),
            'ndvi_anomaly': np.random.uniform(-0.1, 0.1, 60)
        })

        # Calculate annual means to create target variable
        annual_means = merged_data.groupby('year')[['condition_index', 'ndvi_anomaly']].mean()

        # Create target yield with a known relationship
        yield_values = 100 + 1.5 * annual_means['condition_index'] + 100 * annual_means['ndvi_anomaly']

        yield_data = pd.DataFrame({
            'year': years,
            'yield_value': yield_values.values
        })

        # Train model
        metrics = self.model.train(merged_data, yield_data)

        # Since we have enough data (6 years > 5), it should return metrics
        self.assertIn('mse', metrics)
        self.assertIn('r2', metrics)
        # Perfect relationship should yield high R2
        self.assertGreater(metrics['r2'], 0.9)

        # Test prediction
        new_data = pd.DataFrame({
            'year': [2023, 2023],
            'week_ending': pd.to_datetime(['2023-07-01', '2023-07-08']),
            'condition_index': [80, 85],
            'ndvi_anomaly': [0.05, 0.05]
        })

        predictions = self.model.predict(new_data)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions['year'].iloc[0], 2023)
        self.assertGreater(predictions['predicted_yield'].iloc[0], 0)

    def test_empty_data(self):
        metrics = self.model.train(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(metrics, {})

        preds = self.model.predict(pd.DataFrame())
        self.assertTrue(preds.empty)

if __name__ == '__main__':
    unittest.main()
