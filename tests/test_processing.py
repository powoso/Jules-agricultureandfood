import unittest
import pandas as pd
import numpy as np
from src.features.processing import FeaturePipeline

class TestFeaturePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = FeaturePipeline()

    def test_process_crop_conditions(self):
        raw_data = [
            {"year": 2023, "week_ending": "2023-06-04", "unit_desc": "PCT EXCELLENT", "value": "15"},
            {"year": 2023, "week_ending": "2023-06-11", "unit_desc": "PCT EXCELLENT", "value": "18"},
             # Simulate multiple entries for same date (e.g. if we had Good + Excellent)
            {"year": 2023, "week_ending": "2023-06-11", "unit_desc": "PCT GOOD", "value": "50"}
        ]

        df = self.pipeline.process_crop_conditions(raw_data)

        self.assertEqual(len(df), 2)
        # Check aggregation for 2023-06-11 (18 + 50 = 68)
        self.assertEqual(df[df['week_ending'] == '2023-06-11']['condition_index'].values[0], 68)

    def test_process_ndvi_data_fallback(self):
        # Test fallback behavior (without fit)
        dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
        ndvi_values = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
        df_input = pd.DataFrame({'date': dates, 'ndvi': ndvi_values, 'region': 'TEST'})

        df_output = self.pipeline.process_ndvi_data(df_input)

        self.assertEqual(len(df_output), 10)
        self.assertIn('ndvi_anomaly', df_output.columns)
        # Without fit, anomaly is deviation from self-mean (which is self for unique days) -> 0
        self.assertTrue((df_output['ndvi_anomaly'] == 0).all())

    def test_process_ndvi_data_with_fit(self):
        # Test with fit
        # History: consistently 0.5
        history_dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
        history_df = pd.DataFrame({'date': history_dates, 'ndvi': [0.5]*len(history_dates), 'region': 'TEST'})

        self.pipeline.fit_ndvi_stats(history_df)

        # Current: 0.6 (anomaly should be +0.1)
        current_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq='D')
        current_df = pd.DataFrame({'date': current_dates, 'ndvi': [0.6]*len(current_dates), 'region': 'TEST'})

        df_output = self.pipeline.process_ndvi_data(current_df)

        # Check anomalies
        # Allow small float error
        np.testing.assert_allclose(df_output['ndvi_anomaly'].values, 0.1)

    def test_merge_data(self):
        condition_df = pd.DataFrame({
            'year': [2023],
            'week_ending': pd.to_datetime(['2023-06-04']),
            'condition_index': [70]
        })

        ndvi_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-06-04']),
            'ndvi_anomaly': [0.05]
        })

        merged_df = self.pipeline.merge_data(condition_df, ndvi_df)

        self.assertEqual(len(merged_df), 1)
        self.assertEqual(merged_df.iloc[0]['condition_index'], 70)
        self.assertEqual(merged_df.iloc[0]['ndvi_anomaly'], 0.05)

if __name__ == '__main__':
    unittest.main()
