import unittest
import pandas as pd
import numpy as np
from src.data.satellite_client import SatelliteClient

class TestSatelliteClient(unittest.TestCase):
    def setUp(self):
        self.client = SatelliteClient(use_mock=True)

    def test_get_ndvi_data_mock(self):
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        region = "US-TEST"

        df = self.client.get_ndvi_data(region, start_date, end_date)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertListEqual(list(df.columns), ['date', 'ndvi', 'region'])
        self.assertTrue((df['ndvi'] >= -1).all() and (df['ndvi'] <= 1).all())
        self.assertEqual(df['region'].iloc[0], region)

    def test_mock_seasonality(self):
        # Check if summer values are generally higher than winter values
        winter_df = self.client.get_ndvi_data("TEST", "2023-01-01", "2023-01-31")
        summer_df = self.client.get_ndvi_data("TEST", "2023-07-01", "2023-07-31")

        winter_mean = winter_df['ndvi'].mean()
        summer_mean = summer_df['ndvi'].mean()

        self.assertGreater(summer_mean, winter_mean)

if __name__ == '__main__':
    unittest.main()
