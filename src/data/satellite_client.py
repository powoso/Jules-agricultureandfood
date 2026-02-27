import pandas as pd
import numpy as np
import logging

class SatelliteClient:
    """
    Client for fetching Satellite NDVI (Normalized Difference Vegetation Index) data.
    Currently implements a mock mode for development.
    """
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.logger = logging.getLogger(__name__)

    def get_ndvi_data(self, region, start_date, end_date):
        """
        Fetches NDVI data for a given region and date range.

        Args:
            region (str): Region identifier (e.g., "US-CORN-BELT").
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'ndvi', 'region']
        """
        if self.use_mock:
            return self._generate_mock_data(start_date, end_date, region)
        else:
            # Placeholder for real API implementation (e.g., using pystac_client)
            self.logger.warning("Real API access not implemented yet. Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'ndvi', 'region'])

    def _generate_mock_data(self, start_date, end_date, region):
        """
        Generates synthetic NDVI data for testing.
        NDVI typically ranges from -1 to 1, with healthy vegetation between 0.2 and 0.8.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)

        # Simulate seasonal curve (sine wave) + noise
        # Peak around day 180 (end of June)
        day_of_year = dates.dayofyear
        seasonal_trend = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        noise = np.random.normal(0, 0.05, n_days)
        ndvi_values = np.clip(seasonal_trend + noise, -1, 1)

        df = pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values,
            'region': region
        })
        return df
