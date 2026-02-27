import pandas as pd
import numpy as np

class FeaturePipeline:
    """
    Pipeline for processing agricultural data into features for prediction models.
    Supports a stateful approach for NDVI anomaly calculation.
    """
    def __init__(self):
        self.ndvi_stats = None

    def process_crop_conditions(self, crop_condition_data):
        """
        Processes raw USDA crop condition data.
        Aggregates 'PCT EXCELLENT' and 'PCT GOOD' into a single 'condition_index'.

        Args:
            crop_condition_data (list): List of dictionaries from UsdaClient.

        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'week_ending', 'condition_index']
        """
        if not crop_condition_data:
            return pd.DataFrame()

        df = pd.DataFrame(crop_condition_data)

        # Ensure relevant columns exist
        # Note: 'Value' key from USDA API is often capitalized, but let's handle case sensitivity if needed.
        # Based on previous tests, it might be 'value' or 'Value'. The mock used 'value'.
        # Let's standardize column names to lowercase.
        df.columns = [c.lower() for c in df.columns]

        # Convert value to numeric, handling potential non-numeric strings
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

        # Convert week_ending to datetime
        df['week_ending'] = pd.to_datetime(df['week_ending'])

        # Group by year and week_ending, summing the percentages (e.g. Good + Excellent)
        df_agg = df.groupby(['year', 'week_ending'])['value'].sum().reset_index()
        df_agg.rename(columns={'value': 'condition_index'}, inplace=True)

        return df_agg

    def fit_ndvi_stats(self, ndvi_df):
        """
        Calculates and stores historical mean NDVI per day of year.

        Args:
            ndvi_df (pd.DataFrame): Historical NDVI data.
        """
        if ndvi_df.empty:
            return

        df = ndvi_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear

        # Calculate mean per day of year
        self.ndvi_stats = df.groupby('day_of_year')['ndvi'].mean()

    def process_ndvi_data(self, ndvi_df):
        """
        Processes NDVI data to calculate anomalies using stored historical stats.
        If stats are not fitted, calculates anomalies based on the input data itself (fallback).

        Args:
            ndvi_df (pd.DataFrame): DataFrame from SatelliteClient.

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'ndvi_anomaly']
        """
        if ndvi_df.empty:
            return pd.DataFrame()

        df = ndvi_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear

        if self.ndvi_stats is not None:
            # Map the historical mean to the current dataframe
            # Use map or merge. Map is cleaner if index is unique day_of_year
            historical_mean = df['day_of_year'].map(self.ndvi_stats)
            # Handle cases where day_of_year might not be in stats (e.g. leap year day 366)
            # Fill NaNs with the mean of the input df for that day if available, or just 0 anomaly
            historical_mean = historical_mean.fillna(df.groupby('day_of_year')['ndvi'].transform('mean'))
        else:
            # Fallback: calculate mean from input data itself
            historical_mean = df.groupby('day_of_year')['ndvi'].transform('mean')

        df['ndvi_anomaly'] = df['ndvi'] - historical_mean

        return df[['date', 'ndvi_anomaly']]

    def merge_data(self, condition_df, ndvi_df):
        """
        Merges crop condition data with NDVI data.
        Since conditions are weekly and NDVI is daily, we can resample NDVI to weekly or merge on nearest date.
        Here we will merge on the nearest date.

        Args:
            condition_df (pd.DataFrame): Processed crop conditions.
            ndvi_df (pd.DataFrame): Processed NDVI data.

        Returns:
            pd.DataFrame: Merged DataFrame ready for modeling.
        """
        if condition_df.empty or ndvi_df.empty:
            return pd.DataFrame()

        # Sort by date
        condition_df = condition_df.sort_values('week_ending')
        ndvi_df = ndvi_df.sort_values('date')

        # Merge using merge_asof
        merged_df = pd.merge_asof(
            condition_df,
            ndvi_df,
            left_on='week_ending',
            right_on='date',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        )

        # Drop rows where match failed (though merge_asof keeps left rows, so we check for NaNs in right cols)
        merged_df.dropna(subset=['ndvi_anomaly'], inplace=True)

        return merged_df[['year', 'week_ending', 'condition_index', 'ndvi_anomaly']]
