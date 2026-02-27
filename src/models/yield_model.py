import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

class YieldModel:
    """
    Basic linear regression model to predict crop yield based on condition index and NDVI anomaly.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.logger = logging.getLogger(__name__)

    def train(self, merged_data, yield_data):
        """
        Trains the yield prediction model.

        Args:
            merged_data (pd.DataFrame): DataFrame with 'year', 'week_ending', 'condition_index', 'ndvi_anomaly'.
            yield_data (pd.DataFrame): DataFrame with 'year' and 'yield_value' (actual yield).

        Returns:
            dict: Training metrics (MSE, R2).
        """
        if merged_data.empty or yield_data.empty:
            self.logger.warning("Empty data provided for training.")
            return {}

        # We need to aggregate features by year to match annual yield
        # For simplicity, let's take the mean condition and NDVI anomaly during the growing season (which is what we assume merged_data contains)
        annual_features = merged_data.groupby('year')[['condition_index', 'ndvi_anomaly']].mean().reset_index()

        # Merge with target
        training_set = pd.merge(annual_features, yield_data, on='year')

        if training_set.empty:
            self.logger.warning("No overlapping years between features and yield data.")
            return {}

        X = training_set[['condition_index', 'ndvi_anomaly']]
        y = training_set['yield_value']

        # Need at least 2 points to fit a line, ideally more for a meaningful split
        if len(training_set) < 2:
             self.logger.warning("Not enough data points to train (need at least 2).")
             return {}

        # Train/Test split (mock scenario might have very few points, so be careful)
        # If very few points, just train on all
        if len(training_set) < 5:
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {"mse": mse, "r2": r2}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"mse": mse, "r2": r2}

    def predict(self, current_data):
        """
        Predicts yield for the current year(s) based on current season data.

        Args:
            current_data (pd.DataFrame): DataFrame with 'year', 'week_ending', 'condition_index', 'ndvi_anomaly'.

        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'predicted_yield'].
        """
        if current_data.empty:
            return pd.DataFrame()

        # Aggregate features by year
        annual_features = current_data.groupby('year')[['condition_index', 'ndvi_anomaly']].mean().reset_index()

        X = annual_features[['condition_index', 'ndvi_anomaly']]

        # Check if model is fitted
        try:
            predictions = self.model.predict(X)
        except Exception: # NotFittedError but catching broadly for simplicity if sklearn version differs
            self.logger.warning("Model not fitted yet.")
            return pd.DataFrame()

        return pd.DataFrame({'year': annual_features['year'], 'predicted_yield': predictions})
