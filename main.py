import logging
import pandas as pd
import numpy as np
import requests
from src.data.usda_client import UsdaClient
from src.data.satellite_client import SatelliteClient
from src.features.processing import FeaturePipeline
from src.models.yield_model import YieldModel

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Agriculture Prediction System...")

    # Initialize components
    usda_client = UsdaClient(api_key="DEMO_KEY") # Mock key for demo
    satellite_client = SatelliteClient(use_mock=True)
    pipeline = FeaturePipeline()
    model = YieldModel()

    # 1. Fetch Historical Data (Training Phase)
    logger.info("Fetching Historical Data (2010-2022)...")

    # 1a. USDA Conditions
    # Use synthetic USDA data for demonstration
    years = range(2010, 2023)
    historical_conditions = []

    for year in years:
        # Generate 10 weeks of data per year during growing season (approx weeks 20-30)
        base_condition = 60 + (year % 5) * 5 # Vary base condition by year
        for week_offset in range(10):
            date = pd.Timestamp(f"{year}-06-01") + pd.Timedelta(weeks=week_offset)
            condition_val = base_condition + np.sin(week_offset/3) * 10 + np.random.normal(0, 2)
            condition_val = max(0, min(100, condition_val))

            historical_conditions.append({
                "year": year,
                "week_ending": date.strftime("%Y-%m-%d"),
                "unit_desc": "PCT EXCELLENT",
                "value": str(int(condition_val))
            })

    logger.info(f"Generated {len(historical_conditions)} synthetic historical crop condition records.")

    # 1b. Satellite NDVI
    start_date = "2010-01-01"
    end_date = "2022-12-31"
    historical_ndvi_df = satellite_client.get_ndvi_data("US-CORN-BELT", start_date, end_date)
    logger.info(f"Fetched {len(historical_ndvi_df)} historical NDVI records.")

    # 2. Process Historical Features
    logger.info("Processing historical features...")

    # Fit NDVI stats on history
    logger.info("Fitting NDVI baseline statistics...")
    pipeline.fit_ndvi_stats(historical_ndvi_df)

    # Transform history
    hist_condition_df = pipeline.process_crop_conditions(historical_conditions)
    hist_ndvi_processed = pipeline.process_ndvi_data(historical_ndvi_df)

    # Merge
    hist_merged_df = pipeline.merge_data(hist_condition_df, hist_ndvi_processed)
    logger.info(f"Historical merged data shape: {hist_merged_df.shape}")

    # 3. Train Model
    logger.info("Training yield prediction model...")

    if hist_merged_df.empty:
        logger.error("Merged data is empty. Cannot train.")
        return

    # Generate synthetic yield target based on the features
    annual_data = hist_merged_df.groupby('year')[['condition_index', 'ndvi_anomaly']].mean()

    # Simple formula: Yield = 100 + 0.8 * Condition + 20 * NDVI_Anomaly
    true_yield = 100 + 0.8 * annual_data['condition_index'] + 20 * annual_data['ndvi_anomaly']
    # Add random noise
    noise = np.random.normal(0, 2, len(true_yield))
    yield_values = true_yield + noise

    yield_data = pd.DataFrame({
        'year': yield_values.index,
        'yield_value': yield_values.values
    })

    metrics = model.train(hist_merged_df, yield_data)
    logger.info(f"Model Training Metrics: {metrics}")

    # 4. Predict for Current Season (Hypothetical 2023)
    logger.info("Predicting for 2023 season...")

    # Generate 2023 data manually
    current_season_conditions = []
    # Let's say 2023 is a good year
    for week_offset in range(5):
        date = pd.Timestamp("2023-06-01") + pd.Timedelta(weeks=week_offset)
        current_season_conditions.append({
            "year": 2023,
            "week_ending": date.strftime("%Y-%m-%d"),
            "unit_desc": "PCT EXCELLENT",
            "value": "75" # High condition
        })

    # Get mock NDVI for 2023
    # Use a slightly different pattern to ensure non-zero anomaly
    # Mock client generates consistent seasonal pattern, so let's shift it manually or assume mock changes by year?
    # The mock client implementation is stateless and deterministic by date.
    # To get an anomaly, we rely on the noise in the mock or differences in the specific dates fetched vs average.
    # Or we can just mock the dataframe return here for better demonstration.

    current_ndvi = satellite_client.get_ndvi_data("US-CORN-BELT", "2023-05-01", "2023-07-30")
    # Artificial boost to simulate good year (anomaly)
    current_ndvi['ndvi'] += 0.05

    current_cond_df = pipeline.process_crop_conditions(current_season_conditions)

    # Transform using fitted stats
    current_ndvi_proc = pipeline.process_ndvi_data(current_ndvi)

    # Merge
    current_merged = pipeline.merge_data(current_cond_df, current_ndvi_proc)

    if not current_merged.empty:
        # Check anomaly values
        avg_anomaly = current_merged['ndvi_anomaly'].mean()
        logger.info(f"Average NDVI Anomaly for 2023: {avg_anomaly:.4f}")

        prediction = model.predict(current_merged)
        if not prediction.empty:
            predicted_yield = prediction['predicted_yield'].iloc[0]
            logger.info(f"Predicted Yield for 2023: {predicted_yield:.2f} Bu/Acre")
        else:
            logger.warning("Prediction returned empty result.")
    else:
        logger.warning("Could not merge current season data for prediction.")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
