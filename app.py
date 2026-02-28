import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.data.usda_client import UsdaClient
from src.data.satellite_client import SatelliteClient
from src.features.processing import FeaturePipeline
from src.models.yield_model import YieldModel

st.set_page_config(
    page_title="AgriPredict - Yield Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Agriculture Prediction Markets System")
st.markdown("Predict agricultural outcomes by integrating USDA crop reports and satellite imagery.")

# Sidebar
st.sidebar.header("Configuration")
use_mock_data = st.sidebar.checkbox("Use Mock Data (No API Key Required)", value=True)
api_key = st.sidebar.text_input("USDA NASS API Key", type="password", disabled=use_mock_data)

st.sidebar.markdown("---")
st.sidebar.header("Prediction Settings")
target_crop = st.sidebar.selectbox("Target Crop", ["CORN", "SOYBEANS", "WHEAT"])
target_year = st.sidebar.slider("Prediction Year", min_value=2023, max_value=2025, value=2023)

@st.cache_resource
def load_and_process_data(mock, key):
    # Initialize components
    usda_client = UsdaClient(api_key=key if not mock else "DEMO")
    satellite_client = SatelliteClient(use_mock=True) # Satellite is always mocked for now
    pipeline = FeaturePipeline()
    model = YieldModel()

    # Generate/Fetch Historical Data (2010-2022)
    years = range(2010, 2023)
    historical_conditions = []

    if not mock and key:
        # Attempt to fetch real data
        try:
            for year in years:
                # Fetching one year at a time based on our client design
                year_data = usda_client.get_crop_condition("CORN", year)
                if year_data:
                    historical_conditions.extend(year_data)
        except Exception as e:
            st.warning(f"Failed to fetch real data from USDA API: {e}. Falling back to synthetic data.")
            historical_conditions = [] # Reset to fall back

    # Fallback to synthetic data if mock is checked or real fetch failed
    if not historical_conditions:
        for year in years:
            base_condition = 60 + (year % 5) * 5
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

    start_date = "2010-01-01"
    end_date = "2022-12-31"
    historical_ndvi_df = satellite_client.get_ndvi_data("US-CORN-BELT", start_date, end_date)

    # Process
    pipeline.fit_ndvi_stats(historical_ndvi_df)
    hist_condition_df = pipeline.process_crop_conditions(historical_conditions)
    hist_ndvi_processed = pipeline.process_ndvi_data(historical_ndvi_df)
    hist_merged_df = pipeline.merge_data(hist_condition_df, hist_ndvi_processed)

    # Generate synthetic yield target
    annual_data = hist_merged_df.groupby('year')[['condition_index', 'ndvi_anomaly']].mean()
    true_yield = 100 + 0.8 * annual_data['condition_index'] + 20 * annual_data['ndvi_anomaly']
    yield_values = true_yield + np.random.normal(0, 2, len(true_yield))

    yield_data = pd.DataFrame({
        'year': yield_values.index,
        'yield_value': yield_values.values
    })

    metrics = model.train(hist_merged_df, yield_data)

    return pipeline, model, hist_merged_df, yield_data, metrics, satellite_client

# Main execution
with st.spinner('Loading data and training models...'):
    pipeline, model, hist_merged_df, yield_data, metrics, sat_client = load_and_process_data(use_mock_data, api_key)

if not hist_merged_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model R² Score</div>
            <div class="metric-value">{metrics.get('r2', 0):.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training MSE</div>
            <div class="metric-value">{metrics.get('mse', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.header("📈 Historical Data Analysis")

    # Historical Yield Chart
    chart_data = yield_data.rename(columns={'yield_value': 'Yield (Bu/Acre)', 'year': 'Year'})
    yield_chart = alt.Chart(chart_data).mark_line(point=True, color="#2ca02c").encode(
        x='Year:O',
        y=alt.Y('Yield (Bu/Acre):Q', scale=alt.Scale(zero=False)),
        tooltip=['Year', 'Yield (Bu/Acre)']
    ).properties(title="Historical Crop Yield")

    st.altair_chart(yield_chart, use_container_width=True)

    # Feature trends
    st.subheader("Feature Trends")
    tab1, tab2 = st.tabs(["Crop Condition Index", "NDVI Anomaly"])

    annual_features = hist_merged_df.groupby('year')[['condition_index', 'ndvi_anomaly']].mean().reset_index()

    with tab1:
        cond_chart = alt.Chart(annual_features).mark_bar(color="#1f77b4").encode(
            x='year:O',
            y='condition_index:Q',
            tooltip=['year', 'condition_index']
        ).properties(title="Average Crop Condition Index by Year")
        st.altair_chart(cond_chart, use_container_width=True)

    with tab2:
        ndvi_chart = alt.Chart(annual_features).mark_bar().encode(
            x='year:O',
            y='ndvi_anomaly:Q',
            color=alt.condition(
                alt.datum.ndvi_anomaly > 0,
                alt.value("steelblue"),  # The positive color
                alt.value("orange")  # The negative color
            ),
            tooltip=['year', 'ndvi_anomaly']
        ).properties(title="Average NDVI Anomaly by Year")
        st.altair_chart(ndvi_chart, use_container_width=True)

    st.header("🔮 Prediction for Current Season")
    st.markdown(f"Running simulation for **{target_crop}** in **{target_year}**.")

    # Simulate current season data interactively
    col3, col4 = st.columns(2)
    with col3:
        current_condition_base = st.slider("Simulate Crop Condition (Avg % Excellent)", min_value=10, max_value=90, value=75)
    with col4:
        ndvi_boost = st.slider("Simulate NDVI Shift (Vegetation Health)", min_value=-0.1, max_value=0.1, value=0.05, step=0.01)

    # Generate Prediction Data
    current_season_conditions = []
    for week_offset in range(5):
        date = pd.Timestamp(f"{target_year}-06-01") + pd.Timedelta(weeks=week_offset)
        current_season_conditions.append({
            "year": target_year,
            "week_ending": date.strftime("%Y-%m-%d"),
            "unit_desc": "PCT EXCELLENT",
            "value": str(current_condition_base + np.random.normal(0,2))
        })

    current_ndvi = sat_client.get_ndvi_data("US-CORN-BELT", f"{target_year}-05-01", f"{target_year}-07-30")
    current_ndvi['ndvi'] += ndvi_boost

    current_cond_df = pipeline.process_crop_conditions(current_season_conditions)
    current_ndvi_proc = pipeline.process_ndvi_data(current_ndvi)
    current_merged = pipeline.merge_data(current_cond_df, current_ndvi_proc)

    if not current_merged.empty:
        prediction = model.predict(current_merged)
        if not prediction.empty:
            predicted_yield = prediction['predicted_yield'].iloc[0]

            st.markdown(f"""
            <div style="background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: #155724; border-bottom: none; margin-bottom: 0;">Predicted Yield for {target_year}</h2>
                <div style="font-size: 3rem; font-weight: bold;">{predicted_yield:.2f} Bu/Acre</div>
            </div>
            """, unsafe_allow_html=True)

            # Show features used
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("Features used for this prediction:")
            st.dataframe(current_merged[['week_ending', 'condition_index', 'ndvi_anomaly']].tail())

        else:
            st.error("Failed to generate prediction.")
    else:
        st.warning("Not enough current season data to make a prediction.")

else:
    st.error("Failed to load historical data.")
