# Agriculture Prediction Markets System

This system is designed to predict agricultural outcomes (e.g., crop yield, food prices, drought conditions) by integrating diverse data sources like USDA crop reports and satellite imagery (NDVI). The insights generated can be used to inform trading decisions in prediction markets.

## Features

*   **Data Collection**:
    *   **USDA NASS Client**: Fetches crop condition reports (e.g., "PCT EXCELLENT") from the USDA Quick Stats API.
    *   **Satellite Client**: Fetches Normalized Difference Vegetation Index (NDVI) data to assess vegetation health. Currently includes a mock mode for development.
*   **Feature Engineering**:
    *   **Crop Condition Index**: Aggregates weekly USDA condition ratings.
    *   **NDVI Anomalies**: Calculates deviations from historical vegetation health baselines.
    *   **Data Alignment**: Merges weekly crop reports with daily satellite data.
*   **Prediction Model**:
    *   **Yield Model**: A linear regression model that predicts crop yield based on condition indices and NDVI anomalies.
*   **Simulation**:
    *   **Synthetic Data**: The system can generate synthetic historical data to demonstrate the workflow without requiring immediate API keys.

## Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)

## Installation

### macOS specific instructions

If you are on macOS and don't have Python or Git installed, you can use [Homebrew](https://brew.sh/):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python git
```

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repo-url>
    cd ag-prediction-market
    ```

2.  **Create a virtual environment** (highly recommended to avoid conflicting packages):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

To use real data from the USDA, you will need an API key.

1.  Request an API key from [USDA NASS Quick Stats API](https://quickstats.nass.usda.gov/api).
2.  Set the API key as an environment variable:
    ```bash
    export USDA_API_KEY="your_api_key_here"
    ```
    (On Windows: `set USDA_API_KEY=your_api_key_here`)

    *Note: The system will default to synthetic data if no valid key is provided or if the API call fails.*

## Usage

### 1. Interactive Web Dashboard (Beautiful UI)

To launch the beautiful interactive Streamlit dashboard:

```bash
streamlit run app.py
```

This will open a browser window displaying:
*   Historical yield and feature trends with interactive charts.
*   Model performance metrics.
*   An interactive prediction simulator where you can adjust crop condition and NDVI expectations to see real-time yield forecasts.

### 2. Command Line Pipeline

To run the full simulation pipeline from the terminal without the UI:

```bash
python3 main.py
```

This script will attempt to fetch real data (or fallback to synthetic data), process historical features, train the yield prediction model, and output a prediction for a hypothetical current season.

## Testing

To run the unit tests:

```bash
python3 -m unittest discover tests
```

## Uploading to GitHub

If you are setting this up as a new repository on GitHub, follow these steps:

1.  **Create a new repository** on GitHub (do not initialize with README, .gitignore, or license).
2.  **Initialize the local directory** (if not already a git repo):
    ```bash
    git init
    ```
3.  **Add files**:
    ```bash
    git add .
    ```
4.  **Commit changes**:
    ```bash
    git commit -m "Initial commit: Agriculture Prediction System"
    ```
5.  **Rename branch to main**:
    ```bash
    git branch -M main
    ```
6.  **Add remote origin**:
    ```bash
    git remote add origin https://github.com/<your-username>/<your-repo-name>.git
    ```
7.  **Push to GitHub**:
    ```bash
    git push -u origin main
    ```

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── usda_client.py       # USDA API interactions
│   │   └── satellite_client.py  # Satellite data handling (mocked)
│   ├── features/
│   │   └── processing.py        # Feature engineering pipeline
│   └── models/
│       └── yield_model.py       # Prediction logic
├── tests/                       # Unit tests
├── app.py                       # Streamlit interactive UI dashboard
├── main.py                      # CLI application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
