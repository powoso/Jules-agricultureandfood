import os
import requests
import logging

class UsdaClient:
    """
    Client for interacting with the USDA NASS Quick Stats API.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("USDA_API_KEY")
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
        self.logger = logging.getLogger(__name__)

    def get_crop_condition(self, commodity_desc, year, state_alpha=None):
        """
        Fetches crop condition data (PCT EXCELLENT, PCT GOOD, etc.) for a specific commodity and year.

        Args:
            commodity_desc (str): e.g., "CORN", "SOYBEANS", "WHEAT"
            year (int): Year to fetch data for.
            state_alpha (str, optional): Two-letter state code (e.g., "IA", "IL").

        Returns:
            list: List of dictionaries containing the API response.
        """
        params = {
            "key": self.api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "commodity_desc": commodity_desc.upper(),
            "statisticcat_desc": "CONDITION",
            "unit_desc": "PCT EXCELLENT", # Fetches excellent only for now, can expand later
            "year": year,
            "format": "JSON"
        }

        if state_alpha:
            params["state_alpha"] = state_alpha.upper()

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from USDA API: {e}")
            return []

    def get_yield_data(self, commodity_desc, year, state_alpha=None):
        """
        Fetches yield data for a specific commodity and year.

        Args:
            commodity_desc (str): e.g., "CORN", "SOYBEANS"
            year (int): Year to fetch data for.
            state_alpha (str, optional): Two-letter state code.

        Returns:
            list: List of dictionaries containing the API response.
        """
        params = {
            "key": self.api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "commodity_desc": commodity_desc.upper(),
            "statisticcat_desc": "YIELD",
            "unit_desc": "BU / ACRE",
            "year": year,
            "format": "JSON"
        }

        if state_alpha:
            params["state_alpha"] = state_alpha.upper()

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching yield data from USDA API: {e}")
            return []
