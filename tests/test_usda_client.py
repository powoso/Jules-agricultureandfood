import unittest
from unittest.mock import patch, MagicMock
import requests
from src.data.usda_client import UsdaClient

class TestUsdaClient(unittest.TestCase):

    def setUp(self):
        self.client = UsdaClient(api_key="TEST_API_KEY")

    @patch('src.data.usda_client.requests.get')
    def test_get_crop_condition(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"commodity_desc": "CORN", "statisticcat_desc": "CONDITION", "unit_desc": "PCT EXCELLENT", "value": "15"}
            ]
        }
        mock_get.return_value = mock_response

        data = self.client.get_crop_condition("CORN", 2023)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['value'], "15")

        # Verify call arguments
        args, kwargs = mock_get.call_args
        self.assertIn('commodity_desc', kwargs['params'])
        self.assertEqual(kwargs['params']['commodity_desc'], "CORN")
        self.assertEqual(kwargs['params']['year'], 2023)

    @patch('src.data.usda_client.requests.get')
    def test_get_yield_data(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"commodity_desc": "CORN", "statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE", "value": "175.5"}
            ]
        }
        mock_get.return_value = mock_response

        data = self.client.get_yield_data("CORN", 2023)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['value'], "175.5")

    @patch('src.data.usda_client.requests.get')
    def test_api_failure(self, mock_get):
        # Mock failure
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        data = self.client.get_crop_condition("CORN", 2023)
        self.assertEqual(data, [])

if __name__ == '__main__':
    unittest.main()
