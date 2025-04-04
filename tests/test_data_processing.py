import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_csv_data
from src.data.generator import generate_sample_buildings, generate_sample_energy_data
from src.data.processor import clean_dataframe, prepare_esg_data, normalize_esg_metrics

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create test buildings
        self.buildings_df = pd.DataFrame({
            'building_id': ['B1', 'B2', 'B3'],
            'building_type': ['Office', 'Retail', 'Residential'],
            'building_size_sqm': [10000, 5000, 20000],
            'location': ['Paris', 'London', 'Berlin']
        })
        
        # Create test energy data
        self.energy_df = pd.DataFrame({
            'building_id': ['B1', 'B1', 'B2', 'B2', 'B3'],
            'date': [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1)
            ],
            'energy_consumption_kwh': [1000, 1100, 500, 550, 2000],
            'data_source': ['BMS', 'BMS', 'Manual', 'Manual', 'API']
        })
        
        # Create test water data
        self.water_df = pd.DataFrame({
            'building_id': ['B1', 'B1', 'B2', 'B2', 'B3'],
            'date': [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1)
            ],
            'water_consumption_m3': [100, 110, 50, 55, 200],
            'data_source': ['Utility', 'Utility', 'Manual', 'Manual', 'API']
        })
    
    def test_clean_dataframe(self):
        """Test clean_dataframe function"""
        # Create a dataframe with duplicates and non-datetime dates
        df = pd.DataFrame({
            'building_id': ['B1', 'B1', 'B2'],
            'date': ['2024-01-01', '2024-01-01', '2024-02-01'],
            'value': [1, 1, 2]
        })
        
        # Clean dataframe
        cleaned_df = clean_dataframe(df)
        
        # Check duplicates removed
        self.assertEqual(len(cleaned_df), 2)
        
        # Check date converted to datetime
        self.assertEqual(cleaned_df['date'].dtype, 'datetime64[ns]')
    
    def test_prepare_esg_data(self):
        """Test prepare_esg_data function"""
        # Prepare ESG data
        esg_data = prepare_esg_data(
            self.buildings_df,
            self.energy_df,
            self.water_df
        )
        
        # Check merged data
        self.assertEqual(len(esg_data), 5)
        
        # Check columns
        required_columns = [
            'building_id', 'date', 'energy_consumption_kwh', 'data_source',
            'building_type', 'building_size_sqm', 'water_consumption_m3',
            'co2_emissions_kg', 'waste_kg'
        ]
        for col in required_columns:
            self.assertIn(col, esg_data.columns)
        
        # Check CO2 calculation
        self.assertEqual(esg_data.loc[0, 'co2_emissions_kg'], 1000 * 0.2)
    
    def test_normalize_esg_metrics(self):
        """Test normalize_esg_metrics function"""
        # Create test dataframe
        df = pd.DataFrame({
            'building_id': ['B1', 'B2'],
            'building_size_sqm': [10000, 5000],
            'energy_consumption_kwh': [1000, 500],
            'water_consumption_m3': [100, 50],
            'co2_emissions_kg': [200, 100],
            'waste_kg': [50, 25]
        })
        
        # Normalize metrics
        normalized_df = normalize_esg_metrics(df)
        
        # Check normalized columns
        self.assertEqual(normalized_df.loc[0, 'energy_intensity'], 0.1)
        self.assertEqual(normalized_df.loc[0, 'water_intensity'], 0.01)
        self.assertEqual(normalized_df.loc[0, 'co2_intensity'], 0.02)
        self.assertEqual(normalized_df.loc[0, 'waste_intensity'], 0.005)

if __name__ == '__main__':
    unittest.main()