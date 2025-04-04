# Quality metrics tests
import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.quality_metrics import (
    calculate_completeness,
    detect_outliers,
    calculate_consistency,
    calculate_quality_score,
    calculate_all_quality_metrics
)

class TestQualityMetrics(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create test dataframe
        self.df = pd.DataFrame({
            'building_id': ['B1', 'B1', 'B2', 'B2', 'B3'],
            'date': [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 1, 1)
            ],
            'energy_consumption_kwh': [1000, 1100, 500, 550, np.nan],
            'water_consumption_m3': [100, 110, 50, 55, 200],
            'co2_emissions_kg': [200, 220, 100, 110, np.nan],
            'waste_kg': [50, 55, 25, 28, 100],
            'data_source': ['BMS', 'BMS', 'Manual', 'Manual', 'API'],
            'building_type': ['Office', 'Office', 'Retail', 'Retail', 'Residential'],
            'building_size_sqm': [10000, 10000, 5000, 5000, 20000]
        })
        
        # Add an outlier
        self.df.loc[4, 'water_consumption_m3'] = 2000  # 10x typical value
    
    def test_calculate_completeness(self):
        """Test calculate_completeness function"""
        # Calculate completeness for ESG metrics
        esg_columns = ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']
        completeness = calculate_completeness(self.df, columns=esg_columns)
        
        # Check completeness values
        self.assertEqual(completeness['energy_consumption_kwh'], 80.0)
        self.assertEqual(completeness['water_consumption_m3'], 100.0)
        self.assertEqual(completeness['co2_emissions_kg'], 80.0)
        self.assertEqual(completeness['waste_kg'], 100.0)
    
    def test_detect_outliers(self):
        """Test detect_outliers function"""
        # Detect outliers for ESG metrics
        esg_columns = ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']
        outliers = detect_outliers(self.df, columns=esg_columns)
        
        # Check outlier percentages
        self.assertAlmostEqual(outliers['water_consumption_m3'], 20.0)
        self.assertAlmostEqual(outliers['waste_kg'], 20.0)
    
    def test_calculate_consistency(self):
        """Test calculate_consistency function"""
        # Calculate consistency for ESG metrics
        esg_columns = ['energy_consumption_kwh', 'water_consumption_m3']
        consistency = calculate_consistency(
            self.df, 
            columns=esg_columns, 
            groupby_cols=['data_source']
        )
        
        # Check if consistency metrics exist
        self.assertIn('energy_consumption_kwh', consistency)
        self.assertIn('water_consumption_m3', consistency)
        
        # Check if data sources exist in consistency metrics
        self.assertIn('BMS', consistency['energy_consumption_kwh'])
        self.assertIn('Manual', consistency['energy_consumption_kwh'])
    
    def test_calculate_quality_score(self):
        """Test calculate_quality_score function"""
        # Sample data
        completeness = {
            'energy_consumption_kwh': 80.0,
            'water_consumption_m3': 100.0
        }
        
        outliers = {
            'energy_consumption_kwh': 0.0,
            'water_consumption_m3': 20.0
        }
        
        consistency = {
            'energy_consumption_kwh': {'BMS': 0.05, 'Manual': 0.05},
            'water_consumption_m3': {'BMS': 0.05, 'Manual': 0.05}
        }
        
        # Calculate quality scores
        quality_scores = calculate_quality_score(completeness, outliers, consistency)
        
        # Check quality scores
        self.assertGreater(quality_scores['energy_consumption_kwh'], 0)
        self.assertGreater(quality_scores['water_consumption_m3'], 0)
    
    def test_calculate_all_quality_metrics(self):
        """Test calculate_all_quality_metrics function"""
        # Calculate all quality metrics
        quality_metrics = calculate_all_quality_metrics(self.df)
        
        # Check if all metrics are present
        self.assertIn('completeness', quality_metrics)
        self.assertIn('outliers', quality_metrics)
        self.assertIn('consistency_by_source', quality_metrics)
        self.assertIn('quality_scores', quality_metrics)
        self.assertIn('completeness_over_time', quality_metrics)

if __name__ == '__main__':
    unittest.main()