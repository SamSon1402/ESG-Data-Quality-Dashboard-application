# Recommendations tests
import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.recommendations import generate_recommendations, generate_action_plan

class TestRecommendations(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample quality metrics
        self.quality_metrics = {
            'completeness': {
                'energy_consumption_kwh': 80.0,
                'water_consumption_m3': 100.0,
                'co2_emissions_kg': 60.0,
                'waste_kg': 90.0
            },
            'outliers': {
                'energy_consumption_kwh': 5.0,
                'water_consumption_m3': 15.0,
                'co2_emissions_kg': 3.0,
                'waste_kg': 8.0
            },
            'consistency_by_source': {
                'energy_consumption_kwh': {'BMS': 0.05, 'Manual': 0.15, 'API': 0.25},
                'water_consumption_m3': {'BMS': 0.05, 'Manual': 0.15, 'API': 0.25},
                'co2_emissions_kg': {'BMS': 0.05, 'Manual': 0.15, 'API': 0.25},
                'waste_kg': {'BMS': 0.05, 'Manual': 0.15, 'API': 0.25}
            },
            'quality_scores': {
                'energy_consumption_kwh': 85.0,
                'water_consumption_m3': 92.0,
                'co2_emissions_kg': 75.0,
                'waste_kg': 88.0
            },
            'completeness_over_time': pd.DataFrame()
        }
    
    def test_generate_recommendations(self):
        """Test generate_recommendations function"""
        # Generate recommendations
        recommendations = generate_recommendations(self.quality_metrics)
        
        # Check if recommendations exist
        self.assertTrue(len(recommendations) > 0)
        
        # Check if high impact recommendation exists
        high_impact_recs = [r for r in recommendations if r['impact'] == 'HIGH']
        self.assertTrue(len(high_impact_recs) > 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('issue', rec)
            self.assertIn('description', rec)
            self.assertIn('impact', rec)
            self.assertIn('recommendation', rec)
    
    def test_generate_action_plan(self):
        """Test generate_action_plan function"""
        # Generate recommendations
        recommendations = generate_recommendations(self.quality_metrics)
        
        # Generate action plan
        action_plan = generate_action_plan(recommendations, priority_level="all")
        
        # Check action plan structure
        self.assertIn('summary', action_plan)
        self.assertIn('timeline_data', action_plan)
        self.assertIn('implementation_steps', action_plan)
        
        # Check implementation steps
        self.assertEqual(len(action_plan['implementation_steps']), len(recommendations))
        
        for step in action_plan['implementation_steps']:
            self.assertIn('recommendation', step)
            self.assertIn('steps', step)
            self.assertIn('timeline', step)
            self.assertIn('responsible_team', step)
            self.assertTrue(len(step['steps']) > 0)
        
        # Test with different priority levels
        critical_plan = generate_action_plan(recommendations, priority_level="critical")
        medium_plan = generate_action_plan(recommendations, priority_level="medium")
        
        # Critical plan should only have HIGH impact recommendations
        critical_recs = [step['recommendation'] for step in critical_plan['implementation_steps']]
        self.assertTrue(all(rec['impact'] == 'HIGH' for rec in critical_recs))
        
        # Medium plan should have HIGH and MEDIUM impact recommendations
        medium_recs = [step['recommendation'] for step in medium_plan['implementation_steps']]
        self.assertTrue(all(rec['impact'] in ['HIGH', 'MEDIUM'] for rec in medium_recs))

if __name__ == '__main__':
    unittest.main()