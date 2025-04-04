# Sample data generation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_buildings(num_buildings=50):
    """Generate sample building data"""
    np.random.seed(42)
    buildings = [f"Building_{i}" for i in range(1, num_buildings+1)]
    building_types = np.random.choice(['Office', 'Retail', 'Residential', 'Industrial', 'Mixed-Use'], num_buildings)
    building_sizes = np.random.randint(1000, 100000, num_buildings)
    
    locations = [
        "Paris", "London", "Berlin", "Madrid", "Rome",
        "Amsterdam", "Brussels", "Vienna", "Warsaw", "Stockholm"
    ]
    building_locations = np.random.choice(locations, num_buildings)
    
    buildings_df = pd.DataFrame({
        'building_id': buildings,
        'building_type': building_types,
        'building_size_sqm': building_sizes,
        'location': building_locations
    })
    
    return buildings_df

def generate_sample_energy_data(buildings_df, num_months=12):
    """Generate sample energy consumption data"""
    np.random.seed(43)
    # Create date range for the past num_months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*num_months)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Initialize empty dataframe
    data_list = []
    
    # Create energy consumption data for each building and date
    for _, building in buildings_df.iterrows():
        for date in dates:
            # Energy consumption (kWh)
            energy = np.random.normal(building['building_size_sqm'] * 0.1, building['building_size_sqm'] * 0.02)
            energy_missing = np.random.random() < 0.1  # 10% chance of missing value
            
            # Add outliers occasionally
            if np.random.random() < 0.05:  # 5% chance of outlier
                energy = energy * np.random.uniform(3, 5)
            
            # Add data sources
            data_source = np.random.choice(['API', 'Manual Entry', 'BMS', 'Utility Provider', 'Invoice'])
            
            # Add row to data
            data_list.append({
                'building_id': building['building_id'],
                'date': date,
                'energy_consumption_kwh': None if energy_missing else energy,
                'data_source': data_source
            })
    
    energy_df = pd.DataFrame(data_list)
    return energy_df

def generate_sample_water_data(buildings_df, num_months=12):
    """Generate sample water consumption data"""
    np.random.seed(44)
    # Create date range for the past num_months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*num_months)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Initialize empty dataframe
    data_list = []
    
    # Create water consumption data for each building and date
    for _, building in buildings_df.iterrows():
        for date in dates:
            # Water consumption (m3)
            water = np.random.normal(building['building_size_sqm'] * 0.01, building['building_size_sqm'] * 0.005)
            water_missing = np.random.random() < 0.15  # 15% chance of missing value
            
            # Add outliers occasionally
            if np.random.random() < 0.05:  # 5% chance of outlier
                water = water * np.random.uniform(3, 5)
            
            # Add data sources
            data_source = np.random.choice(['API', 'Manual Entry', 'BMS', 'Utility Provider', 'Invoice'])
            
            # Add row to data
            data_list.append({
                'building_id': building['building_id'],
                'date': date,
                'water_consumption_m3': None if water_missing else water,
                'data_source': data_source
            })
    
    water_df = pd.DataFrame(data_list)
    return water_df

def generate_and_save_sample_data(output_dir="data"):
    """Generate and save all sample datasets"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    buildings_df = generate_sample_buildings()
    energy_df = generate_sample_energy_data(buildings_df)
    water_df = generate_sample_water_data(buildings_df)
    
    # Save to CSV
    buildings_df.to_csv(os.path.join(output_dir, "sample_buildings.csv"), index=False)
    energy_df.to_csv(os.path.join(output_dir, "sample_energy.csv"), index=False)
    water_df.to_csv(os.path.join(output_dir, "sample_water.csv"), index=False)
    
    return buildings_df, energy_df, water_df

if __name__ == "__main__":
    # Execute this script to generate sample data
    generate_and_save_sample_data()