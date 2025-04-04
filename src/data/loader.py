import pandas as pd
import os
from datetime import datetime

def load_csv_data(file_path):
    """
    Load data from CSV file
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_building_data(data_dir="data"):
    """
    Load building data from CSV
    """
    file_path = os.path.join(data_dir, "sample_buildings.csv")
    return load_csv_data(file_path)

def load_energy_data(data_dir="data"):
    """
    Load energy consumption data
    """
    file_path = os.path.join(data_dir, "sample_energy.csv")
    df = load_csv_data(file_path)
    if df is not None:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
    return df

def load_water_data(data_dir="data"):
    """
    Load water consumption data
    """
    file_path = os.path.join(data_dir, "sample_water.csv")
    df = load_csv_data(file_path)
    if df is not None:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
    return df

def merge_datasets(buildings_df, energy_df, water_df):
    """
    Merge building, energy, and water datasets
    """
    # Merge energy data with buildings
    merged_df = pd.merge(
        energy_df,
        buildings_df,
        on='building_id',
        how='left'
    )
    
    # Merge water data
    merged_df = pd.merge(
        merged_df,
        water_df.rename(columns={'data_source': 'water_data_source'}),
        on=['building_id', 'date'],
        how='outer'
    )
    
    return merged_df