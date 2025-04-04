# Data processing logic
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """Basic data cleaning"""
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Convert date columns to datetime if they exist
    date_columns = [col for col in df_cleaned.columns if 'date' in col.lower()]
    for col in date_columns:
        if df_cleaned[col].dtype != 'datetime64[ns]':
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
    
    return df_cleaned

def prepare_esg_data(buildings_df, energy_df, water_df):
    """
    Prepare ESG data for analysis by merging and transforming datasets
    """
    # Merge datasets
    merged_df = pd.merge(
        energy_df,
        buildings_df[['building_id', 'building_type', 'building_size_sqm']],
        on='building_id',
        how='left'
    )
    
    merged_df = pd.merge(
        merged_df,
        water_df[['building_id', 'date', 'water_consumption_m3']],
        on=['building_id', 'date'],
        how='outer'
    )
    
    # Calculate CO2 emissions (simplified calculation)
    merged_df['co2_emissions_kg'] = merged_df['energy_consumption_kwh'] * 0.2
    
    # Calculate waste (simplified placeholder)
    merged_df['waste_kg'] = merged_df['building_size_sqm'] * 0.005 * np.random.normal(1, 0.2, merged_df.shape[0])
    
    # Add some missing values for demonstration
    sample_indices = np.random.choice(merged_df.index, size=int(merged_df.shape[0] * 0.2), replace=False)
    merged_df.loc[sample_indices, 'co2_emissions_kg'] = np.nan
    
    sample_indices = np.random.choice(merged_df.index, size=int(merged_df.shape[0] * 0.25), replace=False)
    merged_df.loc[sample_indices, 'waste_kg'] = np.nan
    
    return merged_df

def normalize_esg_metrics(df):
    """
    Normalize ESG metrics by building size
    """
    df_normalized = df.copy()
    
    # Energy intensity (kWh/m²)
    df_normalized['energy_intensity'] = df_normalized['energy_consumption_kwh'] / df_normalized['building_size_sqm']
    
    # Water intensity (m³/m²)
    df_normalized['water_intensity'] = df_normalized['water_consumption_m3'] / df_normalized['building_size_sqm']
    
    # CO2 intensity (kg/m²)
    df_normalized['co2_intensity'] = df_normalized['co2_emissions_kg'] / df_normalized['building_size_sqm']
    
    # Waste intensity (kg/m²)
    df_normalized['waste_intensity'] = df_normalized['waste_kg'] / df_normalized['building_size_sqm']
    
    return df_normalized