# Overview dashboard
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_csv_data
from src.data.generator import generate_sample_data
from src.analysis.quality_metrics import calculate_all_quality_metrics
from src.analysis.recommendations import generate_recommendations
from src.visualization.dashboards import display_overview_dashboard
from src.utils.styling import apply_custom_css

# Set page configuration
st.set_page_config(
    page_title="ESG Data Quality Dashboard - Overview",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# App state initialization
if 'data' not in st.session_state:
    try:
        # Try to load data from CSV
        buildings_df = load_csv_data("data/sample_buildings.csv")
        energy_df = load_csv_data("data/sample_energy.csv")
        water_df = load_csv_data("data/sample_water.csv")
        
        # If any data is missing, generate sample data
        if buildings_df is None or energy_df is None or water_df is None:
            st.session_state.data = generate_sample_data()
        else:
            # Convert date columns to datetime
            if 'date' in energy_df.columns:
                energy_df['date'] = pd.to_datetime(energy_df['date'])
            if 'date' in water_df.columns:
                water_df['date'] = pd.to_datetime(water_df['date'])
            
            # Merge datasets
            merged_df = pd.merge(
                energy_df,
                buildings_df,
                on='building_id',
                how='left'
            )
            
            merged_df = pd.merge(
                merged_df,
                water_df,
                on=['building_id', 'date'],
                how='outer',
                suffixes=('', '_water')
            )
            
            # Generate CO2 and waste data (placeholder)
            merged_df['co2_emissions_kg'] = merged_df['energy_consumption_kwh'] * 0.2
            merged_df['waste_kg'] = merged_df['building_size_sqm'] * 0.005 * np.random.normal(1, 0.2, merged_df.shape[0])
            
            # Add some missing values for demonstration
            sample_indices = np.random.choice(merged_df.index, size=int(merged_df.shape[0] * 0.2), replace=False)
            merged_df.loc[sample_indices, 'co2_emissions_kg'] = np.nan
            
            sample_indices = np.random.choice(merged_df.index, size=int(merged_df.shape[0] * 0.25), replace=False)
            merged_df.loc[sample_indices, 'waste_kg'] = np.nan
            
            st.session_state.data = merged_df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.session_state.data = generate_sample_data()

# Sidebar
with st.sidebar:
    st.title("🏢 ESG Data Quality Dashboard")
    st.markdown("### Overview")
    st.markdown("This page provides a high-level overview of the ESG data quality across your real estate portfolio.")
    
    st.markdown("---")
    st.markdown("### Filters")
    st.markdown("##### Date Range")
    
    # Get min and max dates from data
    min_date = st.session_state.data['date'].min().date()
    max_date = st.session_state.data['date'].max().date()
    
    # Date filters
    date_range = st.date_input(
        "",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Building type filter
    building_types = ['All'] + list(st.session_state.data['building_type'].unique())
    selected_building_type = st.selectbox("Building Type", building_types)
    
    # Data source filter
    data_sources = ['All'] + list(st.session_state.data['data_source'].unique())
    selected_data_source = st.selectbox("Data Source", data_sources)
    
    # Apply filters
    from src.utils.helpers import filter_dataframe
    
    filtered_data = filter_dataframe(
        st.session_state.data,
        date_range=date_range,
        building_type=selected_building_type,
        data_source=selected_data_source
    )
    
    # Calculate quality metrics
    quality_metrics = calculate_all_quality_metrics(filtered_data)
    
    # Generate recommendations
    recommendations = generate_recommendations(quality_metrics)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This dashboard analyzes the quality and completeness of ESG data across your real estate portfolio.")
    st.markdown("Created by **Sameer M**")

# Main content - Overview Dashboard
display_overview_dashboard(quality_metrics, recommendations, filtered_data)