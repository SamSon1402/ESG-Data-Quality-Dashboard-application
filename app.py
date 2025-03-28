import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="ESG Data Quality Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it futuristic, minimalistic with black and white theme
st.markdown("""
<style>
    /* Main background and text colors */
    .main, .block-container {
        background-color: #111111;
        color: #f0f0f0;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        font-weight: 300;
    }
    
    /* Card styling */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1c1c1c;
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 3px;
        border: none;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #cccccc;
        color: #000000;
    }
    
    /* Metric styling */
    div.css-12w0qpk.e1tzin5v1 {
        background-color: #1c1c1c;
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Plot background */
    .js-plotly-plot .plotly {
        background-color: #1c1c1c !important;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1c1c1c;
        color: #f0f0f0;
    }
    
    /* Widget labels */
    .css-2trqyj {
        color: #f0f0f0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ----- Helper Functions -----

def generate_sample_data(num_buildings=50, num_months=12):
    """Generate sample ESG data for demonstration"""
    np.random.seed(42)
    buildings = [f"Building_{i}" for i in range(1, num_buildings+1)]
    building_types = np.random.choice(['Office', 'Retail', 'Residential', 'Industrial', 'Mixed-Use'], num_buildings)
    building_sizes = np.random.randint(1000, 100000, num_buildings)
    
    # Create date range for the past num_months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*num_months)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Initialize empty dataframe
    data_list = []
    
    # Create metrics with different quality issues
    for i, building in enumerate(buildings):
        for date in dates:
            # Energy consumption (kWh)
            energy = np.random.normal(building_sizes[i] * 0.1, building_sizes[i] * 0.02)
            energy_missing = np.random.random() < 0.1  # 10% chance of missing value
            
            # Water consumption (m3)
            water = np.random.normal(building_sizes[i] * 0.01, building_sizes[i] * 0.005)
            water_missing = np.random.random() < 0.15  # 15% chance of missing value
            
            # CO2 emissions (kg)
            co2 = np.random.normal(energy * 0.2, energy * 0.05)
            co2_missing = np.random.random() < 0.2  # 20% chance of missing value
            
            # Waste (kg)
            waste = np.random.normal(building_sizes[i] * 0.005, building_sizes[i] * 0.001)
            waste_missing = np.random.random() < 0.25  # 25% chance of missing value
            
            # Add outliers occasionally
            if np.random.random() < 0.05:  # 5% chance of outlier
                energy = energy * np.random.uniform(3, 5)
            
            if np.random.random() < 0.05:  # 5% chance of outlier
                water = water * np.random.uniform(3, 5)
            
            # Add data sources
            data_source = np.random.choice(['API', 'Manual Entry', 'BMS', 'Utility Provider', 'Invoice'])
            
            # Add row to data
            data_list.append({
                'building_id': building,
                'building_type': building_types[i],
                'building_size_sqm': building_sizes[i],
                'date': date,
                'energy_consumption_kwh': None if energy_missing else energy,
                'water_consumption_m3': None if water_missing else water,
                'co2_emissions_kg': None if co2_missing else co2,
                'waste_kg': None if waste_missing else waste,
                'data_source': data_source
            })
    
    df = pd.DataFrame(data_list)
    return df

def calculate_data_quality_metrics(df):
    """Calculate data quality metrics for the dataset"""
    # Calculate completeness
    completeness = {}
    for col in ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']:
        completeness[col] = (1 - df[col].isna().mean()) * 100
    
    # Calculate outliers using IQR method
    outliers = {}
    for col in ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).mean() * 100
    
    # Calculate consistency by source
    consistency_by_source = {}
    for col in ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']:
        grouped = df.groupby('data_source')[col].std() / df.groupby('data_source')[col].mean()
        consistency_by_source[col] = grouped.to_dict()
    
    # Overall quality score (simplified)
    quality_scores = {}
    for col in ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']:
        # Weighted score: completeness (40%), outliers (30%), consistency (30%)
        # For consistency, use the average coefficient of variation across sources
        avg_consistency = np.mean([v for v in consistency_by_source[col].values() if not np.isnan(v)])
        consistency_score = 100 * (1 - min(avg_consistency, 1))  # Lower variation is better
        
        quality_scores[col] = (
            0.4 * completeness[col] + 
            0.3 * (100 - outliers[col]) +
            0.3 * consistency_score
        )
    
    # Prepare completeness over time
    completeness_over_time = df.pivot_table(
        index='date',
        values=['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg'],
        aggfunc=lambda x: 100 * (1 - np.mean(pd.isna(x)))
    )
    
    return {
        'completeness': completeness,
        'outliers': outliers,
        'consistency_by_source': consistency_by_source,
        'quality_scores': quality_scores,
        'completeness_over_time': completeness_over_time
    }

def generate_recommendations(quality_metrics):
    """Generate recommendations based on quality metrics"""
    recommendations = []
    
    # Check completeness
    for metric, value in quality_metrics['completeness'].items():
        if value < 90:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Low data completeness for {metric_name}",
                'description': f"Only {value:.1f}% of {metric_name} data is complete.",
                'impact': 'HIGH' if value < 80 else 'MEDIUM',
                'recommendation': f"Implement automated data collection for {metric_name} or follow up with data providers."
            })
    
    # Check outliers
    for metric, value in quality_metrics['outliers'].items():
        if value > 5:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"High number of outliers in {metric_name}",
                'description': f"{value:.1f}% of {metric_name} data points are outliers.",
                'impact': 'HIGH' if value > 10 else 'MEDIUM',
                'recommendation': f"Review {metric_name} data collection process and implement validation rules."
            })
    
    # Check quality scores
    for metric, value in quality_metrics['quality_scores'].items():
        if value < 80:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Low overall quality score for {metric_name}",
                'description': f"The quality score for {metric_name} is {value:.1f}/100.",
                'impact': 'HIGH' if value < 70 else 'MEDIUM',
                'recommendation': f"Comprehensive review of {metric_name} data collection and validation processes needed."
            })
    
    return recommendations

# ----- Main Application -----

def main():
    # Sidebar
    with st.sidebar:
        st.title("🏢 ESG Data Quality Dashboard")
        st.markdown("### Navigation")
        page = st.radio("", ["Overview", "Data Quality Analysis", "Trends", "Recommendations"])
        
        st.markdown("---")
        st.markdown("### Filters")
        st.markdown("##### Date Range")
        
        # Create dummy data if not already in session state
        if 'data' not in st.session_state:
            st.session_state.data = generate_sample_data()
        
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
        filtered_data = st.session_state.data.copy()
        
        # Date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[(filtered_data['date'].dt.date >= start_date) & 
                                         (filtered_data['date'].dt.date <= end_date)]
        
        # Building type filter
        if selected_building_type != 'All':
            filtered_data = filtered_data[filtered_data['building_type'] == selected_building_type]
        
        # Data source filter
        if selected_data_source != 'All':
            filtered_data = filtered_data[filtered_data['data_source'] == selected_data_source]
        
        # Store filtered data in session state
        st.session_state.filtered_data = filtered_data
        
        # Calculate quality metrics
        st.session_state.quality_metrics = calculate_data_quality_metrics(filtered_data)
        
        # Generate recommendations
        st.session_state.recommendations = generate_recommendations(st.session_state.quality_metrics)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This dashboard analyzes the quality and completeness of ESG data across your real estate portfolio.")
        st.markdown("Created by **Sameer M**")
    
    # Main content
    if page == "Overview":
        display_overview()
    elif page == "Data Quality Analysis":
        display_data_quality_analysis()
    elif page == "Trends":
        display_trends()
    elif page == "Recommendations":
        display_recommendations()

def display_overview():
    st.title("ESG Data Quality Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    overall_score = np.mean(list(st.session_state.quality_metrics['quality_scores'].values()))
    avg_completeness = np.mean(list(st.session_state.quality_metrics['completeness'].values()))
    avg_outliers = np.mean(list(st.session_state.quality_metrics['outliers'].values()))
    
    col1.metric("Overall Quality Score", f"{overall_score:.1f}/100")
    col2.metric("Data Completeness", f"{avg_completeness:.1f}%")
    col3.metric("Problematic Outliers", f"{avg_outliers:.1f}%")
    col4.metric("Critical Issues", f"{sum([1 for r in st.session_state.recommendations if r['impact'] == 'HIGH'])}")
    
    # Quality score gauge charts
    st.subheader("Quality Scores by ESG Category")
    
    fig = go.Figure()
    categories = []
    scores = []
    
    for metric, score in st.session_state.quality_metrics['quality_scores'].items():
        category = metric.split('_')[0].capitalize()
        categories.append(category)
        scores.append(score)
    
    fig.add_trace(go.Bar(
        x=categories,
        y=scores,
        marker_color=['#ffffff' if score >= 80 else '#aaaaaa' if score >= 60 else '#666666' for score in scores],
        text=[f"{score:.1f}" for score in scores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title_text="Data Quality Score by Category (out of 100)",
        xaxis_title="Category",
        yaxis_title="Quality Score",
        yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Building data quality map
    st.subheader("Building Portfolio Quality Map")
    
    # Calculate average quality score per building
    building_quality = st.session_state.filtered_data.groupby('building_id').apply(
        lambda x: pd.Series({
            'energy_quality': 100 * (1 - x['energy_consumption_kwh'].isna().mean()),
            'water_quality': 100 * (1 - x['water_consumption_m3'].isna().mean()),
            'co2_quality': 100 * (1 - x['co2_emissions_kg'].isna().mean()),
            'waste_quality': 100 * (1 - x['waste_kg'].isna().mean()),
            'building_size': x['building_size_sqm'].iloc[0],
            'building_type': x['building_type'].iloc[0]
        })
    )
    
    # Calculate average score
    building_quality['avg_quality'] = building_quality[['energy_quality', 'water_quality', 'co2_quality', 'waste_quality']].mean(axis=1)
    
    # Plot bubble chart
    fig = px.scatter(
        building_quality.reset_index(),
        x="building_size",
        y="avg_quality",
        size="building_size",
        color="avg_quality",
        color_continuous_scale=["#666666", "#888888", "#aaaaaa", "#cccccc", "#ffffff"],
        hover_name="building_id",
        hover_data=["building_type", "energy_quality", "water_quality", "co2_quality", "waste_quality"],
        size_max=50,
        opacity=0.7
    )
    
    fig.update_layout(
        title="Building Data Quality vs Size",
        xaxis_title="Building Size (sqm)",
        yaxis_title="Average Data Quality Score (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent quality issues
    st.subheader("Top Data Quality Issues")
    
    with st.container():
        for i, recommendation in enumerate(st.session_state.recommendations[:3]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{recommendation['issue']}**")
                st.markdown(recommendation['description'])
            with col2:
                st.markdown(f"**Impact: {recommendation['impact']}**")

def display_data_quality_analysis():
    st.title("ESG Data Quality Analysis")
    
    # Create tabs for different ESG categories
    tabs = st.tabs(["Energy", "Water", "CO2 Emissions", "Waste"])
    
    metrics = [
        {'id': 'energy_consumption_kwh', 'name': 'Energy', 'unit': 'kWh'},
        {'id': 'water_consumption_m3', 'name': 'Water', 'unit': 'm³'},
        {'id': 'co2_emissions_kg', 'name': 'CO2 Emissions', 'unit': 'kg'},
        {'id': 'waste_kg', 'name': 'Waste', 'unit': 'kg'}
    ]
    
    for i, tab in enumerate(tabs):
        with tab:
            metric = metrics[i]
            metric_id = metric['id']
            metric_name = metric['name']
            metric_unit = metric['unit']
            
            # Create columns for scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Completeness Score", 
                    f"{st.session_state.quality_metrics['completeness'][metric_id]:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Outlier Score", 
                    f"{100 - st.session_state.quality_metrics['outliers'][metric_id]:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Overall Quality Score", 
                    f"{st.session_state.quality_metrics['quality_scores'][metric_id]:.1f}/100",
                    delta=None
                )
            
            # Data completeness by source
            st.subheader(f"{metric_name} Data Completeness by Source")
            
            # Calculate completeness by source
            completeness_by_source = st.session_state.filtered_data.groupby('data_source')[metric_id].apply(
                lambda x: 100 * (1 - x.isna().mean())
            ).reset_index()
            completeness_by_source.columns = ['Data Source', 'Completeness (%)']
            
            # Plot bar chart
            fig = px.bar(
                completeness_by_source, 
                x='Data Source', 
                y='Completeness (%)',
                color='Completeness (%)',
                color_continuous_scale=["#666666", "#888888", "#aaaaaa", "#cccccc", "#ffffff"],
                text_auto=True
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis_range=[0, 100],
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of values
            st.subheader(f"{metric_name} Data Distribution")
            
            # Plot histogram
            fig = px.histogram(
                st.session_state.filtered_data,
                x=metric_id,
                nbins=30,
                marginal="box",
                opacity=0.7,
                color_discrete_sequence=["#ffffff"]
            )
            
            fig.update_layout(
                xaxis_title=f"{metric_name} ({metric_unit})",
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing data calendar
            st.subheader(f"{metric_name} Missing Data Calendar")
            
            # Prepare data
            missing_data = st.session_state.filtered_data.pivot_table(
                index='date',
                columns='building_id',
                values=metric_id,
                aggfunc=lambda x: x.isna().mean()
            ).fillna(0)
            
            # Calculate building average completion rate to sort
            building_completion = missing_data.mean().sort_values(ascending=False)
            missing_data = missing_data[building_completion.index[:15]]  # Top 15 buildings
            
            # Convert to long format for heatmap
            missing_data_long = missing_data.reset_index().melt(
                id_vars='date',
                var_name='Building',
                value_name='Missing Data Rate'
            )
            
            # Plot heatmap
            fig = px.density_heatmap(
                missing_data_long,
                x='date',
                y='Building',
                z='Missing Data Rate',
                color_continuous_scale=["#ffffff", "#aaaaaa", "#666666", "#333333"],
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Building",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_trends():
    st.title("Data Quality Trends")
    
    # Completeness over time
    st.subheader("Data Completeness Trends")
    
    # Plot line chart for completeness over time
    completeness_df = st.session_state.quality_metrics['completeness_over_time'].reset_index()
    
    # Rename columns for better display
    completeness_df.columns = ['Date', 'Energy', 'Water', 'CO2', 'Waste']
    
    # Melt dataframe for plotting
    completeness_long = completeness_df.melt(
        id_vars='Date',
        var_name='Metric',
        value_name='Completeness (%)'
    )
    
    # Plot
    fig = px.line(
        completeness_long,
        x='Date',
        y='Completeness (%)',
        color='Metric',
        color_discrete_sequence=["#ffffff", "#cccccc", "#aaaaaa", "#888888"],
        markers=True
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data source quality over time
    st.subheader("Data Source Quality Trends")
    
    # Calculate quality score by source and month
    source_quality = st.session_state.filtered_data.copy()
    source_quality['month'] = source_quality['date'].dt.to_period('M')
    
    # Create quality score (simplified as completeness for now)
    metrics = ['energy_consumption_kwh', 'water_consumption_m3', 'co2_emissions_kg', 'waste_kg']
    for metric in metrics:
        source_quality[f'{metric}_quality'] = source_quality[metric].notna().astype(int) * 100
    
    # Calculate average quality by source and month
    source_quality_agg = source_quality.groupby(['month', 'data_source'])[
        [f'{m}_quality' for m in metrics]
    ].mean().reset_index()
    
    # Calculate overall quality
    source_quality_agg['overall_quality'] = source_quality_agg[[f'{m}_quality' for m in metrics]].mean(axis=1)
    
    # Convert month to datetime for plotting
    source_quality_agg['month'] = source_quality_agg['month'].dt.to_timestamp()
    
    # Plot
    fig = px.line(
        source_quality_agg,
        x='month',
        y='overall_quality',
        color='data_source',
        color_discrete_sequence=["#ffffff", "#dddddd", "#bbbbbb", "#999999", "#777777"],
        markers=True
    )
    
    fig.update_layout(
        title="Data Quality by Source Over Time",
        xaxis_title="Month",
        yaxis_title="Average Quality Score (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Building type quality comparison
    st.subheader("Data Quality by Building Type")
    
    # Calculate quality by building type
    building_type_quality = st.session_state.filtered_data.copy()
    
    # Create quality score (simplified as completeness for now)
    for metric in metrics:
        building_type_quality[f'{metric}_quality'] = building_type_quality[metric].notna().astype(int) * 100
    
    # Calculate average quality by building type
    building_type_quality_agg = building_type_quality.groupby('building_type')[
        [f'{m}_quality' for m in metrics]
    ].mean().reset_index()
    
    # Melt for plotting
    building_type_quality_long = building_type_quality_agg.melt(
        id_vars='building_type',
        value_vars=[f'{m}_quality' for m in metrics],
        var_name='metric',
        value_name='quality'
    )
    
    # Clean up metric names
    building_type_quality_long['metric'] = building_type_quality_long['metric'].str.replace('_quality', '').str.replace('_consumption_kwh', '').str.replace('_consumption_m3', '').str.replace('_emissions_kg', '').str.replace('_kg', '')
    building_type_quality_long['metric'] = building_type_quality_long['metric'].str.capitalize()
    
    # Plot
    fig = px.bar(
        building_type_quality_long,
        x='building_type',
        y='quality',
        color='metric',
        barmode='group',
        color_discrete_sequence=["#ffffff", "#cccccc", "#aaaaaa", "#888888"],
        text_auto=True
    )
    
    fig.update_layout(
        title="Data Quality by Building Type and ESG Category",
        xaxis_title="Building Type",
        yaxis_title="Quality Score (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="ESG Category",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations():
    st.title("Data Quality Recommendations")
    
    # Create tabs for different recommendation categories
    tabs = st.tabs(["Critical Issues", "Medium Priority", "All Recommendations"])
    
    # Filter recommendations by priority
    critical_recommendations = [r for r in st.session_state.recommendations if r['impact'] == 'HIGH']
    medium_recommendations = [r for r in st.session_state.recommendations if r['impact'] == 'MEDIUM']
    
    # Critical Issues Tab
    with tabs[0]:
        if critical_recommendations:
            for i, recommendation in enumerate(critical_recommendations):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {i+1}. {recommendation['issue']}")
                        st.markdown(f"**Description:** {recommendation['description']}")
                        st.markdown(f"**Recommendation:** {recommendation['recommendation']}")
                    with col2:
                        # Display impact with color coding
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #ff4444;
                                padding: 10px;
                                border-radius: 5px;
                                text-align: center;
                                color: white;
                                margin-top: 20px;
                            ">
                            <strong>HIGH IMPACT</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("---")
        else:
            st.markdown("### No critical issues found.")
            st.markdown("All ESG data quality metrics are within acceptable ranges. Continue monitoring for any changes.")
    
    # Medium Priority Tab
    with tabs[1]:
        if medium_recommendations:
            for i, recommendation in enumerate(medium_recommendations):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {i+1}. {recommendation['issue']}")
                        st.markdown(f"**Description:** {recommendation['description']}")
                        st.markdown(f"**Recommendation:** {recommendation['recommendation']}")
                    with col2:
                        # Display impact with color coding
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #ff9900;
                                padding: 10px;
                                border-radius: 5px;
                                text-align: center;
                                color: white;
                                margin-top: 20px;
                            ">
                            <strong>MEDIUM IMPACT</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("---")
        else:
            st.markdown("### No medium priority issues found.")
            st.markdown("All ESG data quality metrics are within acceptable ranges. Continue monitoring for any changes.")
    
    # All Recommendations Tab
    with tabs[2]:
        if st.session_state.recommendations:
            for i, recommendation in enumerate(st.session_state.recommendations):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {i+1}. {recommendation['issue']}")
                        st.markdown(f"**Description:** {recommendation['description']}")
                        st.markdown(f"**Recommendation:** {recommendation['recommendation']}")
                    with col2:
                        # Display impact with color coding
                        color = "#ff4444" if recommendation['impact'] == 'HIGH' else "#ff9900"
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {color};
                                padding: 10px;
                                border-radius: 5px;
                                text-align: center;
                                color: white;
                                margin-top: 20px;
                            ">
                            <strong>{recommendation['impact']} IMPACT</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("---")
        else:
            st.markdown("### No recommendations found.")
            st.markdown("All ESG data quality metrics are within acceptable ranges. Continue monitoring for any changes.")
    
    # Action plan generator
    st.subheader("Generate Action Plan")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Create a customized action plan to address the identified data quality issues.")
        priority = st.selectbox("Priority Level", ["Critical Issues Only", "Medium and Critical Issues", "All Issues"])
    
    with col2:
        st.markdown("&nbsp;")  # Add some spacing
        if st.button("Generate Action Plan", key="generate_plan"):
            st.session_state.show_action_plan = True
    
    if 'show_action_plan' in st.session_state and st.session_state.show_action_plan:
        st.markdown("---")
        st.markdown("## ESG Data Quality Action Plan")
        st.markdown("### Executive Summary")
        
        if priority == "Critical Issues Only":
            selected_recommendations = critical_recommendations
        elif priority == "Medium and Critical Issues":
            selected_recommendations = critical_recommendations + medium_recommendations
        else:
            selected_recommendations = st.session_state.recommendations
        
        st.markdown(f"This action plan addresses {len(selected_recommendations)} data quality issues identified in the ESG data assessment.")
        
        if selected_recommendations:
            st.markdown("### Implementation Timeline")
            
            timeline_data = []
            for i, rec in enumerate(selected_recommendations):
                # Create a simple timeline based on priority
                if rec['impact'] == 'HIGH':
                    start_day = i * 2 + 1
                    duration = 7
                else:
                    start_day = i * 2 + 10
                    duration = 14
                
                timeline_data.append({
                    'Task': rec['issue'],
                    'Start': start_day,
                    'Duration': duration,
                    'Priority': rec['impact']
                })
            
            # Create a Gantt chart
            fig = px.timeline(
                timeline_data, 
                x_start="Start", 
                x_end=timeline_data['Start'] + timeline_data['Duration'], 
                y="Task",
                color="Priority",
                color_discrete_map={"HIGH": "#ff4444", "MEDIUM": "#ff9900"},
            )
            
            fig.update_layout(
                title="Implementation Timeline (days)",
                xaxis_title="Days",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=60, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Detailed Implementation Steps")
            
            for i, recommendation in enumerate(selected_recommendations):
                st.markdown(f"#### {i+1}. {recommendation['issue']}")
                st.markdown(f"**Description:** {recommendation['description']}")
                st.markdown(f"**Action Items:**")
                
                # Generate dummy action items based on the recommendation
                if "completeness" in recommendation['issue'].lower():
                    st.markdown("1. Identify data sources with missing values")
                    st.markdown("2. Contact data providers to establish regular data delivery")
                    st.markdown("3. Implement automated validation checks for incoming data")
                    st.markdown("4. Create alert system for missing data points")
                elif "outliers" in recommendation['issue'].lower():
                    st.markdown("1. Review current data validation rules")
                    st.markdown("2. Implement statistical checks for anomaly detection")
                    st.markdown("3. Create reporting process for detected outliers")
                    st.markdown("4. Establish correction procedures for confirmed outliers")
                else:
                    st.markdown("1. Conduct comprehensive review of data collection process")
                    st.markdown("2. Standardize data formats across all sources")
                    st.markdown("3. Implement regular quality audits")
                    st.markdown("4. Establish data governance policy")
                
                st.markdown(f"**Timeline:** {'1-2 weeks' if recommendation['impact'] == 'HIGH' else '3-4 weeks'}")
                st.markdown(f"**Responsible Team:** {'Data Quality Team' if i % 2 == 0 else 'ESG Analytics Team'}")
                st.markdown("---")
        else:
            st.markdown("No issues to address in the action plan.")

if __name__ == "__main__":
    main()