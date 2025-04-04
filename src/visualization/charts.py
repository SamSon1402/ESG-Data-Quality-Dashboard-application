# Chart generation functions
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_quality_score_chart(quality_scores):
    """
    Create a bar chart of quality scores by category
    """
    categories = []
    scores = []
    
    for metric, score in quality_scores.items():
        category = metric.split('_')[0].capitalize()
        categories.append(category)
        scores.append(score)
    
    fig = go.Figure()
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
    
    return fig

def create_building_quality_map(df):
    """
    Create a bubble chart showing building quality vs size
    """
    # Calculate average quality score per building
    building_quality = df.groupby('building_id').apply(
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
    
    return fig

def create_completeness_by_source_chart(df, metric_id):
    """
    Create a bar chart showing data completeness by source
    """
    # Calculate completeness by source
    completeness_by_source = df.groupby('data_source')[metric_id].apply(
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
    
    return fig

def create_metric_distribution_chart(df, metric_id, metric_name, metric_unit):
    """
    Create a histogram with box plot showing distribution of values
    """
    # Plot histogram
    fig = px.histogram(
        df,
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
    
    return fig

def create_missing_data_calendar(df, metric_id):
    """
    Create a heatmap showing missing data patterns
    """
    # Prepare data
    missing_data = df.pivot_table(
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
    
    return fig

def create_completeness_trend_chart(completeness_over_time):
    """
    Create a line chart showing data completeness trends over time
    """
    # Prepare data
    completeness_df = completeness_over_time.reset_index()
    
    # Rename columns for better display
    renamed_columns = {col: col.split('_')[0].capitalize() for col in completeness_df.columns if col != 'date'}
    completeness_df = completeness_df.rename(columns=renamed_columns)
    
    # Melt dataframe for plotting
    metrics = [col for col in completeness_df.columns if col != 'date']
    completeness_long = completeness_df.melt(
        id_vars='date',
        value_vars=metrics,
        var_name='Metric',
        value_name='Completeness (%)'
    )
    
    # Plot
    fig = px.line(
        completeness_long,
        x='date',
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
    
    return fig

def create_action_plan_timeline(timeline_data):
    """
    Create a Gantt chart for the action plan timeline
    """
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
    
    return fig