# Dashboard layout components
import streamlit as st

def display_overview_dashboard(quality_metrics, recommendations, filtered_data):
    """
    Display the overview dashboard
    """
    from src.visualization.charts import create_quality_score_chart, create_building_quality_map
    
    st.title("ESG Data Quality Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    overall_score = sum(quality_metrics['quality_scores'].values()) / len(quality_metrics['quality_scores'])
    avg_completeness = sum(quality_metrics['completeness'].values()) / len(quality_metrics['completeness'])
    avg_outliers = sum(quality_metrics['outliers'].values()) / len(quality_metrics['outliers'])
    
    col1.metric("Overall Quality Score", f"{overall_score:.1f}/100")
    col2.metric("Data Completeness", f"{avg_completeness:.1f}%")
    col3.metric("Problematic Outliers", f"{avg_outliers:.1f}%")
    col4.metric("Critical Issues", f"{sum([1 for r in recommendations if r['impact'] == 'HIGH'])}")
    
    # Quality score gauge charts
    st.subheader("Quality Scores by ESG Category")
    fig = create_quality_score_chart(quality_metrics['quality_scores'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Building data quality map
    st.subheader("Building Portfolio Quality Map")
    fig = create_building_quality_map(filtered_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent quality issues
    st.subheader("Top Data Quality Issues")
    
    with st.container():
        for i, recommendation in enumerate(recommendations[:3]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{recommendation['issue']}**")
                st.markdown(recommendation['description'])
            with col2:
                st.markdown(f"**Impact: {recommendation['impact']}**")

def display_data_quality_analysis(filtered_data, quality_metrics):
    """
    Display the data quality analysis dashboard
    """
    from src.visualization.charts import (
        create_completeness_by_source_chart,
        create_metric_distribution_chart,
        create_missing_data_calendar
    )
    
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
                    f"{quality_metrics['completeness'][metric_id]:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Outlier Score", 
                    f"{100 - quality_metrics['outliers'][metric_id]:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Overall Quality Score", 
                    f"{quality_metrics['quality_scores'][metric_id]:.1f}/100",
                    delta=None
                )
            
            # Data completeness by source
            st.subheader(f"{metric_name} Data Completeness by Source")
            fig = create_completeness_by_source_chart(filtered_data, metric_id)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of values
            st.subheader(f"{metric_name} Data Distribution")
            fig = create_metric_distribution_chart(filtered_data, metric_id, metric_name, metric_unit)
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing data calendar
            st.subheader(f"{metric_name} Missing Data Calendar")
            fig = create_missing_data_calendar(filtered_data, metric_id)
            st.plotly_chart(fig, use_container_width=True)

def display_trends_dashboard(filtered_data, quality_metrics):
    """
    Display the trends dashboard
    """
    from src.visualization.charts import create_completeness_trend_chart
    
    st.title("Data Quality Trends")
    
    # Completeness over time
    st.subheader("Data Completeness Trends")
    fig = create_completeness_trend_chart(quality_metrics['completeness_over_time'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Data source quality over time
    st.subheader("Data Source Quality Trends")
    
    # Calculate quality score by source and month
    source_quality = filtered_data.copy()
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

def display_recommendations_dashboard(recommendations):
    """
    Display the recommendations dashboard
    """
    from src.analysis.recommendations import generate_action_plan
    from src.visualization.charts import create_action_plan_timeline
    
    st.title("Data Quality Recommendations")
    
    # Create tabs for different recommendation categories
    tabs = st.tabs(["Critical Issues", "Medium Priority", "All Recommendations"])
    
    # Filter recommendations by priority
    critical_recommendations = [r for r in recommendations if r['impact'] == 'HIGH']
    medium_recommendations = [r for r in recommendations if r['impact'] == 'MEDIUM']
    
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
        if recommendations:
            for i, recommendation in enumerate(recommendations):
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
        # Map priority selection to parameter for generate_action_plan
        priority_param = "critical" if priority == "Critical Issues Only" else (
            "medium" if priority == "Medium and Critical Issues" else "all"
        )
        
        # Generate action plan
        action_plan = generate_action_plan(recommendations, priority_level=priority_param)
        
        st.markdown("---")
        st.markdown("## ESG Data Quality Action Plan")
        st.markdown("### Executive Summary")
        st.markdown(action_plan['summary'])
        
        if action_plan['timeline_data']:
            st.markdown("### Implementation Timeline")
            fig = create_action_plan_timeline(pd.DataFrame(action_plan['timeline_data']))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Detailed Implementation Steps")
            
            for i, step in enumerate(action_plan['implementation_steps']):
                recommendation = step['recommendation']
                st.markdown(f"#### {i+1}. {recommendation['issue']}")
                st.markdown(f"**Description:** {recommendation['description']}")
                st.markdown(f"**Action Items:**")
                
                for j, action_item in enumerate(step['steps']):
                    st.markdown(f"{j+1}. {action_item}")
                
                st.markdown(f"**Timeline:** {step['timeline']}")
                st.markdown(f"**Responsible Team:** {step['responsible_team']}")
                st.markdown("---")
        else:
            st.markdown("No issues to address in the action plan.")