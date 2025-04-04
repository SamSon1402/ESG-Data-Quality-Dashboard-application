# Recommendation generation
import pandas as pd
import numpy as np

def generate_recommendations(quality_metrics, threshold_critical=80, threshold_medium=90):
    """
    Generate recommendations based on quality metrics
    
    Parameters:
    -----------
    quality_metrics : dict
        Dictionary with quality metrics
    threshold_critical : float, default 80
        Threshold for critical issues (below this is critical)
    threshold_medium : float, default 90
        Threshold for medium issues (below this is medium, above is good)
        
    Returns:
    --------
    list
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Check completeness
    for metric, value in quality_metrics['completeness'].items():
        if value < threshold_critical:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Low data completeness for {metric_name}",
                'description': f"Only {value:.1f}% of {metric_name} data is complete.",
                'impact': 'HIGH',
                'recommendation': f"Implement automated data collection for {metric_name} or follow up with data providers."
            })
        elif value < threshold_medium:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Moderate data completeness for {metric_name}",
                'description': f"Only {value:.1f}% of {metric_name} data is complete.",
                'impact': 'MEDIUM',
                'recommendation': f"Review {metric_name} data collection process and identify improvement opportunities."
            })
    
    # Check outliers
    for metric, value in quality_metrics['outliers'].items():
        if value > 10:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"High number of outliers in {metric_name}",
                'description': f"{value:.1f}% of {metric_name} data points are outliers.",
                'impact': 'HIGH',
                'recommendation': f"Review {metric_name} data collection process and implement validation rules."
            })
        elif value > 5:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Moderate number of outliers in {metric_name}",
                'description': f"{value:.1f}% of {metric_name} data points are outliers.",
                'impact': 'MEDIUM',
                'recommendation': f"Implement outlier detection for {metric_name} data."
            })
    
    # Check quality scores
    for metric, value in quality_metrics['quality_scores'].items():
        if value < threshold_critical:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Low overall quality score for {metric_name}",
                'description': f"The quality score for {metric_name} is {value:.1f}/100.",
                'impact': 'HIGH',
                'recommendation': f"Comprehensive review of {metric_name} data collection and validation processes needed."
            })
        elif value < threshold_medium:
            metric_name = metric.replace('_', ' ').replace('consumption', '').replace('emissions', '')
            recommendations.append({
                'issue': f"Moderate overall quality score for {metric_name}",
                'description': f"The quality score for {metric_name} is {value:.1f}/100.",
                'impact': 'MEDIUM',
                'recommendation': f"Review {metric_name} data collection process and implement quality checks."
            })
    
    # Deduplicate recommendations (remove duplicates with the same issue)
    unique_recommendations = []
    seen_issues = set()
    
    for rec in recommendations:
        if rec['issue'] not in seen_issues:
            unique_recommendations.append(rec)
            seen_issues.add(rec['issue'])
    
    return unique_recommendations

def generate_action_plan(recommendations, priority_level="all"):
    """
    Generate an action plan based on recommendations
    
    Parameters:
    -----------
    recommendations : list
        List of recommendation dictionaries
    priority_level : str, default "all"
        Priority level to include in the action plan
        Options: "critical", "medium", "all"
        
    Returns:
    --------
    dict
        Action plan details
    """
    # Filter recommendations based on priority level
    if priority_level == "critical":
        selected_recommendations = [r for r in recommendations if r['impact'] == 'HIGH']
    elif priority_level == "medium":
        selected_recommendations = [r for r in recommendations if r['impact'] in ['HIGH', 'MEDIUM']]
    else:
        selected_recommendations = recommendations
    
    # Create timeline data
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
    
    # Generate implementation steps for each recommendation
    implementation_steps = []
    for i, recommendation in enumerate(selected_recommendations):
        steps = []
        
        # Generate steps based on the recommendation type
        if "completeness" in recommendation['issue'].lower():
            steps = [
                "Identify data sources with missing values",
                "Contact data providers to establish regular data delivery",
                "Implement automated validation checks for incoming data",
                "Create alert system for missing data points"
            ]
        elif "outliers" in recommendation['issue'].lower():
            steps = [
                "Review current data validation rules",
                "Implement statistical checks for anomaly detection",
                "Create reporting process for detected outliers",
                "Establish correction procedures for confirmed outliers"
            ]
        else:
            steps = [
                "Conduct comprehensive review of data collection process",
                "Standardize data formats across all sources",
                "Implement regular quality audits",
                "Establish data governance policy"
            ]
        
        implementation_steps.append({
            'recommendation': recommendation,
            'steps': steps,
            'timeline': '1-2 weeks' if recommendation['impact'] == 'HIGH' else '3-4 weeks',
            'responsible_team': 'Data Quality Team' if i % 2 == 0 else 'ESG Analytics Team'
        })
    
    return {
        'summary': f"This action plan addresses {len(selected_recommendations)} data quality issues identified in the ESG data assessment.",
        'timeline_data': timeline_data,
        'implementation_steps': implementation_steps
    }