# User Guide
# ESG Data Quality Dashboard - User Guide

## Overview

The ESG Data Quality Dashboard is a tool for analyzing and monitoring the quality of Environmental, Social, and Governance (ESG) data across your real estate portfolio. This guide will help you understand how to use the dashboard effectively.

## Getting Started

### Installation

1. Make sure you have Python 3.8+ installed
2. Clone the repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

### Dashboard Pages

The application is organized into four main pages:

1. **Overview** - High-level summary of ESG data quality
2. **Data Quality Analysis** - Detailed analysis by ESG category
3. **Trends** - Data quality trends over time
4. **Recommendations** - Action items to improve data quality

## Using the Dashboard

### Filters

All pages include the following filters:

- **Date Range** - Select the time period for analysis
- **Building Type** - Filter by building category (Office, Retail, etc.)
- **Data Source** - Filter by data collection method (API, Manual Entry, etc.)

### Overview Page

The Overview page provides a high-level summary of ESG data quality:

- **Key Metrics** - Overall quality score, data completeness, and outlier percentages
- **Quality by Category** - Bar chart showing quality scores for each ESG category
- **Building Portfolio Map** - Bubble chart showing quality vs. building size
- **Top Issues** - Summary of the most critical quality issues

### Data Quality Analysis Page

The Data Quality Analysis page provides detailed analysis for each ESG category:

- **Metric Scores** - Completeness, outlier, and overall quality scores
- **Completeness by Source** - Bar chart showing data completeness by source
- **Data Distribution** - Histogram showing the distribution of values
- **Missing Data Calendar** - Heatmap showing missing data patterns by building

### Trends Page

The Trends page shows how data quality metrics have changed over time:

- **Completeness Trends** - Line chart showing data completeness over time
- **Source Quality Trends** - Line chart showing data quality by source over time
- **Building Type Comparison** - Bar chart comparing quality metrics by building type

### Recommendations Page

The Recommendations page provides actionable insights to improve data quality:

- **Critical Issues** - High-impact quality issues requiring immediate attention
- **Medium Priority** - Medium-impact issues for secondary focus
- **All Recommendations** - Complete list of improvement recommendations
- **Action Plan Generator** - Create a customized action plan based on selected priorities

## Best Practices

- Regularly review the dashboard to identify data quality issues
- Focus on critical issues first before addressing medium-priority items
- Use the action plan generator to create a structured improvement approach
- Track quality metrics over time to measure improvement
- Review completeness by source to identify problematic data providers