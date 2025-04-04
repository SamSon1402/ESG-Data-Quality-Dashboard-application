# Quality metrics calculation
import pandas as pd
import numpy as np

def calculate_completeness(df, columns=None):
    """
    Calculate completeness (% of non-null values) for specified columns
    """
    if columns is None:
        columns = df.columns
    
    completeness = {}
    for col in columns:
        if col in df.columns:
            completeness[col] = (1 - df[col].isna().mean()) * 100
    
    return completeness

def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers using IQR method or Z-score
    Returns percentage of outliers in each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).mean() * 100
            elif method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers[col] = (z_scores > threshold).mean() * 100
                else:
                    outliers[col] = 0
    
    return outliers

def calculate_consistency(df, columns=None, groupby_cols=None):
    """
    Calculate consistency metrics by measuring variation within groups
    Returns coefficient of variation for each column by group
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if groupby_cols is None:
        groupby_cols = ['data_source']
    
    consistency = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            # Check if all groupby columns exist
            if all(gc in df.columns for gc in groupby_cols):
                # Calculate coefficient of variation (std/mean) for each group
                grouped = df.groupby(groupby_cols)[col].agg(['std', 'mean'])
                grouped['cv'] = grouped['std'] / grouped['mean']
                consistency[col] = grouped['cv'].to_dict()
    
    return consistency

def calculate_quality_score(completeness, outliers, consistency, weights=None):
    """
    Calculate overall quality score based on completeness, outliers, and consistency
    """
    if weights is None:
        weights = {
            'completeness': 0.4,
            'outliers': 0.3,
            'consistency': 0.3
        }
    
    quality_scores = {}
    
    # Get common columns
    columns = set(completeness.keys()) & set(outliers.keys())
    
    for col in columns:
        # Completeness score (higher is better)
        completeness_score = completeness[col]
        
        # Outlier score (lower outlier percentage is better)
        outlier_score = 100 - outliers[col]
        
        # Consistency score (lower variation is better)
        # For consistency, use the average coefficient of variation across sources
        if col in consistency:
            avg_consistency = np.mean([v for v in consistency[col].values() if not np.isnan(v)])
            consistency_score = 100 * (1 - min(avg_consistency, 1))
        else:
            consistency_score = 100
        
        # Calculate weighted score
        quality_scores[col] = (
            weights['completeness'] * completeness_score + 
            weights['outliers'] * outlier_score +
            weights['consistency'] * consistency_score
        )
    
    return quality_scores

def calculate_all_quality_metrics(df, esg_columns=None):
    """
    Calculate all quality metrics for the dataset
    """
    if esg_columns is None:
        esg_columns = [
            'energy_consumption_kwh', 
            'water_consumption_m3', 
            'co2_emissions_kg', 
            'waste_kg'
        ]
    
    # Calculate completeness
    completeness = calculate_completeness(df, columns=esg_columns)
    
    # Calculate outliers
    outliers = detect_outliers(df, columns=esg_columns)
    
    # Calculate consistency by source
    consistency_by_source = calculate_consistency(
        df, 
        columns=esg_columns, 
        groupby_cols=['data_source']
    )
    
    # Calculate overall quality score
    quality_scores = calculate_quality_score(
        completeness, 
        outliers, 
        consistency_by_source
    )
    
    # Prepare completeness over time
    completeness_over_time = df.pivot_table(
        index='date',
        values=esg_columns,
        aggfunc=lambda x: 100 * (1 - np.mean(pd.isna(x)))
    )
    
    return {
        'completeness': completeness,
        'outliers': outliers,
        'consistency_by_source': consistency_by_source,
        'quality_scores': quality_scores,
        'completeness_over_time': completeness_over_time
    }