# Outlier detection algorithms
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def identify_outliers_iqr(df, column, group_by=None, threshold=1.5):
    """
    Identify outliers using the IQR method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with the data
    column : str
        The column to check for outliers
    group_by : str or list, optional
        Column(s) to group by before outlier detection
    threshold : float, default 1.5
        The threshold multiplier for IQR
        
    Returns:
    --------
    pandas.Series
        Boolean series where True indicates outlier
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    
    # If group_by is specified, detect outliers within each group
    if group_by is not None:
        outliers = pd.Series(False, index=df.index)
        
        for name, group in df.groupby(group_by):
            if len(group) > 10:  # Only process groups with enough data points
                Q1 = group[column].quantile(0.25)
                Q3 = group[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                group_outliers = (group[column] < lower_bound) | (group[column] > upper_bound)
                outliers.loc[group.index] = group_outliers
        
        return outliers
    else:
        # Process entire dataset
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)

def identify_outliers_zscore(df, column, group_by=None, threshold=3.0):
    """
    Identify outliers using the Z-score method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with the data
    column : str
        The column to check for outliers
    group_by : str or list, optional
        Column(s) to group by before outlier detection
    threshold : float, default 3.0
        The threshold for z-score
        
    Returns:
    --------
    pandas.Series
        Boolean series where True indicates outlier
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    
    # If group_by is specified, detect outliers within each group
    if group_by is not None:
        outliers = pd.Series(False, index=df.index)
        
        for name, group in df.groupby(group_by):
            if len(group) > 10:  # Only process groups with enough data points
                mean = group[column].mean()
                std = group[column].std()
                
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((group[column] - mean) / std)
                    group_outliers = z_scores > threshold
                    outliers.loc[group.index] = group_outliers
        
        return outliers
    else:
        # Process entire dataset
        mean = df[column].mean()
        std = df[column].std()
        
        if std > 0:  # Avoid division by zero
            z_scores = np.abs((df[column] - mean) / std)
            return z_scores > threshold
        else:
            return pd.Series(False, index=df.index)

def identify_temporal_anomalies(df, column, date_column='date', window=3, threshold=2.0):
    """
    Identify anomalies based on temporal patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with the data
    column : str
        The column to check for anomalies
    date_column : str, default 'date'
        The date column to use for temporal analysis
    window : int, default 3
        The rolling window size for analysis
    threshold : float, default 2.0
        The threshold for anomaly detection
        
    Returns:
    --------
    pandas.Series
        Boolean series where True indicates anomaly
    """
    if column not in df.columns or date_column not in df.columns:
        return pd.Series(False, index=df.index)
    
    # Ensure the dataframe is sorted by date
    df_sorted = df.sort_values(date_column)
    
    # Calculate rolling statistics
    rolling_mean = df_sorted[column].rolling(window=window, min_periods=1).mean()
    rolling_std = df_sorted[column].rolling(window=window, min_periods=1).std()
    
    # Replace zero standard deviation with mean to avoid division by zero
    rolling_std = rolling_std.replace(0, rolling_std.mean())
    
    # Calculate z-scores based on rolling statistics
    z_scores = np.abs((df_sorted[column] - rolling_mean) / rolling_std)
    
    # Identify anomalies
    anomalies = z_scores > threshold
    
    # Reindex to match original dataframe
    return anomalies.reindex(df.index)

def get_outlier_details(df, column, outlier_mask):
    """
    Get details about identified outliers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with the data
    column : str
        The column with outliers
    outlier_mask : pandas.Series
        Boolean series where True indicates outlier
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with outlier details
    """
    if not outlier_mask.any():
        return pd.DataFrame()
    
    # Get outlier rows
    outliers_df = df[outlier_mask].copy()
    
    # Calculate how far the outlier is from the IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Add deviation columns
    outliers_df['value'] = outliers_df[column]
    outliers_df['expected_range'] = f"{lower_bound:.2f} - {upper_bound:.2f}"
    outliers_df['deviation_percent'] = np.where(
        outliers_df[column] > upper_bound,
        ((outliers_df[column] - upper_bound) / upper_bound * 100),
        ((lower_bound - outliers_df[column]) / lower_bound * 100)
    )
    
    return outliers_df