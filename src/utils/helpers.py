# Helper functions
from datetime import datetime, timedelta
import pandas as pd

def date_range_to_filter(date_range):
    """
    Convert a date range to a filter function
    
    Parameters:
    -----------
    date_range : list of datetime.date
        Date range [start_date, end_date]
        
    Returns:
    --------
    function
        Filter function for dataframe
    """
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        def date_filter(df):
            return df[(df['date'].dt.date >= start_date) & 
                      (df['date'].dt.date <= end_date)]
        
        return date_filter
    else:
        return lambda df: df

def filter_dataframe(df, date_range=None, building_type=None, data_source=None):
    """
    Apply filters to dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_range : list of datetime.date, optional
        Date range [start_date, end_date]
    building_type : str, optional
        Building type to filter by
    data_source : str, optional
        Data source to filter by
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & 
                                 (filtered_df['date'].dt.date <= end_date)]
    
    # Building type filter
    if building_type and building_type != 'All':
        filtered_df = filtered_df[filtered_df['building_type'] == building_type]
    
    # Data source filter
    if data_source and data_source != 'All':
        filtered_df = filtered_df[filtered_df['data_source'] == data_source]
    
    return filtered_df

def format_date(date):
    """
    Format a date for display
    """
    return date.strftime('%b %d, %Y')

def format_number(number, decimals=1):
    """
    Format a number for display
    """
    if number is None or pd.isna(number):
        return 'N/A'
    
    return f"{number:,.{decimals}f}"

def format_percentage(percentage, decimals=1):
    """
    Format a percentage for display
    """
    if percentage is None or pd.isna(percentage):
        return 'N/A'
    
    return f"{percentage:.{decimals}f}%"

def get_date_suffix(day):
    """
    Get the suffix for a day (st, nd, rd, th)
    """
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    
    return suffix

def format_date_with_suffix(date):
    """
    Format a date with a suffix (e.g., January 1st, 2023)
    """
    day = date.day
    suffix = get_date_suffix(day)
    
    return date.strftime(f'%B %-d{suffix}, %Y')