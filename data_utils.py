"""
Data utilities for loading and preprocessing electrical load data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def generate_sample_data(start_date='2023-01-01', days=365):
    """
    Generate sample electrical load data for demonstration.
    
    Args:
        start_date: Starting date for the data
        days: Number of days to generate
        
    Returns:
        DataFrame with datetime index and load values
    """
    dates = pd.date_range(start=start_date, periods=days*24, freq='H')
    
    # Create synthetic load pattern with daily and weekly seasonality
    hours = np.array([d.hour for d in dates])
    days_of_week = np.array([d.dayofweek for d in dates])
    
    # Base load with daily pattern (higher during day, lower at night)
    daily_pattern = 50 + 30 * np.sin((hours - 6) * np.pi / 12)
    
    # Weekly pattern (higher on weekdays, lower on weekends)
    weekly_pattern = np.where(days_of_week < 5, 10, -5)
    
    # Add some random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Combine patterns
    load = daily_pattern + weekly_pattern + noise
    
    # Add trend
    trend = np.linspace(0, 10, len(dates))
    load += trend
    
    df = pd.DataFrame({
        'datetime': dates,
        'load': load
    })
    df.set_index('datetime', inplace=True)
    
    return df


def preprocess_data(df, lookback=24):
    """
    Preprocess data for ML models.
    
    Args:
        df: DataFrame with datetime index and 'load' column
        lookback: Number of previous hours to use as features
        
    Returns:
        X, y, scaler
    """
    # Create features from datetime
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    # Create lagged features
    for i in range(1, lookback + 1):
        df[f'load_lag_{i}'] = df['load'].shift(i)
    
    # Drop rows with NaN values (from lagged features)
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = ['hour', 'day_of_week', 'month', 'day_of_year'] + \
                   [f'load_lag_{i}' for i in range(1, lookback + 1)]
    
    X = df[feature_cols].values
    y = df['load'].values
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df.index


def split_data(X, y, train_ratio=0.8):
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target
        train_ratio: Ratio of training data
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test
