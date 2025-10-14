"""
Configuration file for ELF - Electrical Load Forecasting.
Modify these settings to customize the application behavior.
"""

# Data Generation Settings
DEFAULT_DAYS = 365
MIN_DAYS = 30
MAX_DAYS = 730

# Feature Engineering Settings
DEFAULT_LOOKBACK = 24  # hours
MIN_LOOKBACK = 12
MAX_LOOKBACK = 72

# Training Settings
DEFAULT_TRAIN_RATIO = 0.8
MIN_TRAIN_RATIO = 0.6
MAX_TRAIN_RATIO = 0.9

# Model Training Parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'svr': {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 'scale'
    },
    'lstm': {
        'epochs': 30,
        'batch_size': 32,
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'patience': 5
    }
}

# Ensemble Settings
DEFAULT_ENSEMBLE_METHOD = 'mean'
ENSEMBLE_WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]  # LR, RF, GB, XGB, SVR, LSTM

# Visualization Settings
PLOT_HEIGHT = 500
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
ACTUAL_COLOR = 'black'
ENSEMBLE_COLOR = 'red'

# UI Settings
PAGE_TITLE = "ELF - Electrical Load Forecasting"
PAGE_ICON = "⚡"
LAYOUT = "wide"

# Model Names
MODEL_NAMES = [
    'Linear Regression',
    'Random Forest',
    'Gradient Boosting',
    'XGBoost',
    'SVR',
    'LSTM'
]
