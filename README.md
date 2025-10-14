# ELF - Enhanced Load Forecasting

A production-ready ensemble forecasting system that combines 6 machine learning and statistical models for robust electrical load prediction.

## Features

- **6 Optimized Models**: Ridge/ElasticNet, Random Forest, XGBoost, LightGBM, SARIMAX, Prophet
- **Smart Ensemble**: Automatic weighted and meta-learning ensemble with horizon-aware optimization
- **RJPP Integration**: Residual-based modeling with RJPP (Reference Plan) alignment
- **Robust Validation**: Rolling-origin cross-validation with multi-metric evaluation
- **Interactive UI**: Streamlit-based interface with scenario comparison and AI insights
- **Production Ready**: Comprehensive error handling, fallback mechanisms, and quality checks

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt  # if available, or install manually

# Run the application
streamlit run app.py
```

## Model Performance

All 6 models are optimized with:
- ✅ State-of-the-art hyperparameters
- ✅ Comprehensive fallback mechanisms
- ✅ Feature importance tracking
- ✅ Calibration for residual models
- ✅ Multi-metric validation (RMSE, MAE, MAPE, WAPE, sMAPE)

See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for detailed configuration guide.

## Architecture

```
Input Data → Feature Engineering → 6 Base Models → Ensemble → Post-Processing → Final Forecast
              (16+ features)        (parallel)      (weighted +     (guards,      (with bands)
                                                     meta-model)     smoothing)
```

### Base Models
1. **Ridge/ElasticNet** - Linear baseline with regularization
2. **Random Forest** - Non-linear ensemble of decision trees
3. **XGBoost** - Gradient boosting for high accuracy
4. **LightGBM** - Fast and efficient gradient boosting
5. **SARIMAX** - Seasonal ARIMA with exogenous variables
6. **Prophet** - Additive seasonality and trend decomposition

### Ensemble Methods
- **Weighted**: Multi-metric inverse-error weighting (RMSE 60% + MAE 40%)
- **Meta-Model**: ElasticNetCV learns optimal combinations
- **Horizon-Aware**: Different weights for short/medium/long-term forecasts

## Key Features

### Advanced Modeling
- Residual-based modeling when RJPP is available
- Automatic calibration (scale + bias correction)
- Time-decay weighted training
- Rolling-origin validation with no leakage

### Quality Assurance
- Non-negativity constraints
- Month-over-month spike guards
- RJPP deviation bounds
- Auto-calibrated uncertainty bands (P10-P90)
- Comprehensive diagnostics and logging

### Scenario Analysis
- Interactive scenario editor
- Multi-scenario comparison
- Impact analysis on forecasts and bands
- Export capabilities

## Performance Metrics

The system tracks multiple metrics for comprehensive evaluation:

- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAE**: Mean Absolute Error (average absolute difference)
- **MAPE**: Mean Absolute Percentage Error (scale-free)
- **WAPE**: Weighted APE (robust to near-zero values)
- **sMAPE**: Symmetric MAPE (balanced over/under-prediction)
- **RJPP Dev%**: Deviation from reference plan
- **Bias%**: Systematic over/under-prediction

## Documentation

- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - Detailed model configurations and hyperparameters
- [app.py](app.py) - Main application code with inline documentation

## Requirements

Core dependencies:
- Python 3.8+
- streamlit
- pandas, numpy
- scikit-learn
- statsmodels
- plotly

Optional (recommended):
- xgboost
- lightgbm
- prophet (fbprophet)

## Project Status

✅ **Production Ready** - All 6 models optimized and validated
- Comprehensive error handling and fallbacks
- Multi-metric validation and diagnostics
- Horizon-aware ensemble optimization
- Calibrated uncertainty bands
- Quality assurance guards