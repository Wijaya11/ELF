# Enhanced Load Forecasting - Model Configuration Guide

## Overview

This project uses an ensemble of 6 machine learning and statistical models to deliver robust load forecasting:

1. **Ridge/ElasticNet** - Linear model with regularization
2. **Random Forest (RF)** - Ensemble of decision trees
3. **XGBoost (XGB)** - Gradient boosted trees
4. **LightGBM (LGB)** - Fast gradient boosting framework
5. **SARIMAX** - Seasonal ARIMA with exogenous variables
6. **Prophet** - Facebook's time series forecasting

## Model Configurations

### 1. Ridge/ElasticNet Model

**Purpose**: Provides interpretable linear baseline with feature selection

**Hyperparameters**:
- `l1_ratio`: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7] - Balances L1 (Lasso) and L2 (Ridge) regularization
- `alphas`: logspace(-3, 1.5, 15) - Regularization strength search space
- `max_iter`: 30,000 - Maximum iterations for convergence
- `cv`: TimeSeriesSplit with 2-5 folds
- `selection`: 'random' - Faster coordinate descent
- `n_jobs`: -1 - Parallel cross-validation

**Features Used**:
- Lag features: lag1, lag12
- Rolling statistics: roll3_mean, roll12_mean
- Seasonal: month_sin, month_cos
- RJPP-related: rjpp_mom_pct, rjpp_yoy_pct, rjpp_gap

**Fallback**: Ridge(alpha=1.0) if CV fails

### 2. Random Forest (RF) Model

**Purpose**: Captures non-linear patterns and feature interactions

**Hyperparameters**:
- `n_estimators`: 1,000 - Number of trees
- `max_depth`: 7 - Maximum tree depth
- `min_samples_leaf`: 10 - Minimum samples at leaf
- `min_samples_split`: 20 - Minimum samples to split
- `max_features`: 0.65 - Features per split (65%)
- `ccp_alpha`: 2e-4 - Pruning parameter
- `bootstrap`: True - Use bootstrap samples
- `n_jobs`: -1 - Parallel training

**Fallback Chain**:
1. ExtraTreesRegressor with same optimized params
2. GradientBoostingRegressor (600 trees, LR 0.04)

### 3. XGBoost (XGB) Model

**Purpose**: Gradient boosting for high-accuracy predictions

**Hyperparameters**:
- `n_estimators`: 1,500 - Boosting rounds
- `learning_rate`: 0.025 - Step size shrinkage
- `max_depth`: 5 - Maximum tree depth
- `min_child_weight`: 6 - Minimum sum of instance weight
- `subsample`: 0.75 - Row sampling ratio
- `colsample_bytree`: 0.75 - Column sampling ratio
- `reg_lambda`: 10.0 - L2 regularization
- `reg_alpha`: 1.5 - L1 regularization
- `gamma`: 0.3 - Minimum loss reduction
- `tree_method`: 'hist' - Histogram-based algorithm
- `importance_type`: 'gain' - Feature importance metric

**Early Stopping**: Up to 50 rounds on validation set

**Fallback**: GradientBoostingRegressor (1,000 trees, optimized params)

### 4. LightGBM (LGB) Model

**Purpose**: Fast and efficient gradient boosting

**Hyperparameters**:
- `n_estimators`: 5,000 - Maximum boosting rounds
- `learning_rate`: 0.018 - Step size
- `num_leaves`: 40 - Maximum leaves per tree
- `max_depth`: 7 - Maximum tree depth
- `min_data_in_leaf`: 35 - Minimum samples per leaf
- `feature_fraction`: 0.8 - Feature sampling ratio
- `bagging_fraction`: 0.8 - Data sampling ratio
- `bagging_freq`: 1 - Bagging frequency
- `lambda_l1`: 0.08 - L1 regularization
- `lambda_l2`: 8.0 - L2 regularization
- `min_child_weight`: 0.001 - Minimum child weight
- `path_smooth`: 0.1 - Path smoothing parameter
- `importance_type`: 'gain' - Feature importance metric

**Early Stopping**: Up to 60 rounds on validation set

**Fallback Chain**:
1. HistGradientBoostingRegressor (1,500 iter, depth 7, L2=5.0)
2. GradientBoostingRegressor (1,200 trees, optimized params)
3. SeasonalNaive (last resort)

### 5. SARIMAX Model

**Purpose**: Captures seasonal patterns and ARIMA dynamics

**Residual Mode** (when RJPP available):
- Order: (1, 0, 1) - AR(1), MA(1)
- Seasonal: (1, 1, 1, 12) - Seasonal AR, diff, MA with period 12
- Trend: None
- Initialization: 'approximate_diffuse'
- Optimizer: LBFGS with maxiter=200

**Level Mode** (default):
- Order: (1, 1, 1) - AR(1), diff(1), MA(1)
- Seasonal: (1, 1, 1, 12)
- Exogenous: RJPP if available
- Initialization: 'approximate_diffuse'
- Optimizer: LBFGS with maxiter=200

**Fallback**: ExponentialSmoothing (Holt-Winters)

### 6. Prophet Model

**Purpose**: Additive time series with trend and seasonality components

**Hyperparameters**:
- `n_changepoints`: 5 - Potential trend changes
- `changepoint_prior_scale`: 0.05 - Changepoint flexibility
- `changepoint_range`: 0.8 - Focus on first 80% of data
- `seasonality_prior_scale`: 12.0 - Seasonality strength
- `yearly_seasonality`: True
- `growth`: 'linear' (level mode) or 'flat' (residual mode)

**Custom Seasonality**:
- Monthly seasonality: period=12, fourier_order=8

**Fallback**: None - marks as unavailable if training fails

## Feature Engineering

### Core Features (All Models)

**Lag Features**:
- `lag1`, `lag2`, `lag3`: Recent history (1-3 months back)
- `lag12`: Seasonal lag (12 months back)

**Rolling Statistics**:
- `roll3_mean`, `roll6_mean`, `roll12_mean`: Moving averages
- `roll3_std`, `roll12_std`: Rolling volatility

**Growth Rates**:
- `mom_pct`: Month-over-month percentage change
- `yoy_pct`: Year-over-year percentage change

**Seasonal Features**:
- `month_sin`, `month_cos`: Cyclic month encoding
- `quarter`: Quarter indicator

**RJPP Features** (when available):
- `rjpp_mom_pct`: RJPP month-over-month change
- `rjpp_yoy_pct`: RJPP year-over-year change
- `rjpp_gap`: Difference between load and RJPP

## Ensemble Methods

### 1. Weighted Ensemble

**Approach**: Inverse-error weighting with multi-metric optimization

**Metrics Combined**:
- RMSE: 60% weight
- MAE: 40% weight
- RJPP deviation penalty: 100%
- Bias penalty: 50%

**Weight Distribution**:
- Min weight per model: 3%
- Max weight per model: 45%
- Temperature parameter: 0.65

### 2. Meta-Ensemble

**Approach**: ElasticNetCV learns optimal model combinations

**Configuration**:
- `l1_ratio`: [0.1, 0.3, 0.5, 0.7, 0.9]
- `alphas`: logspace(-4, 0, 12)
- `positive`: True - Ensures positive weights
- `cv`: TimeSeriesSplit

**Fallback**: Ridge(alpha=1e-3) with positive constraint

### 3. Horizon-Aware Weighting

**Buckets**:
- h1_3: Months 1-3 (short-term)
- h4_6: Months 4-6 (medium-term)
- h7_12: Months 7-12 (long-term)
- h13p: Months 13+ (very long-term)

**Per-Bucket Weighting**:
- Combines RMSE (60%) and MAE (30%)
- Penalty weights: LAMBDA_DEV=0.8, LAMBDA_BIAS=0.4
- Min weight per model per bucket: 2%
- Max weight per model per bucket: 50%

## Model Calibration

### Residual Model Calibration

Applied to models operating in residual mode (Ridge, RF, XGB, LGB, SARIMAX, Prophet when using RJPP):

**Scale Calibration**:
- Matches predicted residual volatility to actual residual volatility
- Alpha bounds: [0.75, 1.25]
- Formula: `alpha = clip(σ_true / σ_model, 0.75, 1.25)`

**Bias Calibration**:
- Removes systematic over/under-prediction
- Capped at 2 standard deviations to prevent extreme corrections
- Formula: `bias = clip(mean(adjusted_residual), -2σ, +2σ)`

**Application**:
```python
calibrated_pred = (pred_residual * alpha - bias) + RJPP
```

## Validation Strategy

### Rolling-Origin Cross-Validation

**Configuration**:
- `val_months`: 12 (default validation window)
- `rolling_k`: 2 (default number of windows)
- Each window refits all models from scratch (no leakage)

**Metrics Computed**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- WAPE: Weighted Absolute Percentage Error
- sMAPE: Symmetric MAPE
- RJPP_dev%: Average deviation from RJPP
- Bias%: Signed bias relative to RJPP

### Band Calibration

**Target Coverage**: 80% (P10-P90 band)

**Components**:
- Residual uncertainty: Month-specific historical residuals
- Model spread: Standard deviation across base models
- Combined: `σ = sqrt(σ_residual² + σ_spread²)`

**Auto-calibration**:
- Search range: k ∈ [0.6, 1.8]
- Step size: 0.05
- Optimizes for actual coverage ≈ 80%

## Training Summary Output

Example log output during training:

```
============================================================
MODEL PERFORMANCE SUMMARY
============================================================
Ridge      | Weight: 0.185 | Status: ok           | RMSE:   245.32 | MAE:   198.45 | Impl: ElasticNetCV (optimized)
RF         | Weight: 0.195 | Status: ok           | RMSE:   238.15 | MAE:   192.78 | Impl: RandomForestRegressor (optimized)
XGB        | Weight: 0.210 | Status: ok           | RMSE:   232.87 | MAE:   188.34 | Impl: XGBRegressor (optimized)
LGB        | Weight: 0.215 | Status: ok           | RMSE:   230.45 | MAE:   186.92 | Impl: LGBMRegressor (optimized)
SARIMAX    | Weight: 0.095 | Status: ok           | RMSE:   268.23 | MAE:   215.67 | Impl: SARIMAX(1,1,1)x(1,1,1,12) optimized
Prophet    | Weight: 0.100 | Status: ok           | RMSE:   265.78 | MAE:   213.45 | Impl: Prophet (optimized)
============================================================
```

## Performance Optimization Tips

### 1. Data Quality
- Ensure no gaps in historical data
- Remove or interpolate outliers carefully
- Verify RJPP data quality if using residual mode

### 2. Feature Selection
- More features isn't always better for tree models
- Linear models benefit from feature engineering
- Monitor feature importance to identify key drivers

### 3. Hyperparameter Tuning
- Current parameters are optimized for general use
- Consider domain-specific tuning for specialized datasets
- Use validation metrics to guide adjustments

### 4. Ensemble Configuration
- Meta-ensemble works best with diverse base models
- Weighted ensemble is more robust to model failures
- Horizon-aware weights improve long-range forecasts

### 5. Computational Efficiency
- LightGBM is fastest for large datasets
- XGBoost benefits from GPU acceleration
- Prophet is slowest but most interpretable

## Common Issues and Solutions

### Issue: Model marked as "unavailable"
**Solution**: Check data requirements (minimum samples, feature availability)

### Issue: Poor long-range forecast accuracy
**Solution**: Ensure SARIMAX and Prophet are training successfully

### Issue: Ensemble weights heavily favor one model
**Solution**: Check validation metrics - one model may genuinely perform better

### Issue: High RJPP deviation
**Solution**: Verify RJPP data quality and calibration settings

### Issue: Training takes too long
**Solution**: Reduce n_estimators for tree models or use fewer validation windows

## Version History

**Current Version**: Optimized Configuration v2.0

**Changes from Previous Version**:
- Improved hyperparameters for all 6 models
- Enhanced ensemble weighting with multi-metric optimization
- Better fallback mechanisms for each model
- Expanded feature engineering (16 features vs 11)
- Improved SARIMAX specification (added MA terms)
- Enhanced Prophet configuration (more changepoints, linear growth)
- Better calibration with wider alpha bounds
- Horizon-aware weighting system
- Comprehensive logging and diagnostics

## References

- XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- LightGBM: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
- Prophet: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
- Statsmodels SARIMAX: [https://www.statsmodels.org/](https://www.statsmodels.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
