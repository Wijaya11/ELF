# ELF Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELF Application                           │
│                  (Streamlit Web Interface)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  data_utils.py                                            │  │
│  │  - generate_sample_data()                                 │  │
│  │  - preprocess_data()                                      │  │
│  │  - split_data()                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Layer                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  models.py - LoadForecastingModels                        │  │
│  │                                                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │   Linear   │  │   Random   │  │  Gradient  │          │  │
│  │  │ Regression │  │   Forest   │  │  Boosting  │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘          │  │
│  │                                                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │  XGBoost   │  │    SVR     │  │    LSTM    │          │  │
│  │  │            │  │            │  │  (Neural)  │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘          │  │
│  │                                                            │  │
│  │                      ▼                                     │  │
│  │              ┌────────────────┐                            │  │
│  │              │    Ensemble    │                            │  │
│  │              │ (Mean/Median)  │                            │  │
│  │              └────────────────┘                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Data Generation
        │
        ▼
Feature Engineering
    │       │
    ▼       ▼
Hour    Lagged Values
Day     (previous hours)
Month
        │
        ▼
  Normalization
        │
        ▼
Train/Test Split (80/20)
        │
        ├────────────┬────────────┬────────────┐
        │            │            │            │
        ▼            ▼            ▼            ▼
    Model 1      Model 2      Model 3    ... Model 6
        │            │            │            │
        └────────────┴────────────┴────────────┘
                     │
                     ▼
              Ensemble Prediction
                     │
                     ▼
           Evaluation & Visualization
```

## Component Details

### 1. Data Utils (`data_utils.py`)

**Purpose**: Handle data generation, preprocessing, and feature engineering

**Key Functions**:
- `generate_sample_data()`: Creates synthetic load data with patterns
- `preprocess_data()`: Extracts features and creates lagged values
- `split_data()`: Divides data into train/test sets

**Features Created**:
- Temporal: hour, day_of_week, month, day_of_year
- Historical: load_lag_1, load_lag_2, ..., load_lag_N

### 2. Models (`models.py`)

**Purpose**: Implement 6 ML models and ensemble methods

**Class**: `LoadForecastingModels`

**Models Implemented**:

1. **Linear Regression**
   - Type: Statistical
   - Complexity: Low
   - Speed: Fast
   - Use: Baseline

2. **Random Forest**
   - Type: Ensemble (Bagging)
   - Complexity: Medium
   - Speed: Fast
   - Use: Non-linear patterns

3. **Gradient Boosting**
   - Type: Ensemble (Boosting)
   - Complexity: Medium
   - Speed: Medium
   - Use: Sequential learning

4. **XGBoost**
   - Type: Optimized Boosting
   - Complexity: Medium-High
   - Speed: Fast
   - Use: Best tree-based

5. **SVR (Support Vector Regression)**
   - Type: Kernel Method
   - Complexity: Medium
   - Speed: Slow
   - Use: Complex relationships

6. **LSTM (Long Short-Term Memory)**
   - Type: Deep Learning
   - Complexity: High
   - Speed: Slowest
   - Use: Temporal dependencies

**Ensemble Methods**:
- Mean: Simple average of all predictions
- Median: Middle value (robust to outliers)
- Weighted: Custom weights per model

### 3. Streamlit App (`app.py`)

**Purpose**: Interactive web interface for the tool

**Features**:
- Configuration sidebar
- Real-time training progress
- Multiple visualization tabs
- Performance metrics display
- Model comparison charts

**Visualizations**:
- Time series plots (Plotly)
- Comparison charts (Plotly Express)
- Error distribution histograms
- Scatter plots (actual vs predicted)

### 4. Configuration (`config.py`)

**Purpose**: Centralized settings management

**Configurable Parameters**:
- Data generation ranges
- Model hyperparameters
- UI settings
- Visualization colors

## Workflow

### Training Phase

```
1. Configure Parameters
   ├─ Number of days
   ├─ Lookback period
   └─ Train/test split

2. Generate Data
   └─ Synthetic electrical load with patterns

3. Preprocess
   ├─ Create temporal features
   ├─ Create lagged features
   └─ Normalize values

4. Split Data
   ├─ Training set (80%)
   └─ Test set (20%)

5. Train Models (parallel)
   ├─ Linear Regression → trained
   ├─ Random Forest → trained
   ├─ Gradient Boosting → trained
   ├─ XGBoost → trained
   ├─ SVR → trained
   └─ LSTM → trained

6. Store Models
   └─ All models in memory
```

### Prediction Phase

```
1. Load Test Data
   └─ 20% of total data

2. Generate Predictions
   ├─ Model 1 → predictions_1
   ├─ Model 2 → predictions_2
   ├─ Model 3 → predictions_3
   ├─ Model 4 → predictions_4
   ├─ Model 5 → predictions_5
   └─ Model 6 → predictions_6

3. Create Ensemble
   └─ Combine all predictions → ensemble

4. Calculate Metrics
   ├─ RMSE
   ├─ MAE
   ├─ MAPE
   └─ R²

5. Visualize Results
   ├─ Time series plots
   ├─ Error analysis
   └─ Model comparisons
```

## Technology Stack

### Core Libraries

- **Python 3.8+**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation

### Machine Learning

- **scikit-learn**: Traditional ML (LR, RF, GB, SVR)
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: Deep learning (LSTM)

### Visualization

- **Streamlit**: Web interface
- **Plotly**: Interactive plots
- **Matplotlib/Seaborn**: Additional plotting

### Deployment

- **Docker**: Containerization
- **docker-compose**: Multi-container orchestration

## Performance Considerations

### Speed
- **Fastest**: Linear Regression, Random Forest
- **Medium**: Gradient Boosting, XGBoost
- **Slow**: SVR
- **Slowest**: LSTM (training)

### Accuracy
- **Best Overall**: Ensemble
- **Best Individual**: Usually XGBoost or LSTM
- **Consistent**: Random Forest, Gradient Boosting

### Memory
- **Lightweight**: Linear Regression, SVR
- **Moderate**: Tree-based models
- **Heavy**: LSTM (depends on architecture)

## Scalability

### Data Size
- **Current**: 30-730 days (720-17,520 hours)
- **Practical Limit**: ~2 years of hourly data
- **Bottleneck**: LSTM training time

### Model Training
- **Parallel Potential**: High (6 independent models)
- **Current**: Sequential (for progress tracking)
- **Improvement**: Could parallelize with multiprocessing

### Inference
- **Speed**: Real-time for single predictions
- **Batch**: Efficient for multiple time steps

## Extension Points

### Adding New Models
1. Add training method to `LoadForecastingModels`
2. Update `train_all_models()` to include it
3. Add to model names list
4. Update ensemble weights

### Custom Data Sources
1. Modify `data_utils.py`
2. Implement custom loader
3. Ensure datetime index + 'load' column
4. Maintain same preprocessing pipeline

### Enhanced Features
1. Add weather data
2. Include holiday indicators
3. Add price information
4. Incorporate external factors

## Security Considerations

- No external data connections (currently)
- No authentication required (local use)
- No persistent storage
- All data in memory
- Docker isolation available

## Future Enhancements

1. **Model Persistence**: Save/load trained models
2. **Real Data**: Support for CSV/database input
3. **API Endpoint**: REST API for predictions
4. **Batch Processing**: Process multiple scenarios
5. **Hyperparameter Tuning**: Automated optimization
6. **Model Explainability**: SHAP values, feature importance
7. **Real-time Updates**: Live data streaming
8. **Multi-step Forecast**: Predict multiple hours ahead
