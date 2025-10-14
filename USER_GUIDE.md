# ELF User Guide

## Getting Started

ELF (Electrical Load Forecasting) is an interactive tool that uses 6 different machine learning models to predict electrical load demand. This guide will help you use the application effectively.

## Launching the Application

After installation, run:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Understanding the Interface

### Sidebar (Configuration Panel)

The sidebar on the left contains all the configuration options:

#### Data Settings
- **Number of days to generate (30-730)**: Controls how much historical data to create
  - More data = better training but slower processing
  - Recommended: 365 days for balanced results

- **Lookback period (12-72 hours)**: Number of previous hours used as features
  - Higher values capture longer patterns
  - Recommended: 24 hours for daily patterns

- **Training data ratio (60-90%)**: Proportion of data used for training
  - Higher values = more training data, less test data
  - Recommended: 80% (0.8)

#### Ensemble Settings
- **Ensemble Method**: How to combine predictions from all models
  - **Mean**: Simple average (default, most robust)
  - **Median**: Middle value (resistant to outliers)
  - **Weighted**: Custom weights for each model

### Main Interface

After clicking "Run Forecast", you'll see several sections:

## Step-by-Step Workflow

### Step 1: Data Generation
The tool generates synthetic electrical load data with realistic patterns:
- Daily cycles (higher during day, lower at night)
- Weekly patterns (weekdays vs weekends)
- Long-term trends
- Random variations

### Step 2: Data Preprocessing
Creates features from the raw data:
- Temporal features (hour, day of week, month)
- Lagged values (previous hours)
- Normalized scaling

### Step 3: Model Training
Trains 6 different machine learning models:

1. **Linear Regression**: Simple baseline
2. **Random Forest**: Ensemble of decision trees
3. **Gradient Boosting**: Sequential learning
4. **XGBoost**: Optimized gradient boosting
5. **SVR**: Support Vector Regression
6. **LSTM**: Deep learning neural network

### Step 4: Prediction
Each model generates forecasts for the test period.

### Step 5: Ensemble
Combines all predictions into a final forecast.

## Understanding the Results

### Performance Metrics Table

The table shows metrics for each model:

- **RMSE** (Root Mean Square Error): Overall error magnitude
  - Lower is better
  - Penalizes large errors

- **MAE** (Mean Absolute Error): Average error size
  - Lower is better
  - More interpretable than RMSE

- **MAPE** (Mean Absolute Percentage Error): Error as percentage
  - Lower is better
  - Shows relative accuracy

- **R²** (R-squared): Goodness of fit
  - Higher is better (max = 1.0)
  - Shows how well model explains variance

### Visualization Tabs

#### Tab 1: Forecast Comparison
- **Main Plot**: Shows actual load vs ensemble forecast
  - Black line = actual values
  - Red dashed line = ensemble prediction

- **Detailed View**: Last 7 days with markers
  - Useful for examining recent accuracy

#### Tab 2: Individual Models
- Shows predictions from all 6 models plus ensemble
- Compare how different models perform
- Identify which models agree/disagree

#### Tab 3: Error Analysis
- **Error Distribution**: Histogram of prediction errors
  - Should be centered around zero
  - Width shows prediction variability

- **Actual vs Predicted**: Scatter plot
  - Points near diagonal = accurate predictions
  - Distance from line = error magnitude

- **Error Over Time**: Time series of errors
  - Identify when model performs poorly
  - Check for systematic biases

### Model Comparison Charts

Two bar charts compare all models:

1. **RMSE Comparison**: Lower bars are better
2. **R² Score Comparison**: Higher bars are better

The **Ensemble** model typically performs best or near-best, as it combines strengths of all models.

## Tips for Best Results

### Data Configuration
1. **Start with defaults**: 365 days, 24-hour lookback, 80% training
2. **More data for stable patterns**: Use 500-700 days if you want robust training
3. **Shorter lookback for quick changes**: Use 12-18 hours for rapidly changing loads
4. **Higher training ratio for small datasets**: Use 85-90% if you have less than 100 days

### Interpreting Results
1. **Check R² score**: Should be above 0.8 for good models
2. **Compare models**: If all models perform similarly, data might be too simple
3. **Look at error patterns**: Systematic errors suggest missing features
4. **Ensemble should win**: If individual models outperform ensemble, investigate why

### Performance Optimization
1. **Reduce data for faster testing**: Start with 90-180 days while exploring
2. **Increase data for final results**: Use 365-730 days for production forecasts
3. **LSTM is slowest**: Consider disabling if time is critical (requires code modification)

## Common Use Cases

### Case 1: Quick Exploration
```
- Days: 90
- Lookback: 24
- Training ratio: 0.8
- Ensemble: mean
```
Fast results, good for understanding the tool.

### Case 2: Production Forecasting
```
- Days: 365-730
- Lookback: 24-48
- Training ratio: 0.8
- Ensemble: weighted
```
Best accuracy for real applications.

### Case 3: Short-term Patterns
```
- Days: 180
- Lookback: 12
- Training ratio: 0.85
- Ensemble: mean
```
Focus on immediate patterns.

## Exporting Results

Currently, results are displayed in the interface. To save results:

1. **Screenshots**: Use browser screenshot tools
2. **Copy metrics**: Select and copy from the metrics table
3. **Plot download**: Hover over plots and click camera icon

## Troubleshooting

### Models take too long to train
- Reduce number of days
- Close other applications
- LSTM takes the longest (20-30 epochs)

### Poor prediction accuracy (R² < 0.5)
- Increase training data
- Increase lookback period
- Check if ensemble improves results

### Application freezes
- Check terminal for errors
- Reduce data amount
- Restart the application

### Browser doesn't open
- Manually go to http://localhost:8501
- Check terminal for correct URL
- Try different port if 8501 is busy

## Advanced Usage

### Custom Data
To use your own data (requires code modification):
1. Edit `data_utils.py`
2. Replace `generate_sample_data()` with your data loader
3. Ensure data has datetime index and 'load' column

### Model Customization
To adjust model parameters:
1. Edit `models.py`
2. Modify hyperparameters in each `train_*` method
3. Add new models by following existing patterns

### Ensemble Weights
To customize weighted ensemble:
1. Edit `models.py`, find `ensemble_prediction()` method
2. Adjust weights array based on validation performance
3. Weights should sum to 1.0

## Best Practices

1. **Start simple**: Use default settings first
2. **Iterate**: Adjust one parameter at a time
3. **Compare**: Run multiple times with different settings
4. **Document**: Note which settings work best for your use case
5. **Validate**: Check if results make sense for your domain

## Next Steps

- Try the command-line example: `python example.py`
- Read the technical documentation in the code
- Explore the model implementations in `models.py`
- Customize for your specific needs

## Support

For issues or questions:
- Check GitHub Issues
- Read INSTALL.md for setup problems
- Review code comments for technical details
