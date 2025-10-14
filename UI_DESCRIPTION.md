# ELF User Interface Description

## Overview

The ELF Streamlit interface provides an intuitive, interactive experience for electrical load forecasting. This document describes what users will see when running the application.

## Main Interface Layout

### Header Section
```
⚡ ELF - Electrical Load Forecasting
An AI-powered tool using 6 Machine Learning methods with ensemble forecasting
```

### Sidebar (Left Panel)

#### ⚙️ Configuration Panel

**Data Settings**
- Slider: Number of days to generate (30-730, default: 365)
- Slider: Lookback period hours (12-72, default: 24)
- Slider: Training data ratio (0.6-0.9, default: 0.8)

**Ensemble Settings**
- Dropdown: Ensemble Method
  - Mean (default)
  - Median
  - Weighted

**Action Button**
```
🚀 Run Forecast (large primary button)
```

### Main Content Area

#### Before Running Forecast

**Welcome Message**
```
👈 Configure the settings in the sidebar and click '🚀 Run Forecast' to start!
```

**About Section**
- Description of 6 ML methods
- Feature list
- Quick start instructions

#### After Running Forecast

### Step-by-Step Progress Display

**Step 1: Data Generation**
```
📊 Step 1: Generating synthetic electrical load data...
✅ Generated 8,760 hourly data points
```

- Expandable section showing raw data preview
- Interactive time series plot of raw load data

**Step 2: Preprocessing**
```
🔧 Step 2: Preprocessing data and creating features...
✅ Created features with 28 dimensions
- Training samples: 6,998
- Testing samples: 1,750
```

**Step 3: Model Training**
```
🤖 Step 3: Training 6 ML models...
```
- Progress bar (0-100%)
- Status text showing current model being trained
- Updates for each model:
  - Training Linear Regression...
  - Training Random Forest...
  - Training Gradient Boosting...
  - Training XGBoost...
  - Training SVR...
  - Training LSTM...

```
✅ All 6 models trained successfully!
```

**Step 4: Predictions**
```
🔮 Step 4: Generating predictions...
✅ Predictions generated!
```

**Step 5: Evaluation**
```
📊 Step 5: Evaluating model performance...
```

---

### Results Section

#### 📊 Results Header

**Model Performance Metrics Table**

Styled table with gradient colors:
```
| Model              | RMSE    | MAE     | MAPE    | R²      |
|--------------------|---------|---------|---------|---------|
| Linear Regression  | 5.2341  | 4.1234  | 7.23%   | 0.8512  |
| Random Forest      | 4.8912  | 3.8976  | 6.84%   | 0.8789  |
| Gradient Boosting  | 4.7654  | 3.7821  | 6.65%   | 0.8856  |
| XGBoost           | 4.6234  | 3.6543  | 6.42%   | 0.8923  |
| SVR               | 5.0123  | 4.0234  | 7.05%   | 0.8645  |
| LSTM              | 4.5678  | 3.5987  | 6.32%   | 0.8967  |
| Ensemble          | 4.4321  | 3.4876  | 6.12%   | 0.9012  |
```
- Green background for best values
- Red background for worst values

**Ensemble Performance Metrics (4 columns)**
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Ensemble RMSE   │  │ Ensemble MAE    │  │ Ensemble MAPE   │  │ Ensemble R²     │
│   4.4321        │  │   3.4876        │  │   6.12%         │  │   0.9012        │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

### 📈 Forecast Visualization (3 Tabs)

#### Tab 1: Forecast Comparison

**Main Plot - Actual vs Ensemble**
- Interactive Plotly line chart
- X-axis: DateTime
- Y-axis: Load (MW)
- Black line: Actual load
- Red dashed line: Ensemble forecast
- Hover shows exact values
- Zoom, pan, and download controls

**Detailed View - Last 7 Days**
- Similar plot but with markers
- More granular view of recent predictions
- Easier to see individual hour differences

#### Tab 2: Individual Models

**All Models Comparison Plot**
- Black thick line: Actual load
- 6 colored lines: Each model's predictions
- Red line: Ensemble
- Legend on the right
- Toggle models on/off by clicking legend
- Interactive hover showing all values at once

Colors:
- Blue: Linear Regression
- Orange: Random Forest
- Green: Gradient Boosting
- Red: XGBoost
- Purple: SVR
- Brown: LSTM
- Red: Ensemble

#### Tab 3: Error Analysis

**Two Column Layout:**

Left Column:
**Error Distribution Histogram**
- Blue bars showing frequency of errors
- X-axis: Error (MW)
- Y-axis: Frequency
- Should be centered around zero
- Shows if predictions are biased

Right Column:
**Actual vs Predicted Scatter**
- Blue dots: Each prediction
- Red dashed line: Perfect prediction (y=x)
- Points near line = good predictions
- X-axis: Actual Load
- Y-axis: Predicted Load

**Full Width:**
**Error Over Time Line Chart**
- Red line showing error at each time point
- Black dashed line at y=0
- Positive values = over-prediction
- Negative values = under-prediction
- X-axis: DateTime
- Y-axis: Error (MW)

---

### 🏆 Model Comparison

**Two Column Bar Charts:**

Left:
**RMSE Comparison**
- Horizontal bars for each model
- Color gradient (red=bad, green=good)
- Sorted by RMSE (ascending)
- Shows which models have lowest error

Right:
**R² Score Comparison**
- Horizontal bars for each model
- Color gradient (green=good, red=bad)
- Sorted by R² (descending)
- Shows which models fit best

---

### Final Success Message
```
✅ Forecasting complete! The ensemble model combines all 6 ML methods for 
   optimal prediction.
```

## Color Scheme

**Primary Colors:**
- Blue (#1f77b4): Primary UI elements
- Red (#d62728): Ensemble/important highlights
- Green: Success messages
- Orange: Info messages
- Black: Actual values/text

**Background:**
- Light gray (#f0f2f6): Metric cards
- White: Main background

## Interactive Elements

**Clickable:**
- Sidebar controls (sliders, dropdowns)
- Plot legends (toggle series)
- Expandable sections (data preview)

**Hoverable:**
- All plot points show exact values
- Buttons show hover effects
- Table cells can be selected

**Downloadable:**
- All plots have camera icon for PNG export
- Can select and copy table data

## Responsive Design

- Adapts to browser width
- Sidebar collapses on narrow screens
- Plots resize to fit container
- Tables scroll horizontally if needed

## Loading States

- Spinners during computation
- Progress bars during training
- Status messages at each step
- Disabled button during processing

## Error Handling

- Graceful error messages if something fails
- Input validation on sliders
- Clear error descriptions
- Suggestions for fixing issues

## Performance

**Initial Load:**
- <2 seconds to show interface

**After Clicking Run:**
- Data generation: ~1 second
- Preprocessing: ~1 second  
- Model training: 2-5 minutes
- Predictions: ~1 second
- Visualization: ~1 second

**Total:** ~3-6 minutes for complete workflow

## Accessibility

- Clear labels on all controls
- Good color contrast
- Keyboard navigable
- Screen reader compatible (basic)

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers (limited)

## Screen Sizes

**Optimal:**
- Desktop: 1920x1080 or higher
- Minimum: 1280x720

**Mobile:**
- Works but not optimal
- Better in landscape mode

## Tips for Best Experience

1. Use full screen mode
2. Wait for all models to train
3. Explore interactive plots (zoom, hover)
4. Try different ensemble methods
5. Experiment with parameters
6. Compare tab views
7. Download interesting plots

## What Makes It User-Friendly

✅ **No coding required**
✅ **One-click operation**
✅ **Visual progress tracking**
✅ **Interactive visualizations**
✅ **Clear metrics display**
✅ **Helpful guidance text**
✅ **Professional appearance**
✅ **Fast and responsive**

---

This interface provides a complete, professional experience for electrical load forecasting without requiring any programming knowledge from the user.
