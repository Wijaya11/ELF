# Quick Start Guide

Get ELF up and running in 3 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run the Application

```bash
streamlit run app.py
```

## 3. Use the Interface

1. **Configure settings** in the left sidebar:
   - Days of data: 365 (default)
   - Lookback period: 24 hours (default)
   - Training ratio: 0.8 (default)

2. **Click "Run Forecast"** button

3. **Wait for training** (2-5 minutes)

4. **Explore results**:
   - View performance metrics
   - Compare forecasts with actual values
   - Analyze errors
   - Compare all 6 models

## Alternative: Command Line Test

Run without UI to test core functionality:

```bash
python example.py
```

## Docker Deployment

Quick deployment with Docker:

```bash
# Build and run
docker-compose up

# Access at http://localhost:8501
```

## What You'll See

- ✅ 6 ML models trained
- 📊 Performance metrics (RMSE, MAE, MAPE, R²)
- 📈 Interactive visualizations
- 🎯 Ensemble forecast combining all models

## Need Help?

- Read [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions
- Check [INSTALL.md](INSTALL.md) for installation issues
- See [README.md](README.md) for project overview

## The 6 Models

1. **Linear Regression** - Simple baseline
2. **Random Forest** - Tree ensemble
3. **Gradient Boosting** - Sequential learning
4. **XGBoost** - Optimized boosting
5. **SVR** - Support vector machine
6. **LSTM** - Neural network

The **Ensemble** combines all 6 for the final forecast!

---

**Enjoy forecasting! ⚡**
