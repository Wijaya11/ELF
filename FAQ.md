# Frequently Asked Questions (FAQ)

## General Questions

### What is ELF?

ELF (Electrical Load Forecasting) is a tool that uses 6 different machine learning models to predict electrical load demand. It combines predictions from all models into an ensemble forecast for better accuracy.

### Why use multiple models?

Different models have different strengths:
- Some handle trends well
- Others capture seasonality
- Some work better with non-linear patterns
- The ensemble combines these strengths for more robust predictions

### Is this for real-world use?

The current version generates synthetic data for demonstration. However, the models and architecture can be adapted for real-world data by modifying the data loading functions.

## Installation & Setup

### What are the system requirements?

**Minimum**:
- Python 3.8+
- 4GB RAM
- 2 CPU cores

**Recommended**:
- Python 3.10+
- 8GB+ RAM
- 4+ CPU cores
- GPU (optional, speeds up LSTM)

### Why does installation take so long?

TensorFlow and other ML libraries are large packages. First-time installation can take 5-10 minutes depending on your internet speed.

### Can I use Python 3.7?

No, some dependencies require Python 3.8 or higher. We recommend Python 3.10 for best compatibility.

### Installation fails with "module not found"?

Try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Using the Application

### How long does training take?

Typical training time:
- Linear Regression: <1 second
- Random Forest: 2-5 seconds
- Gradient Boosting: 5-10 seconds
- XGBoost: 5-10 seconds
- SVR: 10-30 seconds
- LSTM: 1-3 minutes

Total: 2-5 minutes for all models

### Can I speed up training?

Yes:
1. Reduce number of days (e.g., 90 instead of 365)
2. Reduce lookback period
3. Lower LSTM epochs (in code)
4. Use GPU for LSTM (requires tensorflow-gpu)

### What do the metrics mean?

- **RMSE**: Average prediction error (lower is better)
- **MAE**: Mean absolute error (lower is better)
- **MAPE**: Error as percentage (lower is better)
- **R²**: How well model fits (higher is better, max = 1.0)

### Which ensemble method should I use?

- **Mean**: Best for general use (default)
- **Median**: Good if some models are unreliable
- **Weighted**: If you know which models perform better

### Why is LSTM so slow?

LSTM is a neural network that trains iteratively through multiple epochs. It's powerful but computationally expensive. Consider reducing epochs or data size for faster results.

## Data & Features

### Can I use my own data?

Yes! Modify `data_utils.py`:
1. Create a function to load your data
2. Ensure it has a datetime index
3. Ensure it has a 'load' column
4. Replace `generate_sample_data()` call in `app.py`

### What data format is required?

Required columns:
- Datetime index (hourly frequency)
- 'load' column (numerical values)

Example:
```
datetime            load
2023-01-01 00:00   45.2
2023-01-01 01:00   42.8
...
```

### How much historical data do I need?

**Minimum**: 30 days (720 hours)
**Recommended**: 365 days (8,760 hours)
**Optimal**: 2+ years for seasonal patterns

### What features are used?

**Temporal**:
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Day of year (1-365)

**Lagged**:
- Previous N hours of load (default N=24)

## Models & Predictions

### Which model is most accurate?

It varies by dataset! That's why we use ensemble. Typically:
- **XGBoost** or **LSTM** performs best individually
- **Ensemble** usually outperforms all individual models

### Can I disable certain models?

Yes, modify `app.py` and `models.py` to skip training specific models. Comment out the training call and prediction retrieval.

### How does the ensemble work?

The ensemble combines predictions using:
- **Mean**: Average of all model predictions
- **Median**: Middle value of predictions
- **Weighted**: Custom weights per model

### Why do models disagree?

Models use different algorithms:
- Tree-based models see patterns differently than neural networks
- Some models overfit, others underfit
- Different models handle non-linearity differently

This disagreement is actually useful - the ensemble leverages it!

### Can I add more models?

Yes! See CONTRIBUTING.md for instructions on adding new models.

## Errors & Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"

Run: `pip install -r requirements.txt`

### "Address already in use" error

Port 8501 is busy. Try:
```bash
streamlit run app.py --server.port 8502
```

### Models predict negative values?

This can happen with Linear Regression. Solutions:
1. Use ensemble (usually non-negative)
2. Post-process to clip at zero
3. Use only tree-based or neural models

### "Out of memory" error

Reduce data size:
1. Decrease number of days
2. Decrease lookback period
3. Close other applications
4. Use batch processing

### Training seems stuck at LSTM?

LSTM training is slow. Check terminal for progress. Each epoch should show loss decreasing. Wait 2-5 minutes.

### Results look bad (R² < 0.5)?

Possible causes:
1. **Not enough data**: Increase to 365+ days
2. **Wrong lookback**: Try 24-48 hours
3. **Data quality**: Check for missing values
4. **Model settings**: May need tuning

## Performance & Optimization

### Can I use GPU?

Yes, for LSTM:
1. Install tensorflow-gpu instead of tensorflow
2. Ensure CUDA and cuDNN are installed
3. TensorFlow will automatically use GPU

### Can models run in parallel?

Currently sequential for progress tracking. You could modify to run in parallel using `multiprocessing` or `joblib`.

### How do I save trained models?

Currently not implemented. To add:
1. Use `pickle` for sklearn models
2. Use `model.save()` for LSTM
3. Load before prediction

### Can this scale to big data?

Current design for moderate data (up to 2 years hourly). For larger:
1. Use data batching
2. Implement incremental learning
3. Consider distributed training
4. Use model persistence

## Docker & Deployment

### How do I use Docker?

```bash
# Build and run
docker-compose up

# Or manually
docker build -t elf .
docker run -p 8501:8501 elf
```

### Can I deploy to cloud?

Yes! Deploy to:
- **Heroku**: Use streamlit-specific buildpack
- **AWS**: EC2, ECS, or App Runner
- **Google Cloud**: Cloud Run or Compute Engine
- **Azure**: App Service or Container Instances

### How do I share the app?

Options:
1. **Streamlit Cloud**: Free hosting for public repos
2. **Docker**: Share container
3. **Cloud deployment**: AWS, GCP, Azure
4. **Local network**: Run on server, share URL

## Advanced Usage

### Can I forecast multiple days ahead?

Current version: 1-step ahead (next hour)
To add multi-step: Modify to iteratively predict or use sequence-to-sequence LSTM

### Can I include weather data?

Yes! Add weather features to `preprocess_data()`:
- Temperature
- Humidity  
- Wind speed
- Cloud cover

### Can I add holiday effects?

Yes! Create holiday indicator in `preprocess_data()`:
```python
df['is_holiday'] = df.index.date.isin(holiday_list)
```

### How do I tune hyperparameters?

Modify `config.py` or use grid search:
```python
from sklearn.model_selection import GridSearchCV
# Define param grid and search
```

## Support & Community

### Where do I get help?

1. Check this FAQ
2. Read USER_GUIDE.md
3. Check INSTALL.md for setup issues
4. Create GitHub issue

### How do I report bugs?

Create a GitHub issue with:
- Description
- Steps to reproduce
- Error messages
- Environment details

### Can I contribute?

Yes! See CONTRIBUTING.md for guidelines.

### Is there a community?

Check GitHub Discussions and Issues for community interaction.

## Licensing & Usage

### What license is this under?

MIT License - you can use, modify, and distribute freely.

### Can I use this commercially?

Yes, the MIT License allows commercial use.

### Do I need to credit the project?

Not required, but appreciated! Mention in your documentation or README.

### Can I modify the code?

Yes! Fork it, modify it, make it your own. Consider contributing improvements back.

---

**Still have questions?** Create an issue on GitHub!
