# ⚡ ELF - Electrical Load Forecasting

An advanced Electrical Load Forecasting tool powered by **6 Machine Learning methods** with ensemble forecasting capabilities. Built with Streamlit for an interactive and user-friendly experience.

## 🌟 Features

- **6 Machine Learning Models**: 
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Regression (SVR)
  - LSTM (Long Short-Term Memory Neural Network)

- **Ensemble Forecasting**: Combines predictions from all models using mean, median, or weighted averaging
- **Interactive UI/UX**: Built with Streamlit for real-time visualization and interaction
- **Comprehensive Metrics**: RMSE, MAE, MAPE, R² score for model evaluation
- **Rich Visualizations**: 
  - Time series plots
  - Model comparison charts
  - Error analysis
  - Actual vs Predicted comparisons

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Wijaya11/ELF.git
cd ELF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How It Works

1. **Data Generation**: Creates synthetic electrical load data with realistic daily and weekly patterns
2. **Feature Engineering**: Extracts temporal features (hour, day, month) and creates lagged features
3. **Model Training**: Trains all 6 ML models on the prepared data
4. **Prediction**: Generates forecasts from each model
5. **Ensemble**: Combines predictions using the selected ensemble method
6. **Evaluation**: Calculates performance metrics and displays results

## 📊 Configuration Options

- **Number of days**: Amount of historical data to generate (30-730 days)
- **Lookback period**: Number of previous hours used as features (12-72 hours)
- **Training ratio**: Proportion of data used for training (60-90%)
- **Ensemble method**: How to combine model predictions (mean/median/weighted)

## 📁 Project Structure

```
ELF/
├── app.py              # Streamlit application
├── models.py           # ML model implementations
├── data_utils.py       # Data processing utilities
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## 🔧 Technologies Used

- **Python 3.8+**
- **Streamlit**: Interactive web interface
- **scikit-learn**: Traditional ML models
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: LSTM neural network
- **Pandas & NumPy**: Data manipulation
- **Plotly**: Interactive visualizations

## 📈 Model Performance

The ensemble method typically outperforms individual models by leveraging the strengths of each approach:
- Tree-based models (RF, GB, XGB) capture non-linear patterns
- SVR handles complex relationships
- LSTM captures temporal dependencies
- Linear Regression provides a stable baseline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Wijaya11

## 🙏 Acknowledgments

Built with modern ML and data science tools to provide accurate electrical load forecasting for better energy management and planning.