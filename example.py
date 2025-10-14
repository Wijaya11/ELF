"""
Example script demonstrating the ELF workflow without Streamlit.
Run this after installing dependencies to test the core functionality.
"""

def run_example():
    """Run a simple example of the ELF workflow."""
    print("=" * 60)
    print("ELF - Electrical Load Forecasting Example")
    print("=" * 60)
    
    # Import required modules
    from data_utils import generate_sample_data, preprocess_data, split_data
    from models import LoadForecastingModels, calculate_metrics
    import numpy as np
    
    # Step 1: Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(days=180)
    print(f"   Generated {len(df)} hourly data points")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    X, y, scaler, dates = preprocess_data(df, lookback=24)
    print(f"   Created features with {X.shape[1]} dimensions")
    
    # Step 3: Split data
    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Step 4: Train models
    print("\n4. Training models...")
    models = LoadForecastingModels()
    
    # Split train into train and validation for LSTM
    val_split = int(len(X_train) * 0.9)
    X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
    y_train_sub, y_val = y_train[:val_split], y_train[val_split:]
    
    print("   - Training Linear Regression...")
    models.train_linear_regression(X_train, y_train)
    
    print("   - Training Random Forest...")
    models.train_random_forest(X_train, y_train)
    
    print("   - Training Gradient Boosting...")
    models.train_gradient_boosting(X_train, y_train)
    
    print("   - Training XGBoost...")
    models.train_xgboost(X_train, y_train)
    
    print("   - Training SVR...")
    models.train_svr(X_train, y_train)
    
    print("   - Training LSTM...")
    models.train_lstm(X_train, y_train, X_val, y_val, epochs=20)
    
    # Step 5: Generate predictions
    print("\n5. Generating predictions...")
    predictions = models.predict(X_test)
    ensemble_pred = models.ensemble_prediction(method='mean')
    
    # Step 6: Evaluate models
    print("\n6. Model Performance Metrics:")
    print("-" * 60)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 60)
    
    for name, pred in predictions.items():
        metrics = calculate_metrics(y_test, pred)
        print(f"{name:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R2']:<10.4f}")
    
    print("-" * 60)
    
    # Highlight ensemble performance
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
    print(f"\n✓ Ensemble Model Performance:")
    print(f"  - RMSE: {ensemble_metrics['RMSE']:.4f}")
    print(f"  - MAE: {ensemble_metrics['MAE']:.4f}")
    print(f"  - MAPE: {ensemble_metrics['MAPE']:.2f}%")
    print(f"  - R²: {ensemble_metrics['R2']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("To use the interactive UI, run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_example()
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install the required dependencies first:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
