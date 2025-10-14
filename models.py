"""
Machine Learning models for electrical load forecasting.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class LoadForecastingModels:
    """Collection of ML models for load forecasting."""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model."""
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        model = XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        return model
    
    def train_svr(self, X_train, y_train):
        """Train Support Vector Regression model."""
        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale'
        )
        model.fit(X_train, y_train)
        self.models['SVR'] = model
        return model
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train LSTM model."""
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train model
        if X_val is not None and y_val is not None:
            X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            model.fit(
                X_train_lstm, y_train,
                validation_data=(X_val_lstm, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
        else:
            model.fit(
                X_train_lstm, y_train,
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
        
        self.models['LSTM'] = model
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models."""
        print("Training Linear Regression...")
        self.train_linear_regression(X_train, y_train)
        
        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        
        print("Training Gradient Boosting...")
        self.train_gradient_boosting(X_train, y_train)
        
        print("Training XGBoost...")
        self.train_xgboost(X_train, y_train)
        
        print("Training SVR...")
        self.train_svr(X_train, y_train)
        
        print("Training LSTM...")
        self.train_lstm(X_train, y_train, X_val, y_val)
        
        print("All models trained successfully!")
        
    def predict(self, X_test):
        """Generate predictions from all models."""
        for name, model in self.models.items():
            if name == 'LSTM':
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                pred = model.predict(X_test_lstm, verbose=0).flatten()
            else:
                pred = model.predict(X_test)
            
            self.predictions[name] = pred
        
        return self.predictions
    
    def ensemble_prediction(self, method='mean'):
        """
        Create ensemble prediction from all models.
        
        Args:
            method: 'mean', 'median', or 'weighted'
            
        Returns:
            Ensemble predictions
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict() first.")
        
        pred_array = np.array(list(self.predictions.values()))
        
        if method == 'mean':
            ensemble = np.mean(pred_array, axis=0)
        elif method == 'median':
            ensemble = np.median(pred_array, axis=0)
        elif method == 'weighted':
            # Simple weighted average (can be customized based on model performance)
            weights = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2])  # Adjust based on validation
            ensemble = np.average(pred_array, axis=0, weights=weights)
        else:
            ensemble = np.mean(pred_array, axis=0)
        
        self.predictions['Ensemble'] = ensemble
        return ensemble


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
