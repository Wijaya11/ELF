"""
Streamlit app for Electrical Load Forecasting (ELF).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_utils import generate_sample_data, preprocess_data, split_data
from models import LoadForecastingModels, calculate_metrics
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="ELF - Electrical Load Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Title
    st.markdown('<h1 class="main-header">⚡ ELF - Electrical Load Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("### An AI-powered tool using 6 Machine Learning methods with ensemble forecasting")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data configuration
    st.sidebar.subheader("Data Settings")
    days = st.sidebar.slider("Number of days to generate", 30, 730, 365)
    lookback = st.sidebar.slider("Lookback period (hours)", 12, 72, 24)
    train_ratio = st.sidebar.slider("Training data ratio", 0.6, 0.9, 0.8, 0.05)
    
    # Ensemble method
    st.sidebar.subheader("Ensemble Settings")
    ensemble_method = st.sidebar.selectbox(
        "Ensemble Method",
        ["mean", "median", "weighted"]
    )
    
    # Action button
    run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")
    
    # Main content
    if run_forecast:
        with st.spinner("Generating data and training models..."):
            # Generate and preprocess data
            st.info("📊 Step 1: Generating synthetic electrical load data...")
            df = generate_sample_data(days=days)
            
            st.success(f"✅ Generated {len(df)} hourly data points")
            
            # Show data preview
            with st.expander("📈 View Raw Data"):
                st.dataframe(df.head(50), use_container_width=True)
                
                # Plot raw data
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['load'],
                    mode='lines',
                    name='Load',
                    line=dict(color='#1f77b4', width=1)
                ))
                fig.update_layout(
                    title="Raw Electrical Load Data",
                    xaxis_title="DateTime",
                    yaxis_title="Load (MW)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Preprocess data
            st.info("🔧 Step 2: Preprocessing data and creating features...")
            X, y, scaler, dates = preprocess_data(df, lookback=lookback)
            X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=train_ratio)
            
            st.success(f"✅ Created features with {X.shape[1]} dimensions")
            st.write(f"- Training samples: {len(X_train)}")
            st.write(f"- Testing samples: {len(X_test)}")
            
            # Train models
            st.info("🤖 Step 3: Training 6 ML models...")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Split train into train and validation
            val_split = int(len(X_train) * 0.9)
            X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
            y_train_sub, y_val = y_train[:val_split], y_train[val_split:]
            
            # Initialize model collection
            models = LoadForecastingModels()
            
            # Train models one by one with progress updates
            model_names = [
                'Linear Regression',
                'Random Forest',
                'Gradient Boosting',
                'XGBoost',
                'SVR',
                'LSTM'
            ]
            
            for i, name in enumerate(model_names):
                status_text.text(f"Training {name}...")
                if name == 'Linear Regression':
                    models.train_linear_regression(X_train, y_train)
                elif name == 'Random Forest':
                    models.train_random_forest(X_train, y_train)
                elif name == 'Gradient Boosting':
                    models.train_gradient_boosting(X_train, y_train)
                elif name == 'XGBoost':
                    models.train_xgboost(X_train, y_train)
                elif name == 'SVR':
                    models.train_svr(X_train, y_train)
                elif name == 'LSTM':
                    models.train_lstm(X_train, y_train, X_val, y_val, epochs=30)
                
                progress_bar.progress((i + 1) / len(model_names))
            
            status_text.text("All models trained successfully!")
            st.success("✅ All 6 models trained successfully!")
            
            # Generate predictions
            st.info("🔮 Step 4: Generating predictions...")
            predictions = models.predict(X_test)
            ensemble_pred = models.ensemble_prediction(method=ensemble_method)
            
            st.success("✅ Predictions generated!")
            
            # Calculate metrics
            st.info("📊 Step 5: Evaluating model performance...")
            
            metrics_df = []
            for name, pred in predictions.items():
                metrics = calculate_metrics(y_test, pred)
                metrics['Model'] = name
                metrics_df.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_df)
            metrics_df = metrics_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2']]
            
            # Display results
            st.markdown("---")
            st.header("📊 Results")
            
            # Metrics table
            st.subheader("Model Performance Metrics")
            
            # Style the dataframe
            styled_df = metrics_df.style.background_gradient(
                subset=['RMSE', 'MAE', 'MAPE'],
                cmap='RdYlGn_r'
            ).background_gradient(
                subset=['R2'],
                cmap='RdYlGn'
            ).format({
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'MAPE': '{:.2f}%',
                'R2': '{:.4f}'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Highlight best ensemble performance
            ensemble_metrics = metrics_df[metrics_df['Model'] == 'Ensemble'].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ensemble RMSE", f"{ensemble_metrics['RMSE']:.4f}")
            with col2:
                st.metric("Ensemble MAE", f"{ensemble_metrics['MAE']:.4f}")
            with col3:
                st.metric("Ensemble MAPE", f"{ensemble_metrics['MAPE']:.2f}%")
            with col4:
                st.metric("Ensemble R²", f"{ensemble_metrics['R2']:.4f}")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Forecast Visualization")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Forecast Comparison", "Individual Models", "Error Analysis"])
            
            with tab1:
                # Plot actual vs ensemble prediction
                test_dates = dates[len(X_train):]
                
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_test,
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=2)
                ))
                
                # Ensemble prediction
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=ensemble_pred,
                    mode='lines',
                    name='Ensemble Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Actual Load vs Ensemble Forecast",
                    xaxis_title="DateTime",
                    yaxis_title="Load (MW)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed view (last 7 days)
                last_week = 24 * 7
                if len(test_dates) > last_week:
                    fig_week = go.Figure()
                    
                    fig_week.add_trace(go.Scatter(
                        x=test_dates[-last_week:],
                        y=y_test[-last_week:],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='black', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig_week.add_trace(go.Scatter(
                        x=test_dates[-last_week:],
                        y=ensemble_pred[-last_week:],
                        mode='lines+markers',
                        name='Ensemble Forecast',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig_week.update_layout(
                        title="Detailed View - Last 7 Days",
                        xaxis_title="DateTime",
                        yaxis_title="Load (MW)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_week, use_container_width=True)
            
            with tab2:
                # Plot all model predictions
                fig_all = go.Figure()
                
                # Actual values
                fig_all.add_trace(go.Scatter(
                    x=test_dates,
                    y=y_test,
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=3)
                ))
                
                # All model predictions
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                for i, (name, pred) in enumerate(predictions.items()):
                    fig_all.add_trace(go.Scatter(
                        x=test_dates,
                        y=pred,
                        mode='lines',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        opacity=0.7
                    ))
                
                fig_all.update_layout(
                    title="All Model Predictions Comparison",
                    xaxis_title="DateTime",
                    yaxis_title="Load (MW)",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_all, use_container_width=True)
            
            with tab3:
                # Error analysis
                ensemble_error = y_test - ensemble_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Error distribution
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=ensemble_error,
                        nbinsx=50,
                        name='Error Distribution',
                        marker=dict(color='#1f77b4')
                    ))
                    fig_hist.update_layout(
                        title="Ensemble Forecast Error Distribution",
                        xaxis_title="Error (MW)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Scatter plot: actual vs predicted
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=y_test,
                        y=ensemble_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='#1f77b4', size=5, opacity=0.6)
                    ))
                    
                    # Add perfect prediction line
                    min_val, max_val = y_test.min(), y_test.max()
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_scatter.update_layout(
                        title="Actual vs Predicted Load",
                        xaxis_title="Actual Load (MW)",
                        yaxis_title="Predicted Load (MW)",
                        height=400
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Error over time
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(
                    x=test_dates,
                    y=ensemble_error,
                    mode='lines',
                    name='Error',
                    line=dict(color='#d62728', width=1)
                ))
                fig_error.add_hline(y=0, line_dash="dash", line_color="black")
                fig_error.update_layout(
                    title="Forecast Error Over Time",
                    xaxis_title="DateTime",
                    yaxis_title="Error (MW)",
                    height=400
                )
                st.plotly_chart(fig_error, use_container_width=True)
            
            # Model comparison bar chart
            st.markdown("---")
            st.subheader("🏆 Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rmse = px.bar(
                    metrics_df.sort_values('RMSE'),
                    x='Model',
                    y='RMSE',
                    title='RMSE Comparison (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_rmse.update_layout(height=400)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                fig_r2 = px.bar(
                    metrics_df.sort_values('R2', ascending=False),
                    x='Model',
                    y='R2',
                    title='R² Score Comparison (Higher is Better)',
                    color='R2',
                    color_continuous_scale='RdYlGn'
                )
                fig_r2.update_layout(height=400)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            st.success("✅ Forecasting complete! The ensemble model combines all 6 ML methods for optimal prediction.")
    
    else:
        # Welcome message
        st.info("👈 Configure the settings in the sidebar and click '🚀 Run Forecast' to start!")
        
        st.markdown("### 🔍 About ELF")
        st.markdown("""
        The Electrical Load Forecasting (ELF) tool uses **6 powerful machine learning methods** to predict electrical load demand:
        
        1. **Linear Regression** - Simple baseline model
        2. **Random Forest** - Ensemble of decision trees
        3. **Gradient Boosting** - Sequential ensemble method
        4. **XGBoost** - Optimized gradient boosting
        5. **SVR (Support Vector Regression)** - Kernel-based regression
        6. **LSTM (Long Short-Term Memory)** - Deep learning for time series
        
        The final forecast is calculated using an **ensemble method** that combines predictions from all 6 models,
        providing more robust and accurate forecasts than any single model alone.
        
        ### 📋 Features
        - Interactive data generation with customizable parameters
        - Comprehensive model training and evaluation
        - Real-time visualization of forecasts and errors
        - Multiple ensemble methods (mean, median, weighted)
        - Detailed performance metrics and comparisons
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Adjust the data and model settings in the sidebar
        2. Click the **Run Forecast** button
        3. Wait for models to train (may take a few minutes)
        4. Explore the results and visualizations
        """)


if __name__ == "__main__":
    main()
