#!/usr/bin/env python3
"""
Question 3: Machine Learning and Feature Engineering

This script provides a starting point for building a predictive model for price movements
using market data.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import warnings
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Suppress warnings
warnings.filterwarnings('ignore')

# Import data utilities
from quant_test.data_utils import (
    load_reshaped_data,
    calculate_mid_prices,
    calculate_book_imbalance,
    calculate_spreads,
    get_sample_data
)

def create_target(
    df: pl.DataFrame,
    instrument_id: str,
    forward_periods: int = 50,
    target_type: str = 'returns',
    use_time: bool = False,
    time_window: float = 3.0,  # seconds
) -> pl.DataFrame:
    """
    Create target variables for prediction.
    
    Args:
        df: DataFrame with market data
        instrument_id: Instrument ID
        forward_periods: Number of periods to look forward for target
        target_type: Type of target ('returns', 'direction', 'mid_price')
        use_time: Whether to use time-based forward window instead of row-based
        time_window: Time window in seconds (only used if use_time is True)
        
    Returns:
        DataFrame with target variables
    """
    # Get mid price
    mid_col = f'{instrument_id}_mid_price'
    timestamp_col = f'{instrument_id}_timestamp'
    
    # Create a copy to avoid modifying the original
    result_df = df.clone()
    
    if use_time:
        # Time-based forward window
        result_df = result_df.with_columns([
            pl.col(timestamp_col).alias('timestamp'),
            pl.col(mid_col).alias('mid_price')
        ])
        
        # Convert nanoseconds to seconds for easier calculation
        result_df = result_df.with_columns([
            (pl.col('timestamp') / 1e9).alias('timestamp_seconds')
        ])
        
        # For each row, find the future price after time_window seconds
        future_prices = []
        future_timestamps = []
        
        for i, row in result_df.iter_rows(named=True):
            current_time = row['timestamp_seconds']
            target_time = current_time + time_window
            
            # Find the first row with timestamp >= target_time
            future_rows = result_df.filter(pl.col('timestamp_seconds') >= target_time)
            
            if future_rows.height > 0:
                future_price = future_rows[0, 'mid_price']
                future_timestamp = future_rows[0, 'timestamp']
            else:
                # If no future data, use the last available price
                future_price = result_df[-1, 'mid_price']
                future_timestamp = result_df[-1, 'timestamp']
                
            future_prices.append(future_price)
            future_timestamps.append(future_timestamp)
        
        # Add future prices to the dataframe
        result_df = result_df.with_columns([
            pl.Series(future_prices).alias('future_mid_price'),
            pl.Series(future_timestamps).alias('future_timestamp')
        ])
    else:
        # Row-based forward window
        result_df = result_df.with_columns([
            pl.col(mid_col).shift(-forward_periods).alias('future_mid_price')
        ])
    
    # Calculate returns
    result_df = result_df.with_columns([
        ((pl.col('future_mid_price') - pl.col(mid_col)) / pl.col(mid_col)).alias('future_returns')
    ])
    
    # Calculate direction (1 for up, 0 for down or unchanged)
    result_df = result_df.with_columns([
        (pl.col('future_returns') > 0).cast(pl.Int8).alias('future_direction')
    ])
    
    # Drop rows with NaN in target
    result_df = result_df.drop_nulls(subset=['future_mid_price'])
    
    return result_df

def engineer_features(df: pl.DataFrame, instrument_id: str) -> Tuple[pl.DataFrame, List[str]]:
    """
    Engineer features for the prediction model.
    
    Args:
        df: DataFrame with market data
        instrument_id: Instrument ID
        
    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    # Get relevant columns
    mid_col = f'{instrument_id}_mid_price'
    imbalance_col = f'{instrument_id}_book_imbalance'
    spread_col = f'{instrument_id}_spread'
    bid_vol_1_col = f'{instrument_id}_bid_size_1'
    ask_vol_1_col = f'{instrument_id}_ask_size_1'
    
    # Create a copy to avoid modifying the original
    result_df = df.clone()
    
    # Basic features
    
    # 1. Returns over different horizons
    for window in [1, 5, 10, 20, 50]:
        result_df = result_df.with_columns([
            ((pl.col(mid_col) / pl.col(mid_col).shift(window)) - 1).alias(f'returns_{window}')
        ])
    
    # 2. Moving averages
    for window in [5, 10, 20, 50, 100]:
        result_df = result_df.with_columns([
            pl.col(mid_col).rolling_mean(window).alias(f'ma_{window}')
        ])
    
    # 3. Price distance from moving averages
    for window in [5, 10, 20, 50, 100]:
        ma_col = f'ma_{window}'
        result_df = result_df.with_columns([
            ((pl.col(mid_col) / pl.col(ma_col)) - 1).alias(f'price_dist_ma_{window}')
        ])
    
    # 4. Volatility (standard deviation of returns)
    for window in [10, 20, 50]:
        returns_col = f'returns_{1}'
        result_df = result_df.with_columns([
            pl.col(returns_col).rolling_std(window).alias(f'volatility_{window}')
        ])
    
    # 5. Book imbalance features
    result_df = result_df.with_columns([
        pl.col(imbalance_col).rolling_mean(10).alias('imbalance_ma_10'),
        pl.col(imbalance_col).rolling_mean(50).alias('imbalance_ma_50')
    ])
    
    # 6. Spread features
    result_df = result_df.with_columns([
        pl.col(spread_col).rolling_mean(10).alias('spread_ma_10'),
        pl.col(spread_col).rolling_mean(50).alias('spread_ma_50'),
        (pl.col(spread_col) / pl.col(mid_col)).alias('relative_spread')
    ])
    
    # 7. Volume features
    result_df = result_df.with_columns([
        (pl.col(bid_vol_1_col) / pl.col(ask_vol_1_col)).alias('bid_ask_vol_ratio'),
        pl.col(bid_vol_1_col).rolling_mean(10).alias('bid_vol_ma_10'),
        pl.col(ask_vol_1_col).rolling_mean(10).alias('ask_vol_ma_10')
    ])
    
    # 8. Technical indicators
    
    # RSI (Relative Strength Index)
    for window in [14, 28]:
        # Calculate price changes
        result_df = result_df.with_columns([
            (pl.col(mid_col) - pl.col(mid_col).shift(1)).alias('price_change')
        ])
        
        # Calculate gains and losses
        result_df = result_df.with_columns([
            pl.when(pl.col('price_change') > 0).then(pl.col('price_change')).otherwise(0).alias('gain'),
            pl.when(pl.col('price_change') < 0).then(-pl.col('price_change')).otherwise(0).alias('loss')
        ])
        
        # Calculate average gains and losses
        result_df = result_df.with_columns([
            pl.col('gain').rolling_mean(window).alias(f'avg_gain_{window}'),
            pl.col('loss').rolling_mean(window).alias(f'avg_loss_{window}')
        ])
        
        # Calculate RS and RSI
        result_df = result_df.with_columns([
            (pl.col(f'avg_gain_{window}') / pl.col(f'avg_loss_{window}')).alias(f'rs_{window}')
        ])
        
        result_df = result_df.with_columns([
            (100 - (100 / (1 + pl.col(f'rs_{window}')))).alias(f'rsi_{window}')
        ])
    
    # Drop temporary columns
    result_df = result_df.drop(['price_change', 'gain', 'loss'])
    
    # Drop rows with NaN values (from rolling windows)
    result_df = result_df.drop_nulls()
    
    # Get list of feature columns (excluding target variables and original data)
    exclude_patterns = ['future_', 'timestamp', '_timestamp', '_bid_price_', '_ask_price_', '_bid_size_', '_ask_size_']
    feature_cols = [col for col in result_df.columns if not any(pattern in col for pattern in exclude_patterns)]
    
    # Remove the original columns we don't want as features
    feature_cols = [col for col in feature_cols if col not in [mid_col, imbalance_col, spread_col]]
    
    return result_df, feature_cols

def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Normalize features using the specified method.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        method: Normalization method ('standard', 'robust', 'minmax')
        
    Returns:
        Tuple of (normalized X_train, normalized X_val, normalized X_test, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit on training data only
    X_train_norm = scaler.fit_transform(X_train)
    
    # Transform validation and test data
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler

def train_and_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'linear',
    problem_type: str = 'regression',
    feature_names: List[str] = None
) -> Dict:
    """
    Train and evaluate a model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        model_type: Type of model ('linear', 'rf', 'xgb')
        problem_type: Type of problem ('regression', 'classification')
        feature_names: List of feature names
        
    Returns:
        Dictionary with model and evaluation results
    """
    # Initialize model
    if problem_type == 'regression':
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    elif problem_type == 'classification':
        if model_type == 'linear':
            model = LogisticRegression(random_state=42)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    results = {
        'model': model,
        'model_type': model_type,
        'problem_type': problem_type,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred
    }
    
    if problem_type == 'regression':
        # Calculate MSE
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Calculate RMSE
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Add to results
        results.update({
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse
        })
    elif problem_type == 'classification':
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate precision, recall, and F1 score
        train_precision = precision_score(y_train, y_train_pred)
        val_precision = precision_score(y_val, y_val_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        train_recall = recall_score(y_train, y_train_pred)
        val_recall = recall_score(y_val, y_val_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        train_f1 = f1_score(y_train, y_train_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Add to results
        results.update({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1
        })
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = {
            feature: importance
            for feature, importance in zip(feature_names, model.feature_importances_)
        }
        results['feature_importance'] = feature_importance
    elif hasattr(model, 'coef_') and feature_names is not None:
        if problem_type == 'regression' or len(model.coef_.shape) == 1:
            feature_importance = {
                feature: abs(importance)
                for feature, importance in zip(feature_names, model.coef_)
            }
            results['feature_importance'] = feature_importance
    
    return results

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('plots/q3_feature_importance.png')
    elif hasattr(model, 'coef_'):
        # For linear models
        if len(model.coef_.shape) == 1:
            # Regression or binary classification
            coefs = model.coef_
        else:
            # Multiclass classification
            coefs = model.coef_[0]
        
        # Sort coefficients by absolute value
        indices = np.argsort(np.abs(coefs))[::-1]
        
        plt.title('Feature Coefficients')
        plt.bar(range(len(indices)), coefs[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('plots/q3_feature_coefficients.png')
    else:
        print("Model doesn't have feature_importances_ or coef_ attribute")

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('plots/q3_actual_vs_predicted.png')
    
    # Plot actual and predicted over time
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual and Predicted Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/q3_time_series_prediction.png')

def main():
    """Run the machine learning pipeline on sample data."""
    print("Running Question 3: Machine Learning and Feature Engineering")
    
    # Load sample data
    df = get_sample_data()
    instrument_id = 'BTC-USD'
    
    # Calculate mid prices, book imbalance, and spreads
    df = calculate_mid_prices(df, instrument_id)
    df = calculate_book_imbalance(df, instrument_id)
    df = calculate_spreads(df, instrument_id)
    
    # Create target variable
    df = create_target(
        df=df,
        instrument_id=instrument_id,
        forward_periods=50,
        target_type='returns'
    )
    
    # Engineer features
    df, feature_cols = engineer_features(df, instrument_id)
    
    # Prepare data for modeling
    X = df.select(feature_cols).to_numpy()
    y = df.select(['future_returns']).to_numpy().flatten()
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        method='robust'
    )
    
    # Train and evaluate regression model
    regression_results = train_and_evaluate_model(
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        model_type='rf',
        problem_type='regression',
        feature_names=feature_cols
    )
    
    # Print regression results
    print("\nRegression Results:")
    print(f"Train RMSE: {regression_results['train_rmse']:.6f}")
    print(f"Validation RMSE: {regression_results['val_rmse']:.6f}")
    print(f"Test RMSE: {regression_results['test_rmse']:.6f}")
    
    # Plot feature importance for regression model
    plot_feature_importance(regression_results['model'], feature_cols)
    
    # Plot predictions for regression model
    plot_predictions(y_test, regression_results['y_test_pred'])
    
    # Create binary target for classification
    y_train_class = (y_train > 0).astype(int)
    y_val_class = (y_val > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)
    
    # Train and evaluate classification model
    classification_results = train_and_evaluate_model(
        X_train=X_train_norm,
        y_train=y_train_class,
        X_val=X_val_norm,
        y_val=y_val_class,
        X_test=X_test_norm,
        y_test=y_test_class,
        model_type='rf',
        problem_type='classification',
        feature_names=feature_cols
    )
    
    # Print classification results
    print("\nClassification Results:")
    print(f"Train Accuracy: {classification_results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {classification_results['val_accuracy']:.4f}")
    print(f"Test Accuracy: {classification_results['test_accuracy']:.4f}")
    print(f"Test Precision: {classification_results['test_precision']:.4f}")
    print(f"Test Recall: {classification_results['test_recall']:.4f}")
    print(f"Test F1 Score: {classification_results['test_f1']:.4f}")
    
    # Print top features
    if 'feature_importance' in regression_results:
        print("\nTop 10 Features (Regression):")
        sorted_features = sorted(
            regression_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:10]:
            print(f"{feature}: {importance:.6f}")
    
    print("Question 3 completed successfully.")
    return 0

if __name__ == "__main__":
    main() 