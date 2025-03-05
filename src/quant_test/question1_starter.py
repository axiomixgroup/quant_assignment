#!/usr/bin/env python3
"""
Question 1: Probability and Statistics

This script provides a starting point for analyzing the statistical properties of market data
and calculating relevant probabilities.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Callable, Any
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Import data utilities
from quant_test.data_utils import (
    load_reshaped_data,
    calculate_mid_prices,
    calculate_book_imbalance,
    calculate_spreads,
    get_sample_data
)

def main():
    """Run the probability and statistics analysis on sample data."""
    print("Running Question 1: Probability and Statistics")
    
    # Load sample data
    df = get_sample_data()
    instrument_id = 'BTC-USD'
    
    # Calculate mid prices, book imbalance, and spreads
    df = calculate_mid_prices(df, instrument_id)
    df = calculate_book_imbalance(df, instrument_id)
    df = calculate_spreads(df, instrument_id)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot mid price
    plt.figure(figsize=(12, 6))
    plt.plot(df[f'{instrument_id}_mid_price'])
    plt.title(f'{instrument_id} Mid Price')
    plt.xlabel('Time (ticks)')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.savefig(f'plots/q1_{instrument_id}_mid_price.png')
    
    # Plot book imbalance
    plt.figure(figsize=(12, 6))
    plt.plot(df[f'{instrument_id}_book_imbalance'])
    plt.title(f'{instrument_id} Book Imbalance')
    plt.xlabel('Time (ticks)')
    plt.ylabel('Imbalance')
    plt.tight_layout()
    plt.savefig(f'plots/q1_{instrument_id}_book_imbalance.png')
    
    # Plot spread
    plt.figure(figsize=(12, 6))
    plt.plot(df[f'{instrument_id}_spread'])
    plt.title(f'{instrument_id} Spread')
    plt.xlabel('Time (ticks)')
    plt.ylabel('Spread')
    plt.tight_layout()
    plt.savefig(f'plots/q1_{instrument_id}_spread.png')
    
    # Calculate returns
    df = df.with_columns([
        ((pl.col(f'{instrument_id}_mid_price') / pl.col(f'{instrument_id}_mid_price').shift(1)) - 1).alias('returns')
    ])
    
    # Drop NaN values
    df = df.drop_nulls(subset=['returns'])
    
    # Plot returns distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['returns'], kde=True)
    plt.title(f'{instrument_id} Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'plots/q1_{instrument_id}_returns_dist.png')
    
    # Calculate basic statistics
    mean_return = df['returns'].mean()
    std_return = df['returns'].std()
    skew = stats.skew(df['returns'].to_numpy())
    kurtosis = stats.kurtosis(df['returns'].to_numpy())
    
    print(f"Mean Return: {mean_return:.8f}")
    print(f"Standard Deviation: {std_return:.8f}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    
    # Test for normality
    shapiro_test = stats.shapiro(df['returns'].to_numpy())
    print(f"Shapiro-Wilk Test p-value: {shapiro_test.pvalue:.8f}")
    
    # Calculate conditional probabilities
    
    # Probability of positive return
    prob_positive = (df['returns'] > 0).mean()
    print(f"Probability of positive return: {prob_positive:.4f}")
    
    # Probability of positive return given positive book imbalance
    positive_imbalance = df.filter(pl.col(f'{instrument_id}_book_imbalance') > 0)
    prob_positive_given_positive_imbalance = (positive_imbalance['returns'] > 0).mean()
    print(f"Probability of positive return given positive book imbalance: {prob_positive_given_positive_imbalance:.4f}")
    
    # Probability of positive return given negative book imbalance
    negative_imbalance = df.filter(pl.col(f'{instrument_id}_book_imbalance') < 0)
    prob_positive_given_negative_imbalance = (negative_imbalance['returns'] > 0).mean()
    print(f"Probability of positive return given negative book imbalance: {prob_positive_given_negative_imbalance:.4f}")
    
    # Probability of positive return given wide spread
    median_spread = df[f'{instrument_id}_spread'].median()
    wide_spread = df.filter(pl.col(f'{instrument_id}_spread') > median_spread)
    prob_positive_given_wide_spread = (wide_spread['returns'] > 0).mean()
    print(f"Probability of positive return given wide spread: {prob_positive_given_wide_spread:.4f}")
    
    # Probability of positive return given narrow spread
    narrow_spread = df.filter(pl.col(f'{instrument_id}_spread') <= median_spread)
    prob_positive_given_narrow_spread = (narrow_spread['returns'] > 0).mean()
    print(f"Probability of positive return given narrow spread: {prob_positive_given_narrow_spread:.4f}")
    
    print("Question 1 completed successfully.")
    return 0

if __name__ == "__main__":
    main() 