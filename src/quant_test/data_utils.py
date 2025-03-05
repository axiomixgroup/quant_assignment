#!/usr/bin/env python3
"""
Data utilities for the quant test assignment.
This module provides functions for loading and processing market data.
"""

import os
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import re
from datetime import datetime

def ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9)

def process_files(
    data_dir: str = "data/parsed_ondo",
    use_exchange_time: bool = False,
    merge_precision: int = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process all files in the given directory and merge them into consolidated dataframes.
    
    Args:
        data_dir: Directory containing the data files
        use_exchange_time: Whether to use exchange time instead of receive time
        merge_precision: Precision for merging timestamps (in nanoseconds)
        
    Returns:
        Tuple of (order_book_df, trades_df)
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
    
    # Get all files in the directory
    files = os.listdir(data_dir)
    
    # Separate order book and trade files
    order_book_files = [f for f in files if 'order_book' in f]
    trade_files = [f for f in files if 'trades' in f]
    
    # Process order book files
    order_book_dfs = []
    for file in order_book_files:
        # Extract instrument ID from filename
        match = re.search(r'order_book_([A-Z0-9-]+)_', file)
        if not match:
            continue
        
        instrument_id = match.group(1)
        
        # Load the file
        df = pl.read_parquet(os.path.join(data_dir, file))
        
        # Rename columns to include instrument ID
        rename_map = {col: f'{instrument_id}_{col}' for col in df.columns}
        df = df.rename(rename_map)
        
        order_book_dfs.append(df)
    
    # Process trade files
    trade_dfs = []
    for file in trade_files:
        # Extract instrument ID from filename
        match = re.search(r'trades_([A-Z0-9-]+)_', file)
        if not match:
            continue
        
        instrument_id = match.group(1)
        
        # Load the file
        df = pl.read_parquet(os.path.join(data_dir, file))
        
        # Rename columns to include instrument ID
        rename_map = {col: f'{instrument_id}_{col}' for col in df.columns}
        df = df.rename(rename_map)
        
        trade_dfs.append(df)
    
    # Merge order book dataframes
    if order_book_dfs:
        order_book_df = pl.concat(order_book_dfs, how='diagonal')
    else:
        order_book_df = pl.DataFrame()
    
    # Merge trade dataframes
    if trade_dfs:
        trades_df = pl.concat(trade_dfs, how='diagonal')
    else:
        trades_df = pl.DataFrame()
    
    return order_book_df, trades_df

def load_reshaped_data(
    file_path: str = "data/parsed_ondo/reshaped_data.parquet"
) -> pl.DataFrame:
    """
    Load reshaped data from the given file path.
    
    Args:
        file_path: Path to the reshaped data file
        
    Returns:
        DataFrame with reshaped data
    """
    # Ensure file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reshaped data file not found at {file_path}")
    
    # Load the file
    df = pl.read_parquet(file_path)
    
    return df

def calculate_mid_prices(df: pl.DataFrame, instrument_id: str = None) -> pl.DataFrame:
    """
    Calculate mid prices from order book data.
    
    Args:
        df: DataFrame with order book data
        instrument_id: Instrument ID (if None, calculate for all instruments)
        
    Returns:
        DataFrame with mid prices
    """
    # Create a copy to avoid modifying the original
    result_df = df.clone()
    
    # Get all instrument IDs if not specified
    if instrument_id is None:
        # Extract instrument IDs from column names
        bid_cols = [col for col in df.columns if 'bid_price_1' in col]
        instrument_ids = [col.split('_')[0] for col in bid_cols]
    else:
        instrument_ids = [instrument_id]
    
    # Calculate mid prices for each instrument
    for instr_id in instrument_ids:
        bid_col = f'{instr_id}_bid_price_1'
        ask_col = f'{instr_id}_ask_price_1'
        
        if bid_col in df.columns and ask_col in df.columns:
            result_df = result_df.with_columns([
                ((pl.col(bid_col) + pl.col(ask_col)) / 2).alias(f'{instr_id}_mid_price')
            ])
    
    return result_df

def calculate_book_imbalance(df: pl.DataFrame, instrument_id: str = None) -> pl.DataFrame:
    """
    Calculate order book imbalance.
    
    Args:
        df: DataFrame with order book data
        instrument_id: Instrument ID (if None, calculate for all instruments)
        
    Returns:
        DataFrame with book imbalance
    """
    # Create a copy to avoid modifying the original
    result_df = df.clone()
    
    # Get all instrument IDs if not specified
    if instrument_id is None:
        # Extract instrument IDs from column names
        bid_cols = [col for col in df.columns if 'bid_price_1' in col]
        instrument_ids = [col.split('_')[0] for col in bid_cols]
    else:
        instrument_ids = [instrument_id]
    
    # Calculate book imbalance for each instrument
    for instr_id in instrument_ids:
        bid_size_col = f'{instr_id}_bid_size_1'
        ask_size_col = f'{instr_id}_ask_size_1'
        
        if bid_size_col in df.columns and ask_size_col in df.columns:
            result_df = result_df.with_columns([
                ((pl.col(bid_size_col) - pl.col(ask_size_col)) / (pl.col(bid_size_col) + pl.col(ask_size_col))).alias(f'{instr_id}_book_imbalance')
            ])
    
    return result_df

def calculate_spreads(df: pl.DataFrame, instrument_id: str = None) -> pl.DataFrame:
    """
    Calculate bid-ask spreads.
    
    Args:
        df: DataFrame with order book data
        instrument_id: Instrument ID (if None, calculate for all instruments)
        
    Returns:
        DataFrame with spreads
    """
    # Create a copy to avoid modifying the original
    result_df = df.clone()
    
    # Get all instrument IDs if not specified
    if instrument_id is None:
        # Extract instrument IDs from column names
        bid_cols = [col for col in df.columns if 'bid_price_1' in col]
        instrument_ids = [col.split('_')[0] for col in bid_cols]
    else:
        instrument_ids = [instrument_id]
    
    # Calculate spreads for each instrument
    for instr_id in instrument_ids:
        bid_col = f'{instr_id}_bid_price_1'
        ask_col = f'{instr_id}_ask_price_1'
        
        if bid_col in df.columns and ask_col in df.columns:
            result_df = result_df.with_columns([
                (pl.col(ask_col) - pl.col(bid_col)).alias(f'{instr_id}_spread')
            ])
    
    return result_df

def get_sample_data(n_rows: int = 10000, seed: int = 42) -> pl.DataFrame:
    """
    Get a random sample of reshaped data for testing.
    
    Args:
        n_rows: Number of rows to sample
        seed: Random seed
        
    Returns:
        DataFrame with sample data
    """
    # Create synthetic data for testing
    np.random.seed(seed)
    
    # Create timestamps
    timestamps = np.arange(n_rows) * 1_000_000_000  # 1 second intervals in nanoseconds
    
    # Create synthetic price data
    initial_price = 50000.0
    price_changes = np.random.normal(0, 100, n_rows)
    prices = initial_price + np.cumsum(price_changes)
    
    # Ensure prices are positive
    prices = np.maximum(prices, 1.0)
    
    # Create bid and ask prices with a small spread
    spreads = np.random.uniform(1, 50, n_rows)
    bid_prices = prices - spreads / 2
    ask_prices = prices + spreads / 2
    
    # Create bid and ask sizes
    bid_sizes = np.random.uniform(1, 10, n_rows)
    ask_sizes = np.random.uniform(1, 10, n_rows)
    
    # Create DataFrame
    instrument_id = 'BTC-USD'
    data = {
        f'{instrument_id}_timestamp': timestamps,
        f'{instrument_id}_bid_price_1': bid_prices,
        f'{instrument_id}_ask_price_1': ask_prices,
        f'{instrument_id}_bid_size_1': bid_sizes,
        f'{instrument_id}_ask_size_1': ask_sizes
    }
    
    df = pl.DataFrame(data)
    
    return df 