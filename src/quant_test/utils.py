"""Utility functions for the quant_test package."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


def calculate_returns(
    prices: Union[np.ndarray, pd.Series], 
    log_returns: bool = False
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate simple or logarithmic returns from a price series.
    
    Args:
        prices: Array or Series of prices
        log_returns: If True, calculate logarithmic returns, otherwise simple returns
        
    Returns:
        Array or Series of returns with the same type as input
    """
    if isinstance(prices, pd.Series):
        if log_returns:
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return (prices / prices.shift(1) - 1).dropna()
    else:
        if log_returns:
            return np.log(prices[1:] / prices[:-1])
        else:
            return prices[1:] / prices[:-1] - 1


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Array or Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year (252 for daily data)
        
    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)


def calculate_drawdowns(
    returns: Union[np.ndarray, pd.Series]
) -> Tuple[Union[np.ndarray, pd.Series], float, int]:
    """
    Calculate drawdowns and maximum drawdown.
    
    Args:
        returns: Array or Series of returns
        
    Returns:
        Tuple containing:
        - Drawdown series
        - Maximum drawdown value
        - Maximum drawdown duration
    """
    is_pandas = isinstance(returns, pd.Series)
    
    if is_pandas:
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        # Calculate max drawdown duration
        wealth_idx = wealth_index.values
        peak_idx = np.argmax(np.maximum.accumulate(wealth_idx) - wealth_idx)
        if peak_idx == 0:
            max_dd_duration = 0
        else:
            peak_value = wealth_idx[np.argmax(wealth_idx[:peak_idx])]
            recovery_idx = np.argmax(wealth_idx[peak_idx:] >= peak_value)
            if recovery_idx == 0:  # No recovery
                recovery_idx = len(wealth_idx) - peak_idx
            max_dd_duration = recovery_idx
    else:
        wealth_index = np.cumprod(1 + returns)
        previous_peaks = np.maximum.accumulate(wealth_index)
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        # Calculate max drawdown duration
        peak_idx = np.argmax(np.maximum.accumulate(wealth_index) - wealth_index)
        if peak_idx == 0:
            max_dd_duration = 0
        else:
            peak_value = wealth_index[np.argmax(wealth_index[:peak_idx])]
            recovery_idx = np.argmax(wealth_index[peak_idx:] >= peak_value)
            if recovery_idx == 0:  # No recovery
                recovery_idx = len(wealth_index) - peak_idx
            max_dd_duration = recovery_idx
    
    max_drawdown = np.min(drawdowns)
    
    return drawdowns, max_drawdown, max_dd_duration 