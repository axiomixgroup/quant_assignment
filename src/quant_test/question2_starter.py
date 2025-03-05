#!/usr/bin/env python3
"""
Question 2: Stochastic Processes and Optimal Quoting

This script provides a starting point for implementing and analyzing a basic market making strategy
based on stochastic processes using real market data.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
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

@dataclass
class MarketMaker:
    """
    A simple market maker that quotes around the mid price with a fixed spread.
    """
    
    def __init__(
        self,
        spread_bps: float = 10.0,  # Spread in basis points
        inventory_limit: int = 100,
        position_decay: float = 0.1,  # How quickly to reduce inventory
        transaction_cost_bps: float = 1.0,  # Transaction cost in basis points
        quote_aggressiveness: float = 0.0  # How much to improve on market quotes (in bps)
    ):
        """
        Initialize the market maker with the given parameters.
        
        Args:
            spread_bps: Spread in basis points
            inventory_limit: Maximum inventory the market maker is willing to hold
            position_decay: How quickly to reduce inventory
            transaction_cost_bps: Transaction cost in basis points
            quote_aggressiveness: How much to improve on market quotes (in bps)
        """
        self.spread_bps = spread_bps
        self.inventory_limit = inventory_limit
        self.position_decay = position_decay
        self.transaction_cost_bps = transaction_cost_bps
        self.quote_aggressiveness = quote_aggressiveness
        
        # Initialize state
        self.position = 0
        self.cash = 0.0
        self.trades = []
        
    def get_quotes(self, mid_price: float, market_bid: float = None, market_ask: float = None) -> Tuple[float, float]:
        """
        Calculate the bid and ask quotes based on the mid price and current inventory.
        
        Args:
            mid_price: Current mid price
            market_bid: Current market bid (optional)
            market_ask: Current market ask (optional)
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate half spread in price terms
        half_spread_bps = self.spread_bps / 2
        half_spread = mid_price * (half_spread_bps / 10000)
        
        # Adjust spread based on inventory
        inventory_skew = self.position / self.inventory_limit if self.inventory_limit > 0 else 0
        inventory_adjustment = mid_price * (self.position_decay * inventory_skew / 10000)
        
        # Calculate base quotes
        bid_price = mid_price - half_spread - inventory_adjustment
        ask_price = mid_price + half_spread - inventory_adjustment
        
        # Improve on market quotes if specified
        if market_bid is not None and market_ask is not None and self.quote_aggressiveness > 0:
            improvement = mid_price * (self.quote_aggressiveness / 10000)
            bid_price = max(bid_price, market_bid + improvement)
            ask_price = min(ask_price, market_ask - improvement)
        
        return bid_price, ask_price
    
    def execute_trade(self, is_buy: bool, price: float, quantity: float, timestamp: float) -> None:
        """
        Execute a trade and update the market maker's state.
        
        Args:
            is_buy: True if the market maker is buying, False if selling
            price: Execution price
            quantity: Quantity traded
            timestamp: Time of the trade
        """
        # Update position and cash
        position_change = quantity if is_buy else -quantity
        self.position += position_change
        self.cash -= position_change * price
        
        # Apply transaction cost
        transaction_cost = price * quantity * (self.transaction_cost_bps / 10000)
        self.cash -= transaction_cost
        
        # Record the trade
        self.trades.append({
            'timestamp': timestamp,
            'is_buy': is_buy,
            'price': price,
            'quantity': quantity,
            'position': self.position,
            'cash': self.cash
        })
    
    def calculate_pnl(self, mid_price: float) -> float:
        """Calculate the current PnL based on the mid price."""
        return self.cash + self.position * mid_price

def run_backtest(
    df: pl.DataFrame,
    instrument_id: str,
    market_maker: MarketMaker,
    execution_prob: float = 0.2
) -> Dict:
    """
    Run a backtest of the market making strategy.
    
    Args:
        df: DataFrame with market data
        instrument_id: Instrument ID
        market_maker: MarketMaker instance
        execution_prob: Probability of execution for each quote
        
    Returns:
        Dictionary with backtest results
    """
    # Convert Polars DataFrame to Pandas DataFrame for easier iteration
    pdf = df.to_pandas()
    
    # Initialize results
    timestamps = []
    mid_prices = []
    positions = []
    cash_balances = []
    pnls = []
    bid_quotes = []
    ask_quotes = []
    market_bids = []
    market_asks = []
    
    # Get relevant columns
    timestamp_col = f'{instrument_id}_timestamp'
    bid_col = f'{instrument_id}_bid_price_1'
    ask_col = f'{instrument_id}_ask_price_1'
    
    # Run the backtest
    for i, row in pdf.iterrows():
        timestamp = row[timestamp_col]
        market_bid = row[bid_col]
        market_ask = row[ask_col]
        mid_price = (market_bid + market_ask) / 2
        
        # Get quotes
        bid_price, ask_price = market_maker.get_quotes(mid_price, market_bid, market_ask)
        
        # Check for executions
        if np.random.random() < execution_prob and bid_price > 0 and ask_price > 0:
            # Simulate market orders hitting our quotes
            if np.random.random() < 0.5:  # Someone sells to us (we buy)
                market_maker.execute_trade(True, bid_price, 1, timestamp)
            else:  # Someone buys from us (we sell)
                market_maker.execute_trade(False, ask_price, 1, timestamp)
        
        # Record state
        timestamps.append(timestamp)
        mid_prices.append(mid_price)
        positions.append(market_maker.position)
        cash_balances.append(market_maker.cash)
        pnls.append(market_maker.calculate_pnl(mid_price))
        bid_quotes.append(bid_price)
        ask_quotes.append(ask_price)
        market_bids.append(market_bid)
        market_asks.append(market_ask)
    
    # Create results dictionary
    results = {
        'timestamps': timestamps,
        'mid_prices': mid_prices,
        'positions': positions,
        'cash_balances': cash_balances,
        'pnls': pnls,
        'bid_quotes': bid_quotes,
        'ask_quotes': ask_quotes,
        'market_bids': market_bids,
        'market_asks': market_asks,
        'trades': market_maker.trades,
        'instrument_id': instrument_id
    }
    
    return results

def plot_results(results: Dict, instrument_id: str) -> None:
    """
    Plot the results of the backtest.
    
    Args:
        results: Dictionary with backtest results
        instrument_id: Instrument ID
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Convert timestamps to datetime for better plotting
    timestamps = pd.to_datetime(results['timestamps'], unit='ns')
    
    # Plot mid price and quotes
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, results['mid_prices'], label='Mid Price', color='black', alpha=0.5)
    plt.plot(timestamps, results['bid_quotes'], label='Bid Quote', color='green', alpha=0.7)
    plt.plot(timestamps, results['ask_quotes'], label='Ask Quote', color='red', alpha=0.7)
    plt.plot(timestamps, results['market_bids'], label='Market Bid', color='green', alpha=0.3, linestyle='--')
    plt.plot(timestamps, results['market_asks'], label='Market Ask', color='red', alpha=0.3, linestyle='--')
    plt.title(f'Mid Price and Quotes - {instrument_id}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q2_{instrument_id}_quotes.png')
    
    # Plot position
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, results['positions'], label='Position', color='blue')
    plt.title(f'Position - {instrument_id}')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q2_{instrument_id}_position.png')
    
    # Plot PnL
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, results['pnls'], label='PnL', color='green')
    plt.title(f'PnL - {instrument_id}')
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q2_{instrument_id}_pnl.png')
    
    # Plot cash balance
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, results['cash_balances'], label='Cash Balance', color='orange')
    plt.title(f'Cash Balance - {instrument_id}')
    plt.xlabel('Time')
    plt.ylabel('Cash')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/q2_{instrument_id}_cash.png')

def main():
    """Run the market making strategy on sample data."""
    print("Running Question 2: Stochastic Processes and Optimal Quoting")
    
    # Load sample data
    df = get_sample_data()
    instrument_id = 'BTC-USD'
    
    # Create a market maker
    market_maker = MarketMaker(
        spread_bps=10.0,
        inventory_limit=100,
        position_decay=0.1,
        transaction_cost_bps=1.0,
        quote_aggressiveness=2.0
    )
    
    # Run backtest
    results = run_backtest(
        df=df,
        instrument_id=instrument_id,
        market_maker=market_maker,
        execution_prob=0.2
    )
    
    # Plot results
    plot_results(results, instrument_id)
    
    # Print summary statistics
    final_pnl = results['pnls'][-1]
    max_position = max(abs(pos) for pos in results['positions'])
    num_trades = len(market_maker.trades)
    
    print(f"Final PnL: {final_pnl:.2f}")
    print(f"Max Position: {max_position}")
    print(f"Number of Trades: {num_trades}")
    
    print("Question 2 completed successfully.")
    return 0

if __name__ == "__main__":
    main() 