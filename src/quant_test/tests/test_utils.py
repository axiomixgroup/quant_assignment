"""Tests for the utils module."""

import unittest
import numpy as np
import pandas as pd
from quant_test.utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_drawdowns
)


class TestUtils(unittest.TestCase):
    """Test case for the utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        self.prices_np = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        self.prices_pd = pd.Series(self.prices_np)
        
        # Expected simple returns
        self.expected_simple_returns_np = np.array([0.02, -0.0098, 0.0198, 0.0194])
        self.expected_simple_returns_pd = pd.Series(
            [0.02, -0.0098, 0.0198, 0.0194], 
            index=self.prices_pd.index[1:]
        )
    
    def test_calculate_returns_simple_numpy(self):
        """Test calculating simple returns with numpy array."""
        returns = calculate_returns(self.prices_np, log_returns=False)
        np.testing.assert_almost_equal(returns, self.expected_simple_returns_np, decimal=4)
    
    def test_calculate_returns_simple_pandas(self):
        """Test calculating simple returns with pandas Series."""
        returns = calculate_returns(self.prices_pd, log_returns=False)
        pd.testing.assert_series_equal(
            returns.reset_index(drop=True), 
            self.expected_simple_returns_pd.reset_index(drop=True),
            check_dtype=False,
            atol=1e-4
        )
    
    def test_calculate_sharpe_ratio(self):
        """Test calculating Sharpe ratio."""
        # Create a series of constant returns for easy verification
        constant_returns = np.array([0.01] * 100)  # 1% daily return
        
        # With 0% risk-free rate, the Sharpe should be:
        # (0.01 * sqrt(252)) / 0 = infinity, but we'll get a divide by zero error
        # So we'll use a small standard deviation
        returns_with_noise = constant_returns + np.random.normal(0, 0.001, 100)
        
        sharpe = calculate_sharpe_ratio(returns_with_noise, risk_free_rate=0.0, periods_per_year=252)
        
        # The Sharpe should be approximately:
        # (0.01 * sqrt(252)) / small_std ≈ large positive number
        self.assertGreater(sharpe, 10)  # Should be much larger than 10
    
    def test_calculate_drawdowns(self):
        """Test calculating drawdowns."""
        # Create a series with a clear drawdown
        returns = np.array([0.01, 0.02, -0.05, -0.03, 0.04, 0.03])
        
        drawdowns, max_dd, max_dd_duration = calculate_drawdowns(returns)
        
        # The wealth index would be: [1.01, 1.0302, 0.97869, 0.94933, 0.98730, 1.01692]
        # The peak would be 1.0302 after the second return
        # The trough would be 0.94933 after the fourth return
        # The max drawdown should be approximately (0.94933 - 1.0302) / 1.0302 ≈ -0.0785
        
        self.assertAlmostEqual(max_dd, -0.0785, places=4)
        
        # The drawdown duration should be 4 (from peak to recovery)
        self.assertEqual(max_dd_duration, 4)


if __name__ == "__main__":
    unittest.main() 