# Question 1: Probability and Statistics

## Overview

In this question, you'll explore the statistical properties of market data and calculate relevant probabilities. The focus is on understanding the distributions of key market metrics and their relationships.

## Tasks

1. **Data Exploration**:
   - Load the market data using the provided `data_utils.py` module
   - Calculate and visualize the distributions of:
     - Bid-ask spreads (both absolute and relative to mid price)
     - Order book imbalance
     - Trade sizes
   - Identify and discuss any notable patterns or anomalies

2. **Probability Analysis**:
   - Calculate the conditional probability of the mid price moving up by at least 1 basis point within the next 10 trades, given that:
     - The current order book imbalance is in the top quartile (highly positive)
     - The current order book imbalance is in the bottom quartile (highly negative)
   - Calculate the conditional probability of observing a trade with size greater than the 90th percentile, given that:
     - The bid-ask spread is wider than its median value
     - The bid-ask spread is narrower than its median value
   - Discuss the implications of these probabilities for market behavior

3. **Statistical Hypothesis Testing**:
   - Formulate and test a hypothesis about the relationship between order book imbalance and subsequent price movements
   - Use appropriate statistical tests to determine if the relationship is statistically significant
   - Discuss the limitations of your analysis and potential confounding factors

## Deliverables

1. A Jupyter notebook or Python script containing your analysis, with clear comments and explanations
2. Visualizations of the key distributions and relationships
3. A brief write-up (1-2 pages) summarizing your findings and their implications for trading strategies

## Evaluation Criteria

- Clarity of statistical reasoning and explanations
- Appropriate use of visualization techniques
- Correct calculation and interpretation of probabilities
- Critical thinking about the limitations and implications of the analysis

## Tips

- Consider using kernel density estimation for visualizing distributions
- Be mindful of potential biases in the data
- Think about how these statistical properties might vary across different market conditions
- Consider how your findings might inform trading decisions 