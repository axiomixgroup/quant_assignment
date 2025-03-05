# Question 2: Stochastic Processes and Optimal Quoting

## Overview

In this question, you'll implement and analyze a basic market making strategy based on stochastic processes. You'll need to consider how to optimally place quotes in the order book, taking into account various market factors.

## Tasks

1. **Model Implementation**:
   - Implement a basic market making strategy based on the Avellaneda-Stoikov framework or another approach of your choice
   - Your implementation should include:
     - A stochastic model for price evolution
     - A method for determining optimal bid and ask quotes
     - A simple simulation environment to test the strategy

2. **Strategy Analysis**:
   - Analyze the performance of your strategy under different market conditions
   - Consider the following factors in your analysis:
     - Latency effects (how does performance degrade with increasing latency?)
     - Fill probability (how likely are your quotes to be executed?)
     - Transaction costs and fees
     - Tick size constraints (if applicable)
   - Compare your strategy to a simple benchmark (e.g., quoting at fixed spread)

3. **Extensions and Improvements**:
   - Propose at least one original modification or extension to the basic model
   - Implement your modification and analyze its impact on performance
   - Discuss the theoretical justification for your modification
   - Consider practical implementation challenges

## Deliverables

1. A Jupyter notebook or Python script containing your implementation and analysis
2. Visualizations of strategy performance under different conditions
3. A brief write-up (1-2 pages) explaining your approach, findings, and the reasoning behind your modifications

## Evaluation Criteria

- Understanding of stochastic processes and their application to market making
- Quality of implementation and analysis
- Originality and thoughtfulness of proposed modifications
- Consideration of practical constraints and trade-offs
- Clarity of explanation and reasoning

## Tips

- You can use simplified assumptions where appropriate, but be explicit about them
- Consider using the real market data to calibrate your model parameters
- Think about how your strategy would perform in extreme market conditions
- Balance theoretical elegance with practical considerations
- Focus on explaining your reasoning rather than achieving the highest possible performance 