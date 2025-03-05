# Quantitative Researcher Assignment

## Overview

Welcome to our quantitative researcher assignment! This exercise is designed to assess your quantitative skills, critical thinking, and ability to communicate complex ideas clearly. As a crypto trading firm, we're looking for candidates who can bring fresh perspectives while demonstrating solid foundations in probability theory, stochastic processes, and machine learning.

## Important Notes

- **Time Expectation**: This assignment is designed to take approximately 3 hours to complete.
- **Tools**: Feel free to use any tools at your disposal, including LLMs (like ChatGPT or Claude), programming libraries, or reference materials.
- **Evaluation Criteria**: We're primarily interested in your thought process, reasoning, and ability to explain your work clearly. There are no strictly "right" or "wrong" answers as long as your approach is sound and well-justified.
- **Follow-up**: After submission, you'll have a 1-hour discussion with our head of quant to walk through your solutions and reasoning.

## Disclaimer

**Important**: The code in this assignment is not intended to be an example of how things are done in production environments. This is deliberately designed as a toy exercise to evaluate your skill set, problem-solving approach, and ability to reason about quantitative problems. Production trading systems would require significantly more robustness, optimization, and risk controls than what is expected in this assignment.

## Assignment Structure

This assignment consists of three questions, each testing different aspects of quantitative finance:

1. **Probability and Statistics**: Analyze market data distributions and calculate relevant probabilities.
2. **Stochastic Processes and Optimal Quoting**: Implement and analyze a basic market making strategy.
3. **Machine Learning and Feature Engineering**: Build a predictive model for price movements using market data.

## Data

All questions use the same market data, which can be loaded using the provided `data_utils.py` module. This data includes order book snapshots and trade information from crypto markets. The data is located in the `data/parsed_ondo` directory and includes:

- A pre-processed `reshaped_data.parquet` file that combines book and trade data
- Raw book and trade data files for various instruments

The `data_utils.py` module provides functions to:
- Load the reshaped data directly
- Calculate mid prices, order book imbalance, and spreads
- Sample the data for quick testing and exploration

**Note**: The data paths in the code are set to `../quant_test/data/parsed_ondo/`. If you're running this assignment in a different environment, you may need to update the paths in `data_utils.py` to point to the correct location of the data files.

## Submission Guidelines

Please organize your submission with the following structure:
- A separate script for each question
- Clear comments and explanations within your code
- A brief write-up (1-2 pages) summarizing your approach and findings for each question

## Questions

### Question 1: Probability and Statistics
Explore the statistical properties of the provided market data, focusing on the distribution of bid-ask spreads, order book imbalance, and trade sizes. Calculate conditional probabilities related to specific market events and visualize your findings.

### Question 2: Stochastic Processes and Optimal Quoting
Implement a basic market making strategy based on the Avellaneda-Stoikov framework or another approach of your choice. Analyze the strategy's performance under different market conditions and explain your reasoning. Consider factors such as latency, fill probability, fees, and tick-to-trade constraints.

### Question 3: Machine Learning and Feature Engineering
Using the provided market data, engineer features to predict price movements 50 trades or 3 seconds forward in time. Explain your feature selection, normalization approach, and model choice. The focus should be on demonstrating your understanding of the machine learning pipeline rather than achieving the highest possible accuracy.

## Getting Started

1. Review the detailed description for each question in the corresponding markdown files:
   - `question1_prob_stats.md`
   - `question2_stochastic_processes.md`
   - `question3_ml_prediction.md`

2. Use the starter code provided for each question:
   - `question1_starter.py`
   - `question2_starter.py`
   - `question3_starter.py`

3. Install the required dependencies listed in `requirements.txt`

4. You can run all three questions in sequence using the `run_all.py` script

## Tips for Completing the Assignment Efficiently

### Using LLM Code Generators

This assignment is designed to be completable within 3 hours, and we encourage the use of LLM code generators (like ChatGPT or Claude) to help you work efficiently. Here are some tips:

1. **Start with understanding the data**: Use the provided `data_utils.py` to explore the data structure before diving into complex implementations.

2. **Question-specific hints**:
   - **Question 1**: Focus on calculating basic statistics and visualizing distributions. LLMs can help generate boilerplate code for data analysis and visualization.
   - **Question 2**: For the market making strategy, provide the LLM with the basic Avellaneda-Stoikov framework and ask it to implement a simplified version. Focus on explaining the parameters and their impact.
   - **Question 3**: Break down the ML pipeline into steps (feature engineering, normalization, model training, evaluation). LLMs excel at generating code for standard ML workflows.

3. **Prompt engineering**: Be specific in your requests to LLMs. For example, instead of asking "How do I solve Question 2?", try "Help me implement a basic Avellaneda-Stoikov market making model with these specific parameters..."

4. **Focus on explanation**: Remember that your reasoning is more important than perfect code. Use LLMs to help document your code and explain your approach clearly.

5. **Iterate quickly**: Use LLMs to debug issues and refine your approach rather than spending too much time on any single problem.

Remember, the goal is to demonstrate your understanding of the concepts, not to build production-ready systems. It's perfectly acceptable to make simplifying assumptions and focus on the core aspects of each problem.

## Good Luck!

Remember, we're more interested in your thought process and ability to explain your work than in perfect solutions. Don't hesitate to make simplifying assumptions where appropriate, but be sure to state them clearly.

Happy coding! 