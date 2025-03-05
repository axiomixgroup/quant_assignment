# Question 3: Machine Learning and Feature Engineering

## Overview

In this question, you'll build a predictive model for price movements using market data. The focus is on feature engineering, normalization techniques, and model selection rather than achieving the highest possible accuracy.

## Tasks

1. **Target Definition**:
   - Define a clear prediction target: price change 50 trades OR 3 seconds forward in time
   - Decide what price to use (mid price, last trade price, etc.) and justify your choice
   - Determine how to frame the problem (regression, classification, etc.)

2. **Feature Engineering**:
   - Engineer at least 5 features from the market data that you believe are predictive
   - Features could include (but are not limited to):
     - Order book imbalance metrics
     - Recent price momentum
     - Volatility measures
     - Trade flow indicators
     - Time-based features
   - Explain the intuition behind each feature and why you believe it might be predictive

3. **Data Preprocessing**:
   - Implement appropriate normalization and standardization techniques for your features
   - Explain why you chose these techniques and how they address the specific characteristics of your features
   - Handle any missing values or outliers in a principled way
   - Implement a proper train/validation/test split that avoids look-ahead bias

4. **Model Selection and Evaluation**:
   - Choose a machine learning model suitable for your problem formulation
   - Explain why you selected this model and its advantages/disadvantages for this specific task
   - Implement the model and evaluate its performance using appropriate metrics
   - Analyze which features contribute most to the model's predictions

## Deliverables

1. A Jupyter notebook or Python script containing your implementation
2. Visualizations of feature distributions, correlations, and model performance
3. A brief write-up (1-2 pages) explaining your approach, feature engineering decisions, and model selection reasoning

## Evaluation Criteria

- Quality and creativity of feature engineering
- Understanding of appropriate normalization and standardization techniques
- Thoughtfulness of model selection and evaluation
- Clarity of explanation and reasoning
- Consideration of practical implementation issues

## Tips

- Focus on explaining your thought process rather than achieving the highest accuracy
- Consider the computational efficiency of your features and model
- Think about how your approach would scale to production environments
- Be mindful of potential biases in your evaluation methodology
- Consider how your features might behave in different market regimes 