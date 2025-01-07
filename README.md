# Monte Carlo Portfolio Simulation

## Overview

This Python based Monte Carlo Portfolio Simulation allows investors to analyze and understand the potential future behavior of their investment portfolios. By using historical data to calculate annualized returns and volatilities, it utilizes Monte Carlo techniques to generate thousands of possible scenarios for one year into the future from the current date. This is an interactive program, and users are provided with insights into their portfolio risk and return characteristics, including expected returns, volatility, Value at Risk, and other key risk metrics.

## Features

- Fetches real historical data using yfinance
- Supports multiple stocks in the portfolio
- Gives the user the option to either weight their stocks equally, or input custom weights
- Implements a Monte Carlo simulation using Geometric Brownian Motion
- Calculates key risk metrics:
  - Expected return
  - Portfolio volatility
  - Sharpe Ratio
  - Value at Risk (VaR)
  - Expected Shortfall (ES)
- Provides visual representation:
  - Individual simulated stock paths
  - Overall portfolio simulated stock paths

## Prerequisites

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
yfinance==0.1.87
```

**Note:** This model works best with yfinance version 0.1.87 since [AdjClose] has been deprecated since.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/isaiahnick/monte-carlo-portfolio.git
cd monte-carlo-portfolio
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python monteCarloSimulation.py
```

2. Enter stock tickers when prompted:
```
Enter stock tickers separated by spaces (e.g. AAPL AMZN MSFT):
```

3. Choose portfolio weighting method:
```
Would you like to:
1) Weight all stocks equally
2) Input weights by dollar amount
3) Input weights by number of shares
```

4. View the simulation results and visualizations.

## Example Output

The simulation will provide:
- Portfolio composition and weights
- Portfolio expected annual return
- Portfolio volatility
- Sharpe Ratio
- Value at Risk (95% confidence level)
- Expected Shortfall
- Visualizations of randomly chosen simulated price paths

## Configuration

You are able to modify these parameters in the PortfolioConfig class:
- `start_date`: Historical data start date
- `end_date`: Historical data end date
- `n_paths`: Number of simulated paths
- `n_steps`: Number of trading days to simulate
- `risk_free_rate`: Annual risk-free rate
- `confidence_level`: VaR confidence level

## Mathematical Background

The simulation employs several advanced mathematical techniques to model portfolio behavior:

### Geometric Brownian Motion (GBM)

The core of our simulation uses GBM to model stock price movements, defined by the stochastic differential equation:
```
dS = μSdt + σSdW
```
where:
- S is the stock price
- μ is the drift (expected return)
- σ is the volatility
- dW is a Wiener process

This model is particularly effective because it:
- Ensures stock prices remain positive
- Reflects the multiplicative nature of returns
- Captures both trend (drift) and uncertainty (diffusion) in price movements

### Linear Algebra and Correlations

The simulation handles multiple assets using advanced linear algebra techniques:
- Cholesky decomposition breaks down the correlation matrix to generate correlated random walks
- Matrix operations efficiently handle the simultaneous simulation of multiple stocks
- Eigenvalue decomposition ensures correlation matrices are positive semi-definite

### Statistical Methods

Several statistical approaches are employed:
- Log returns are used instead of simple returns to better approximate normality, giving more realistic results
- Monte Carlo methods generate thousands of possible paths to build reliable distributions
- Value at Risk (VaR) calculations use quantile analysis of simulated outcomes
- Expected Shortfall provides a more robust measure of tail risk than VaR alone

### Portfolio Theory Implementation

The model incorporates key concepts from Modern Portfolio Theory:
- Portfolio returns are weighted sums of individual asset returns
- Risk is measured through the variance-covariance matrix of returns
- The Sharpe Ratio measures risk-adjusted performance
- Correlation effects are fully accounted for in portfolio risk calculations

### Why These Methods Work

1. Linear Algebra Advantages:
   - Efficient handling of large datasets
   - Precise modeling of inter-asset relationships
   - Computationally optimal for large portfolios

2. Statistical Robustness:
   - Log returns provide better statistical properties
   - Monte Carlo simulation captures a wide range of possible outcomes
   - Multiple risk metrics offer comprehensive risk assessment

3. Numerical Stability:
   - Cholesky decomposition ensures stable correlation modeling
   - Matrix operations maintain numerical precision
   - Logarithmic transformations help manage scaling issues

## Limitations

- Assumes log-normal distribution of returns
- Uses historical data to predict future behavior
- Assumes constant volatility, meaning historically volatile stocks are not as accurately predicted
- Does not account for some realistic factors, such as:
  - Transaction costs
  - Taxes
  - Dividends
  - Market regime changes

## Future Improvements

- Add transaction costs and taxes
- Implement portfolio rebalancing using optimization techniques
- Add more risk metrics for analyzation
- Include dividend calculations
- Add stress testing scenarios

## Acknowledgments

- Data provided by Yahoo Finance through the yfinance package
- Mathematical concepts derived from Steven E. Shreve's Stochastic Calculus for Finance II
- Inspired by modern portfolio theory and risk management practices

## Contact

Isaiah Nick - isaiahdnick@gmail.com

Project Link: https://github.com/isaiahnick/monte-carlo-portfolio.git