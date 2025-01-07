import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

@dataclass
class PortfolioConfig:
    """Configuration settings for portfolio simulation"""
    start_date: str = "2019-01-01"  # Start date for historical data collection
    end_date: str = "2024-12-31"    # End date for historical data collection
    n_paths: int = 10000            # Number of simulation paths
    n_steps: int = 252              # Number of trading days to simulate (one year forward from today)
    risk_free_rate: float = 0.05
    confidence_level: float = 0.95

class PortfolioSimulator:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = self._setup_logger()
        np.random.seed(11)  # For reproducibility

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('PortfolioSimulator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_stock_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch and validate stock data"""
        try:
            data = yf.download(tickers, 
                             start=self.config.start_date, 
                             end=self.config.end_date)['Adj Close']            
            
            if data.empty:
                raise ValueError("No data retrieved. Please check ticker symbols.")
            
            # Handle missing or zero data
            data = data.replace(0, np.nan).dropna()
            
            self.logger.info(f"Data retrieved for period: {data.index.min()} to {data.index.max()}")
            return data
        
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}")
            raise

    def get_portfolio_weights(self, tickers: List[str], initial_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get user input for portfolio weights"""
        print("\nWould you like to:")
        print("1) Weight all stocks equally")
        print("2) Input weights by dollar amount")
        print("3) Input weights by number of shares")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        n_assets = len(tickers)
        
        if choice == "1":
            # Equal weights for all stocks
            weights = np.array([1/n_assets] * n_assets)
            initial_portfolio_value = np.dot(weights, initial_prices)
            self.logger.info("Using equal weights for all stocks")
            
        elif choice == "2":
            # User specified dollar amounts for each stock
            print("\nEnter the dollar amount invested in each stock:")
            amounts = []
            for ticker in tickers:
                amount = float(input(f"Amount for {ticker}: $"))
                amounts.append(amount)
            
            total_amount = sum(amounts)
            weights = np.array([amt / total_amount for amt in amounts])
            initial_portfolio_value = total_amount
            self.logger.info("Using weights based on dollar amounts")
            
        elif choice == "3":
            # User specified number of shares for eahc stock
            print("\nEnter the number of shares owned for each stock:")
            shares = []
            for ticker in tickers:
                share_count = float(input(f"Shares of {ticker}: "))
                shares.append(share_count)
            
            values = np.array(shares) * initial_prices
            total_value = np.sum(values)
            weights = values / total_value
            initial_portfolio_value = total_value
            self.logger.info("Using weights based on share counts")
            
        else:
            raise ValueError("Invalid choice. Please select 1, 2, or 3.")

        # Display the calculated weights
        print("\nPortfolio Weights:")
        for ticker, weight in zip(tickers, weights):
            print(f"{ticker}: {weight:.2%}")
        
        return weights, initial_portfolio_value

    def calculate_yearly_metrics(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        """Calculate average annual returns, volatilities, and correlation matrix"""
        # Split data by year to calculate annual metrics separately
        yearly_data = {year: data[data.index.year == year] 
                      for year in data.index.year.unique()}
        
        annual_metrics = {
            'mu': [],      # Store annulalized returns
            'sigma': [],   # Store annualized volatilities 
            'corr': []     # Store correlation matrices
        }

        for year_data in yearly_data.values():
            # Calulate daily log returns
            log_returns = np.log(year_data / year_data.shift(1))
            
            # Annualize returns by multiplying by trading days
            annual_metrics['mu'].append(log_returns.mean() * 252)
            # Annualize volatilities by multiplying by sqrt of trading days
            annual_metrics['sigma'].append(log_returns.std() * np.sqrt(252))
            annual_metrics['corr'].append(log_returns.corr())

        # Average the annual metrics across years
        mu_avg = pd.DataFrame(annual_metrics['mu']).mean()
        sigma_avg = pd.DataFrame(annual_metrics['sigma']).mean()
        corr_avg = sum(annual_metrics['corr']) / len(annual_metrics['corr'])
        
        # Add a small constant to the diagonal to ensure numerical stability
        corr_avg += np.eye(corr_avg.shape[0]) * 1e-10
        
        return mu_avg, sigma_avg, corr_avg

    def simulate_prices(self, initial_prices: np.ndarray, 
                        mu: pd.Series, sigma: pd.Series, 
                        corr: np.ndarray) -> np.ndarray:
        """Generate Monte Carlo simulation paths"""
        n_assets = len(initial_prices)
        # dt represents one trading day as a fraction of year
        # Used to scale annual rates to daily values
        dt = 1 / 252
        
        # Cholesky decomposition for correlated random walks
        # Decompose correlation matrix into L where corr = L * L^T
        L = np.linalg.cholesky(corr)

        # Generate standard normal random variables for each path
        Z = np.random.randn(self.config.n_steps, n_assets, self.config.n_paths)

        # Transform independent random variables into correlated ones using Cholesky
        correlated_Z = np.einsum('ij,tjk->tik', L, Z)
        
        # Initialize and simulate price paths using GBM
        S = np.zeros((self.config.n_steps, n_assets, self.config.n_paths))
        # Set initial prices for all paths
        S[0] = initial_prices[:, np.newaxis]
        
        # Simulate price paths using Geometric Brownian Motion
        for t in range(1, self.config.n_steps):
            S[t] = S[t-1] * np.exp(
                (mu.values - 0.5 * sigma.values**2)[:, np.newaxis] * dt +   # Drift term
                sigma.values[:, np.newaxis] * np.sqrt(dt) * correlated_Z[t] # Diffusion term
            )
        
        return S

    def calculate_portfolio_metrics(self, portfolio_values: np.ndarray, 
                                    portfolio_returns: np.ndarray, 
                                    expected_return: float) -> Dict[str, float]:
        """Calculate key portfolio risk metrics"""
        # Annualize volatility by scaling daily standar deviations
        volatility = np.std(portfolio_returns) * np.sqrt(252)

        # Calculate risk adjusted returns using Sharpe Ratio
        sharpe_ratio = (expected_return - self.config.risk_free_rate) / volatility

        # Calculate Value at Risk at specified confidence level
        var_95 = np.percentile(portfolio_returns[-1], 
                              (1 - self.config.confidence_level) * 100)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = portfolio_returns[-1][portfolio_returns[-1] <= var_95].mean()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'es_95': es_95
        }

    def plot_simulations(self, S: np.ndarray, portfolio_values: np.ndarray, 
                         tickers: List[str], initial_prices: np.ndarray,
                         initial_portfolio_value: float) -> None:
        """Generate visualization plots"""
        # Stock price paths (showing first 10 of n_paths simulations)
        plt.figure(figsize=(12, 6))
        for i, ticker in enumerate(tickers):
            plt.subplot(1, len(tickers), i + 1)
            # Plot 252 days (one year) of simulated prices
            plt.plot(S[:, i, :10])
            plt.title(f"{ticker} (${initial_prices[i]:.2f})")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Portfolio value paths
        portfolio_values_scaled = (portfolio_values * 
                                   initial_portfolio_value / portfolio_values[0, :])
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values_scaled[:, :10])
        plt.title("Portfolio Value Paths")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.show()

def main():
    # Initialize configuration
    config = PortfolioConfig()
    simulator = PortfolioSimulator(config)
    
    # Get user input for tickers
    print("Enter stock tickers separated by spaces (e.g., AAPL AMZN MSFT):")
    tickers = sorted(input().strip().split())
    
    # Fetch and process data
    data = simulator.get_stock_data(tickers)
    mu_avg, sigma_avg, corr_avg = simulator.calculate_yearly_metrics(data)
    # Get most recent prices as starting point for simulation
    initial_prices = data.iloc[-1].values
    
    # Get portfolio weights from user
    weights, initial_portfolio_value = simulator.get_portfolio_weights(tickers, initial_prices)
    
    # Run simulation
    S = simulator.simulate_prices(initial_prices, mu_avg, sigma_avg, corr_avg)
    # Calculate portfolio values across all paths using matrix multiplication
    portfolio_values = np.einsum('j,tjk->tk', weights, S)
    # Calculate daily returns for the portfolio
    portfolio_returns = np.diff(portfolio_values, axis=0) / portfolio_values[:-1]
    # Calculate expected portfolio returns using weighted average
    expected_return = np.dot(weights, mu_avg.values)
    
    # Calculate metrics
    metrics = simulator.calculate_portfolio_metrics(
        portfolio_values, portfolio_returns, expected_return
    )
    
    # Display results
    print(f"\nPortfolio Metrics:")
    print(f"Expected Return: {expected_return:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"VaR (95%): {metrics['var_95']:.2%}")
    print(f"Expected Shortfall (95%): {metrics['es_95']:.2%}")
    
    # Plot results
    simulator.plot_simulations(S, portfolio_values, tickers, 
                               initial_prices, initial_portfolio_value)

if __name__ == "__main__":
    main()