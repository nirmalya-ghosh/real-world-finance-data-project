# Real-World Finance Data Science Project

An advanced applied finance data science project using real historical market data. The project covers data ingestion, risk analytics, portfolio optimization, market regime discovery, anomaly detection, supervised prediction, strategy backtesting, stress testing, and a professional HTML report.

## Project Objective

Use real stock-market data to answer applied finance questions:

- Which assets delivered the best risk-adjusted performance?
- How correlated is the investment universe?
- What portfolio mix maximizes Sharpe ratio?
- Where is portfolio risk concentrated?
- How do portfolios behave under adverse scenarios?
- Can market features predict SPY direction over the next five trading days?
- Does a simple signal strategy outperform buy-and-hold in the holdout period?

## Data Source

The pipeline downloads daily OHLCV data from Yahoo Finance's public chart endpoint and caches raw ticker files locally.

Default universe:

- AAPL
- MSFT
- NVDA
- AMZN
- JPM
- UNH
- XOM
- SPY benchmark

## What Makes It The Most Advanced Project In This Set

- Real market data ingestion with local caching
- Data quality audit by ticker
- Daily return alignment across assets
- Annual return, volatility, Sharpe, Sortino, beta, alpha, VaR, CVaR, and drawdown metrics
- Return correlation matrix
- Monte Carlo efficient frontier
- Constrained portfolio optimization
- Max-Sharpe, min-volatility, and equal-weight portfolios
- Portfolio risk contribution analysis
- Scenario stress testing
- Rolling beta diagnostics
- PCA factor decomposition
- Market regime clustering
- Isolation Forest anomaly detection
- SPY next-5-trading-day direction prediction
- Time-series cross-validation
- Model leaderboard, ROC curve, confusion matrix, and feature importance
- Holdout strategy backtest vs SPY buy-and-hold
- Executive insight JSON and HTML report

## Repository Structure

```text
data/
  aligned_adjusted_close.csv
  aligned_volume.csv
  daily_returns.csv
  spy_prediction_features.csv
outputs/
  finance_report.html
  executive_insights.json
  asset_risk_metrics.csv
  optimized_portfolios.csv
  portfolio_risk_contribution.csv
  portfolio_stress_tests.csv
  rolling_beta.csv
  pca_factor_loadings.csv
  pca_factor_variance.csv
  market_regimes.csv
  market_anomaly_watchlist.csv
  model_leaderboard.csv
  strategy_backtest.csv
  *.png
models/
  spy_direction_champion.joblib
src/
  finance_project.py
requirements.txt
README.md
```

Some large reproducible artifacts are ignored in Git, including raw per-ticker cache files, the full efficient-frontier simulation dump, and model binaries. Running the script regenerates them locally.

## Run The Project

```powershell
python src\finance_project.py
```

Open:

```text
outputs\finance_report.html
```

## Analysis Pipeline

1. Download real historical market data.
2. Align adjusted close prices and volumes.
3. Compute daily returns and risk metrics.
4. Optimize portfolios using expected return and covariance.
5. Simulate thousands of random portfolios for the efficient frontier.
6. Measure portfolio risk contribution.
7. Stress-test optimized portfolios under named market scenarios.
8. Detect regimes from volatility, drawdown, returns, and breadth.
9. Detect anomalous market days.
10. Engineer predictive features for SPY next-5-day direction.
11. Train and evaluate time-aware supervised models.
12. Backtest a probability-threshold trading signal.
13. Generate a professional report.

## Important Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice, investment advice, or a recommendation to buy or sell any security.

## Portfolio Value

This project demonstrates the full applied data science loop in a real-world domain: ingestion, cleaning, analytics, optimization, machine learning, backtesting, reporting, and risk communication.
