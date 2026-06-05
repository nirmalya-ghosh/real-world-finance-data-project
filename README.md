# Real-World Finance Data Science Project

This is an advanced applied data science project using real historical stock-market data.

The project downloads daily OHLCV data from Yahoo Finance's public chart endpoint, cleans and aligns the time series, performs portfolio and risk analytics, detects market regimes and anomalies, trains predictive models, backtests a signal strategy, and generates a structured HTML report.

## Domain

Finance: multi-asset equity analysis and prediction.

Default universe:

- AAPL
- MSFT
- NVDA
- AMZN
- JPM
- UNH
- XOM
- SPY benchmark

## Advanced Features

- Real data ingestion with local caching
- Data quality audit
- Return, volatility, drawdown, beta, alpha, Sharpe, Sortino, VaR, and CVaR metrics
- Correlation and diversification analysis
- Monte Carlo efficient frontier
- Max-Sharpe and min-volatility portfolio selection
- Rolling risk and drawdown diagnostics
- K-Means market regime detection
- Isolation Forest anomaly detection
- Next-day direction prediction using supervised learning
- ROC curve, confusion matrix, feature importance, and model leaderboard
- Walk-forward style holdout backtest against buy-and-hold
- HTML report with conclusions and exported CSV evidence

## Run

```powershell
python src\finance_project.py
```

Then open:

```text
outputs\finance_report.html
```

## Important Note

This project is for educational data science practice only. It is not financial advice.
