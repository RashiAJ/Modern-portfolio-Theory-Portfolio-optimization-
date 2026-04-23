# Portfolio Optimization with Modern Portfolio Theory

Production-style Python project for portfolio construction and analysis using MPT, efficient frontier simulation, Black-Litterman expected returns, and strategy backtesting.

## Features

- Historical market data ingestion from `yfinance`
- Daily/annualized return and risk computation
- MPT optimization via Sharpe ratio maximization with SLSQP
- Configurable constraints:
  - full-investment
  - long-only or short-enabled
  - per-asset max weight
  - sector allocation API structure
  - transaction-cost penalty placeholder
- Black-Litterman posterior expected returns
- Efficient frontier simulation (10,000+ random portfolios)
- Backtesting engine comparing:
  - optimized portfolio
  - equal-weight portfolio
  - benchmark
- Streamlit dashboard for interactive analysis

## Project Structure

```text
portfolio_optimization/
│
├── data/
│   └── data_loader.py
├── core/
│   ├── returns.py
│   ├── risk.py
│   └── metrics.py
├── optimization/
│   ├── mpt_optimizer.py
│   ├── black_litterman.py
│   └── constraints.py
├── simulation/
│   └── efficient_frontier.py
├── backtesting/
│   └── backtest.py
├── config/
│   └── settings.py
├── utils/
│   └── helpers.py
├── app/
│   └── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

## Theory Summary

### Modern Portfolio Theory (MPT)
MPT models portfolio construction as a risk-return tradeoff problem. For a given covariance structure, diversification allows improved risk-adjusted outcomes compared with naïve allocations.

### Efficient Frontier
The efficient frontier is the set of portfolios that maximize expected return for each risk level (or minimize risk for each return level). This project approximates the frontier using Monte Carlo random portfolios and identifies the max-Sharpe point.

### Sharpe Ratio
Sharpe ratio measures excess return over risk-free rate per unit of volatility:

`Sharpe = (E[R_p] - R_f) / sigma_p`

### Black-Litterman
Black-Litterman blends market-implied equilibrium returns with investor views (`P`, `Q`, `Omega`) to produce posterior expected returns that are usually more stable than plain historical means.


## Screenshot
<img width="1786" height="874" alt="image" src="https://github.com/user-attachments/assets/93abcf0a-8aad-453b-ac14-595e5b778689" />

