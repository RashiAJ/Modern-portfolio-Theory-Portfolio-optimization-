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

## Installation

```bash
pip install -r requirements.txt
```

## Run (CLI)

```bash
python main.py
```

Outputs include:
- optimization summary (weights, return, vol, Sharpe)
- backtest metric summary
- saved charts:
  - `efficient_frontier.png`
  - `backtest_comparison.png`

## Run (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Dashboard allows:
- ticker/date/risk-free inputs
- short-selling toggle
- max-weight control
- optimization and frontier visualization
- strategy comparison and performance metrics

## Example Usage

1. Set tickers: `AAPL,MSFT,GOOGL,AMZN`
2. Set range: `2021-01-01` to today
3. Set risk-free rate: `0.04`
4. Run optimization and inspect weights/frontier/backtest metrics

## Configuration

Defaults live in `config/settings.py`:
- data tickers and date windows
- risk-free rate and optimizer settings
- simulation count and random seed
- optional factor and sector constraint hooks

## Screenshot Placeholders

- `docs/screenshots/dashboard_overview.png` (placeholder)
- `docs/screenshots/frontier_plot.png` (placeholder)
- `docs/screenshots/backtest_comparison.png` (placeholder)

## Extension Ideas

- Replace static factor signal map with model-driven factor estimation
- Integrate rolling-window rebalancing and transaction costs in backtest loop
- Add downside-risk metrics (Sortino, CVaR) and robust optimization variants
