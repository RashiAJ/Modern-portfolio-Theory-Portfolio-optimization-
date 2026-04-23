"""Streamlit dashboard for portfolio optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure sibling project modules are importable when launched via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.backtest import run_backtest
from core.returns import annualized_expected_returns, compute_daily_returns
from core.risk import annualized_covariance_matrix
from data.data_loader import DataLoader
from optimization.mpt_optimizer import MPTOptimizer
from simulation.efficient_frontier import plot_efficient_frontier, simulate_efficient_frontier


def _parse_tickers(text: str) -> list[str]:
    return [token.strip().upper() for token in text.split(",") if token.strip()]


def main() -> None:
    """Render streamlit application."""
    st.set_page_config(page_title="Portfolio Optimization", layout="wide")
    st.title("Portfolio Optimization with Modern Portfolio Theory")

    st.sidebar.header("Inputs")
    tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today().date())
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", "^GSPC")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual)", min_value=0.0, max_value=0.2, value=0.04, step=0.005)
    allow_short = st.sidebar.checkbox("Allow Short Selling", value=False)
    max_weight = st.sidebar.slider("Max Weight per Asset", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    run_button = st.sidebar.button("Run Optimization")

    if not run_button:
        st.info("Set parameters and click 'Run Optimization'.")
        return

    tickers = _parse_tickers(tickers_input)
    if not tickers:
        st.error("Please provide at least one valid ticker.")
        return

    loader = DataLoader()
    data = loader.load_market_data(
        tickers=tickers,
        start_date=str(start_date),
        end_date=str(end_date),
        benchmark_ticker=benchmark_ticker,
    )
    prices = data.prices
    daily_returns = compute_daily_returns(prices)
    expected_returns = annualized_expected_returns(daily_returns)
    covariance = annualized_covariance_matrix(daily_returns)

    optimizer = MPTOptimizer(
        risk_free_rate=risk_free_rate,
        allow_short_selling=allow_short,
        max_weight=max_weight,
    )
    opt_result = optimizer.optimize(expected_returns, covariance)
    frontier = simulate_efficient_frontier(
        expected_returns=expected_returns,
        covariance_matrix=covariance,
        risk_free_rate=risk_free_rate,
        n_portfolios=10_000,
        random_seed=42,
    )

    benchmark_returns = None
    if data.benchmark is not None:
        benchmark_returns = data.benchmark.pct_change().dropna()
    backtest = run_backtest(daily_returns, opt_result.weights, benchmark_returns, risk_free_rate)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Optimal Weights")
        st.dataframe(opt_result.weights.sort_values(ascending=False).to_frame("weight"))
        st.metric("Expected Return", f"{opt_result.expected_return:.2%}")
        st.metric("Volatility", f"{opt_result.volatility:.2%}")
        st.metric("Sharpe Ratio", f"{opt_result.sharpe_ratio:.3f}")

    with col2:
        st.subheader("Efficient Frontier")
        st.pyplot(plot_efficient_frontier(frontier))

    st.subheader("Backtest: Strategy Comparison")
    st.line_chart(backtest.cumulative_curves)

    metrics_df = pd.DataFrame(
        {
            name: {
                "Cumulative Return": metric.cumulative_return,
                "Annualized Return": metric.annualized_return,
                "Annualized Volatility": metric.annualized_volatility,
                "Sharpe Ratio": metric.sharpe_ratio,
                "Max Drawdown": metric.max_drawdown,
            }
            for name, metric in backtest.metrics.items()
        }
    ).T
    st.dataframe(metrics_df.style.format("{:.2%}", subset=["Cumulative Return", "Annualized Return", "Annualized Volatility", "Max Drawdown"]))
    st.dataframe(metrics_df.style.format("{:.3f}", subset=["Sharpe Ratio"]))

    st.subheader("Frontier Best Portfolio (Monte Carlo)")
    st.dataframe(frontier.best_weights.sort_values(ascending=False).to_frame("weight"))


if __name__ == "__main__":
    main()
