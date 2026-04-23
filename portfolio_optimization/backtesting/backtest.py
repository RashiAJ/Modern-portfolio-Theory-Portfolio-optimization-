"""Backtesting engine for portfolio strategies."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.metrics import PerformanceMetrics, cumulative_returns, summarize_performance


@dataclass
class BacktestResult:
    """Backtesting outputs for strategy comparison."""

    strategy_returns: pd.DataFrame
    cumulative_curves: pd.DataFrame
    metrics: dict[str, PerformanceMetrics]


def _portfolio_series(daily_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(daily_returns.columns).fillna(0.0).values
    returns = daily_returns.values @ aligned_weights
    return pd.Series(returns, index=daily_returns.index, name="portfolio")


def run_backtest(
    daily_returns: pd.DataFrame,
    optimized_weights: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Compare optimized, equal-weight, and benchmark strategies."""
    n_assets = daily_returns.shape[1]
    equal_weights = pd.Series(np.repeat(1.0 / n_assets, n_assets), index=daily_returns.columns)

    optimized = _portfolio_series(daily_returns, optimized_weights)
    equal = _portfolio_series(daily_returns, equal_weights)

    strategy_returns = pd.DataFrame(
        {
            "optimized": optimized,
            "equal_weight": equal,
        }
    )
    if benchmark_returns is not None:
        strategy_returns["benchmark"] = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)

    cumulative_curves = strategy_returns.apply(cumulative_returns, axis=0)
    metrics = {
        name: summarize_performance(series, risk_free_rate=risk_free_rate)
        for name, series in strategy_returns.items()
    }
    return BacktestResult(
        strategy_returns=strategy_returns,
        cumulative_curves=cumulative_curves,
        metrics=metrics,
    )


def plot_backtest_curves(result: BacktestResult) -> plt.Figure:
    """Plot cumulative performance comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in result.cumulative_curves.columns:
        ax.plot(result.cumulative_curves.index, result.cumulative_curves[col], label=col)
    ax.set_title("Backtest Performance Comparison")
    ax.set_ylabel("Cumulative Return Index")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    return fig
