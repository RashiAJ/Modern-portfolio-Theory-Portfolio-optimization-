"""Risk computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


def annualized_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized covariance matrix."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute return correlation matrix."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.corr()


def portfolio_volatility(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    """Compute annualized portfolio volatility."""
    cov = covariance_matrix.values
    return float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))


def drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative return curve."""
    running_max = cumulative_returns.cummax()
    return (cumulative_returns / running_max) - 1.0
