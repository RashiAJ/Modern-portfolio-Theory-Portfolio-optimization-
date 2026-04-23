"""Performance metrics for portfolios and strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR
from core.risk import drawdown_series


@dataclass
class PerformanceMetrics:
    """Summary metrics for return streams."""

    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def cumulative_returns(return_series: pd.Series) -> pd.Series:
    """Compute cumulative return index from periodic returns."""
    return (1.0 + return_series).cumprod()


def sharpe_ratio(return_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    excess_daily = return_series - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    volatility = excess_daily.std(ddof=1)
    if np.isclose(volatility, 0.0):
        return 0.0
    return float((excess_daily.mean() / volatility) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(return_series: pd.Series) -> float:
    """Compute maximum drawdown from periodic returns."""
    curve = cumulative_returns(return_series)
    dd = drawdown_series(curve)
    return float(dd.min())


def summarize_performance(return_series: pd.Series, risk_free_rate: float = 0.0) -> PerformanceMetrics:
    """Generate standard performance summary."""
    total_curve = cumulative_returns(return_series)
    ann_return = float(return_series.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(return_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return PerformanceMetrics(
        cumulative_return=float(total_curve.iloc[-1] - 1.0),
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe_ratio(return_series, risk_free_rate),
        max_drawdown=max_drawdown(return_series),
    )
