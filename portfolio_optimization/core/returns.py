"""Return calculations."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute arithmetic daily returns from price data."""
    if prices.empty:
        raise ValueError("Price data is empty.")
    returns = prices.pct_change().dropna(how="all")
    return returns.dropna(axis=0, how="any")


def annualized_expected_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """Compute annualized expected return vector."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def apply_factor_adjustment(
    base_expected_returns: pd.Series,
    factor_signal_map: Optional[dict[str, float]] = None,
    adjustment_scale: float = 0.0,
) -> pd.Series:
    """Apply optional factor-based expected return adjustment."""
    if not factor_signal_map or adjustment_scale == 0.0:
        return base_expected_returns
    adjusted = base_expected_returns.copy()
    for asset, signal in factor_signal_map.items():
        if asset in adjusted.index:
            adjusted.loc[asset] += adjustment_scale * float(signal)
    return adjusted
