"""Efficient frontier simulation and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FrontierResult:
    """Stores Monte Carlo frontier simulation output."""

    weights: np.ndarray
    returns: np.ndarray
    volatilities: np.ndarray
    sharpes: np.ndarray
    best_index: int
    asset_names: list[str]

    @property
    def best_weights(self) -> pd.Series:
        """Return best Sharpe weights as Series."""
        return pd.Series(self.weights[self.best_index], index=self.asset_names)


def simulate_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
    n_portfolios: int = 10_000,
    random_seed: Optional[int] = 42,
) -> FrontierResult:
    """Simulate random portfolios for efficient frontier."""
    rng = np.random.default_rng(random_seed)
    n_assets = len(expected_returns)
    raw_weights = rng.random((n_portfolios, n_assets))
    weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    mean_vec = expected_returns.values
    cov = covariance_matrix.values
    returns = weights @ mean_vec
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpes = np.where(np.isclose(volatilities, 0.0), 0.0, (returns - risk_free_rate) / volatilities)
    best_index = int(np.argmax(sharpes))

    return FrontierResult(
        weights=weights,
        returns=returns,
        volatilities=volatilities,
        sharpes=sharpes,
        best_index=best_index,
        asset_names=list(expected_returns.index),
    )


def plot_efficient_frontier(frontier: FrontierResult) -> plt.Figure:
    """Create risk-return frontier plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        frontier.volatilities,
        frontier.returns,
        c=frontier.sharpes,
        cmap="viridis",
        s=8,
        alpha=0.7,
    )
    ax.scatter(
        frontier.volatilities[frontier.best_index],
        frontier.returns[frontier.best_index],
        marker="*",
        color="red",
        s=300,
        label="Max Sharpe",
    )
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Expected Return (Annualized)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Sharpe Ratio")
    fig.tight_layout()
    return fig
