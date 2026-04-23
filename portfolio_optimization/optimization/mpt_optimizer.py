"""Modern Portfolio Theory optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.risk import portfolio_volatility
from optimization.constraints import ConstraintSpec, build_constraint_spec, transaction_cost_penalty
from utils.helpers import normalize_weights, setup_logger


logger = setup_logger(__name__)


@dataclass
class OptimizationResult:
    """Optimization output container."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


class MPTOptimizer:
    """MPT optimizer for max Sharpe ratio."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        allow_short_selling: bool = False,
        max_weight: Optional[float] = None,
        min_weight_if_shorting: float = -0.30,
        transaction_cost_penalty_coeff: float = 0.0,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.allow_short_selling = allow_short_selling
        self.max_weight = max_weight
        self.min_weight_if_shorting = min_weight_if_shorting
        self.transaction_cost_penalty_coeff = transaction_cost_penalty_coeff

    @staticmethod
    def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
        """Compute portfolio expected return."""
        return float(np.dot(weights, expected_returns.values))

    @staticmethod
    def portfolio_volatility(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
        """Compute portfolio volatility."""
        return portfolio_volatility(weights, covariance_matrix)

    def _negative_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Negative Sharpe objective for minimizer."""
        p_return = self.portfolio_return(weights, expected_returns)
        p_vol = self.portfolio_volatility(weights, covariance_matrix)
        if np.isclose(p_vol, 0.0):
            return 1e6
        sharpe = (p_return - self.risk_free_rate) / p_vol
        penalty = 0.0
        if current_weights is not None and self.transaction_cost_penalty_coeff > 0:
            penalty = transaction_cost_penalty(current_weights, weights, self.transaction_cost_penalty_coeff)
        return -sharpe + penalty

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        asset_to_sector: Optional[dict[str, str]] = None,
        sector_max_weights: Optional[dict[str, float]] = None,
        current_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize weights via SLSQP for maximum Sharpe ratio."""
        asset_names = list(expected_returns.index)
        n_assets = len(asset_names)
        if n_assets == 0:
            raise ValueError("Expected returns are empty.")

        spec: ConstraintSpec = build_constraint_spec(
            asset_names=asset_names,
            allow_short_selling=self.allow_short_selling,
            max_weight=self.max_weight,
            min_weight_if_shorting=self.min_weight_if_shorting,
            asset_to_sector=asset_to_sector,
            sector_max_weights=sector_max_weights,
        )
        initial = np.repeat(1.0 / n_assets, n_assets)
        result = minimize(
            fun=self._negative_sharpe,
            x0=initial,
            args=(expected_returns, covariance_matrix, current_weights),
            method="SLSQP",
            bounds=spec.bounds,
            constraints=spec.constraints,
        )
        if not result.success:
            logger.warning("Optimization did not converge: %s", result.message)

        optimal_weights = normalize_weights(result.x)
        p_return = self.portfolio_return(optimal_weights, expected_returns)
        p_vol = self.portfolio_volatility(optimal_weights, covariance_matrix)
        sharpe = 0.0 if np.isclose(p_vol, 0.0) else (p_return - self.risk_free_rate) / p_vol

        return OptimizationResult(
            weights=pd.Series(optimal_weights, index=asset_names),
            expected_return=float(p_return),
            volatility=float(p_vol),
            sharpe_ratio=float(sharpe),
            success=bool(result.success),
            message=str(result.message),
        )
