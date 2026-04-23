"""Black-Litterman expected return model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BlackLittermanInputs:
    """Input matrices for Black-Litterman model."""

    covariance: pd.DataFrame
    market_weights: pd.Series
    risk_aversion: float
    tau: float
    p_matrix: np.ndarray
    q_vector: np.ndarray
    omega: Optional[np.ndarray] = None


class BlackLittermanModel:
    """Calculates posterior expected returns using investor views."""

    @staticmethod
    def implied_equilibrium_returns(
        covariance: pd.DataFrame,
        market_weights: pd.Series,
        risk_aversion: float,
    ) -> pd.Series:
        """Compute equilibrium returns (pi = delta * Sigma * w_mkt)."""
        pi = risk_aversion * covariance.values @ market_weights.values
        return pd.Series(pi, index=covariance.index)

    @staticmethod
    def _default_omega(tau_cov: np.ndarray, p_matrix: np.ndarray) -> np.ndarray:
        """Build diagonal omega from view uncertainty."""
        omega_vals = np.diag(p_matrix @ tau_cov @ p_matrix.T)
        return np.diag(omega_vals)

    def posterior_returns(self, inputs: BlackLittermanInputs) -> pd.Series:
        """Compute Black-Litterman posterior expected returns."""
        covariance = inputs.covariance
        p = np.asarray(inputs.p_matrix, dtype=float)
        q = np.asarray(inputs.q_vector, dtype=float).reshape(-1, 1)

        n_assets = covariance.shape[0]
        if p.shape[1] != n_assets:
            raise ValueError("P matrix columns must equal number of assets.")
        if p.shape[0] != q.shape[0]:
            raise ValueError("P matrix rows must equal Q vector length.")

        pi = self.implied_equilibrium_returns(covariance, inputs.market_weights, inputs.risk_aversion).values.reshape(-1, 1)
        sigma = covariance.values
        tau_sigma = inputs.tau * sigma
        omega = inputs.omega if inputs.omega is not None else self._default_omega(tau_sigma, p)

        middle = np.linalg.inv(p @ tau_sigma @ p.T + omega)
        posterior = pi + tau_sigma @ p.T @ middle @ (q - p @ pi)
        return pd.Series(posterior.flatten(), index=covariance.index)
