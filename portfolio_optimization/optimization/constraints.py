"""Constraint builder utilities for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ConstraintSpec:
    """Specification for optimizer constraints and bounds."""

    bounds: List[tuple[float, float]]
    constraints: List[dict]


def build_weight_bounds(
    n_assets: int,
    allow_short_selling: bool = False,
    max_weight: Optional[float] = None,
    min_weight_if_shorting: float = -0.30,
) -> List[tuple[float, float]]:
    """Build per-asset weight bounds."""
    if allow_short_selling:
        upper = max_weight if max_weight is not None else 1.0
        return [(min_weight_if_shorting, upper) for _ in range(n_assets)]
    upper = max_weight if max_weight is not None else 1.0
    return [(0.0, upper) for _ in range(n_assets)]


def build_full_investment_constraint() -> dict:
    """Enforce sum of weights equals one."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def build_sector_constraints(
    asset_names: list[str],
    asset_to_sector: Dict[str, str],
    sector_max_weights: Dict[str, float],
) -> List[dict]:
    """Create sector max allocation constraints if mapping is provided."""
    constraints: List[dict] = []
    if not asset_to_sector or not sector_max_weights:
        return constraints
    sector_indices: Dict[str, list[int]] = {}
    for idx, asset in enumerate(asset_names):
        sector = asset_to_sector.get(asset)
        if sector is not None:
            sector_indices.setdefault(sector, []).append(idx)

    for sector, max_alloc in sector_max_weights.items():
        indices = sector_indices.get(sector, [])
        if not indices:
            continue
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idx=indices, cap=max_alloc: cap - float(np.sum(w[idx])),
            }
        )
    return constraints


def transaction_cost_penalty(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_coefficient: float,
) -> float:
    """Placeholder linear transaction cost penalty."""
    turnover = np.abs(target_weights - current_weights).sum()
    return float(cost_coefficient * turnover)


def build_constraint_spec(
    asset_names: list[str],
    allow_short_selling: bool = False,
    max_weight: Optional[float] = None,
    min_weight_if_shorting: float = -0.30,
    asset_to_sector: Optional[Dict[str, str]] = None,
    sector_max_weights: Optional[Dict[str, float]] = None,
) -> ConstraintSpec:
    """Build full optimizer constraints and bounds."""
    constraints = [build_full_investment_constraint()]
    constraints.extend(
        build_sector_constraints(
            asset_names=asset_names,
            asset_to_sector=asset_to_sector or {},
            sector_max_weights=sector_max_weights or {},
        )
    )
    bounds = build_weight_bounds(
        n_assets=len(asset_names),
        allow_short_selling=allow_short_selling,
        max_weight=max_weight,
        min_weight_if_shorting=min_weight_if_shorting,
    )
    return ConstraintSpec(bounds=bounds, constraints=constraints)
