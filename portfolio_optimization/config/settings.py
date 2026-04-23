"""Application settings for portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional


TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class DataConfig:
    """Data source configuration."""

    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN"])
    benchmark_ticker: str = "^GSPC"
    start_date: str = (date.today() - timedelta(days=365 * 3)).isoformat()
    end_date: str = date.today().isoformat()
    fill_method: str = "ffill_bfill"


@dataclass(frozen=True)
class OptimizationConfig:
    """MPT optimizer configuration."""

    risk_free_rate: float = 0.04
    allow_short_selling: bool = False
    max_weight: Optional[float] = 0.40
    min_weight_if_shorting: float = -0.30
    transaction_cost_penalty: float = 0.0
    risk_aversion: float = 2.5
    tau: float = 0.05


@dataclass(frozen=True)
class SimulationConfig:
    """Efficient frontier simulation configuration."""

    n_portfolios: int = 10_000
    random_seed: int = 42


@dataclass(frozen=True)
class SectorConstraintConfig:
    """Optional sector-level constraints."""

    asset_to_sector: Dict[str, str] = field(default_factory=dict)
    sector_max_weights: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorConfig:
    """Optional factor expected return adjustment placeholder."""

    enabled: bool = False
    factor_signal_map: Dict[str, float] = field(default_factory=dict)
    adjustment_scale: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    """Root app configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sector: SectorConstraintConfig = field(default_factory=SectorConstraintConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)


DEFAULT_CONFIG = AppConfig()
