"""Utility helpers shared across modules."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize weights so they sum to one."""
    arr = np.asarray(weights, dtype=float)
    total = arr.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Weight sum is zero; cannot normalize.")
    return arr / total


def validate_tickers(tickers: Iterable[str]) -> list[str]:
    """Validate and clean ticker list."""
    cleaned = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ValueError("At least one valid ticker is required.")
    return cleaned


def annualize_returns(daily_return_series: pd.Series) -> float:
    """Convert average daily return to annualized return."""
    return float(daily_return_series.mean() * TRADING_DAYS_PER_YEAR)


def annualize_volatility(daily_return_series: pd.Series) -> float:
    """Convert daily volatility to annualized volatility."""
    return float(daily_return_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def dataclass_to_dict(instance: object) -> dict:
    """Serialize dataclass instance to dictionary."""
    if not is_dataclass(instance):
        raise TypeError("Expected dataclass instance.")
    return asdict(instance)
