"""CLI entrypoint for portfolio optimization pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtesting.backtest import plot_backtest_curves, run_backtest
from config.settings import DEFAULT_CONFIG
from core.returns import annualized_expected_returns, apply_factor_adjustment, compute_daily_returns
from core.risk import annualized_covariance_matrix, correlation_matrix
from data.data_loader import DataLoader
from optimization.black_litterman import BlackLittermanInputs, BlackLittermanModel
from optimization.mpt_optimizer import MPTOptimizer
from simulation.efficient_frontier import plot_efficient_frontier, simulate_efficient_frontier
from utils.helpers import setup_logger


logger = setup_logger(__name__)


def _build_black_litterman_expected_returns(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
) -> pd.Series:
    """Build optional Black-Litterman posterior from a neutral default view."""
    n_assets = len(expected_returns)
    market_weights = pd.Series(np.repeat(1.0 / n_assets, n_assets), index=expected_returns.index)
    p_matrix = np.eye(n_assets)
    q_vector = expected_returns.values
    bl_inputs = BlackLittermanInputs(
        covariance=covariance,
        market_weights=market_weights,
        risk_aversion=DEFAULT_CONFIG.optimization.risk_aversion,
        tau=DEFAULT_CONFIG.optimization.tau,
        p_matrix=p_matrix,
        q_vector=q_vector,
    )
    return BlackLittermanModel().posterior_returns(bl_inputs)


def run_pipeline() -> None:
    """Run complete portfolio optimization pipeline."""
    cfg = DEFAULT_CONFIG
    data_loader = DataLoader(fill_method=cfg.data.fill_method)
    market_data = data_loader.load_market_data(
        tickers=cfg.data.tickers,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        benchmark_ticker=cfg.data.benchmark_ticker,
    )
    prices = market_data.prices
    benchmark_prices = market_data.benchmark

    daily_returns = compute_daily_returns(prices)
    expected_returns = annualized_expected_returns(daily_returns)
    expected_returns = apply_factor_adjustment(
        expected_returns,
        factor_signal_map=cfg.factor.factor_signal_map,
        adjustment_scale=cfg.factor.adjustment_scale if cfg.factor.enabled else 0.0,
    )
    covariance = annualized_covariance_matrix(daily_returns)
    corr = correlation_matrix(daily_returns)

    bl_returns = _build_black_litterman_expected_returns(covariance, expected_returns)

    optimizer = MPTOptimizer(
        risk_free_rate=cfg.optimization.risk_free_rate,
        allow_short_selling=cfg.optimization.allow_short_selling,
        max_weight=cfg.optimization.max_weight,
        min_weight_if_shorting=cfg.optimization.min_weight_if_shorting,
        transaction_cost_penalty_coeff=cfg.optimization.transaction_cost_penalty,
    )
    result = optimizer.optimize(
        expected_returns=bl_returns,
        covariance_matrix=covariance,
        asset_to_sector=cfg.sector.asset_to_sector,
        sector_max_weights=cfg.sector.sector_max_weights,
    )

    frontier = simulate_efficient_frontier(
        expected_returns=bl_returns,
        covariance_matrix=covariance,
        risk_free_rate=cfg.optimization.risk_free_rate,
        n_portfolios=cfg.simulation.n_portfolios,
        random_seed=cfg.simulation.random_seed,
    )

    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().dropna()

    backtest = run_backtest(
        daily_returns=daily_returns,
        optimized_weights=result.weights,
        benchmark_returns=benchmark_returns,
        risk_free_rate=cfg.optimization.risk_free_rate,
    )

    logger.info("Optimization success: %s | %s", result.success, result.message)
    logger.info("Optimal weights:\n%s", result.weights.round(4).to_string())
    logger.info("Expected return: %.4f", result.expected_return)
    logger.info("Volatility: %.4f", result.volatility)
    logger.info("Sharpe: %.4f", result.sharpe_ratio)
    logger.info("Correlation matrix:\n%s", corr.round(3).to_string())

    metrics_view = {
        k: {
            "cumulative_return": v.cumulative_return,
            "annualized_return": v.annualized_return,
            "annualized_volatility": v.annualized_volatility,
            "sharpe_ratio": v.sharpe_ratio,
            "max_drawdown": v.max_drawdown,
        }
        for k, v in backtest.metrics.items()
    }
    logger.info("Backtest summary:\n%s", json.dumps(metrics_view, indent=2))

    frontier_fig = plot_efficient_frontier(frontier)
    backtest_fig = plot_backtest_curves(backtest)
    frontier_fig.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
    backtest_fig.savefig("backtest_comparison.png", dpi=150, bbox_inches="tight")
    logger.info("Saved charts: efficient_frontier.png, backtest_comparison.png")


if __name__ == "__main__":
    run_pipeline()
