"""Data loading utilities for market prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from utils.helpers import setup_logger, validate_tickers


logger = setup_logger(__name__)


@dataclass
class MarketData:
    """Container for cleaned market data."""

    prices: pd.DataFrame
    benchmark: Optional[pd.Series]


class DataLoader:
    """Fetches and cleans market price data."""

    def __init__(self, fill_method: str = "ffill_bfill") -> None:
        self.fill_method = fill_method

    def _clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Align and clean price data."""
        clean_prices = prices.sort_index().copy()
        clean_prices = clean_prices.dropna(axis=1, how="all")
        if self.fill_method == "ffill_bfill":
            clean_prices = clean_prices.ffill().bfill()
        else:
            clean_prices = clean_prices.dropna(axis=0, how="any")
        clean_prices = clean_prices.dropna(axis=0, how="any")
        if clean_prices.empty:
            raise ValueError("No clean price data available after missing value handling.")
        return clean_prices

    def load_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load adjusted close prices for tickers."""
        ticker_list = validate_tickers(tickers)
        logger.info("Downloading price data for %d tickers", len(ticker_list))

        raw = yf.download(
            ticker_list,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            raise ValueError("No data returned by yfinance.")

        if len(ticker_list) == 1:
            adj_close = raw.get("Adj Close")
            prices = pd.DataFrame({ticker_list[0]: adj_close})
        else:
            if ("Adj Close" not in raw.columns.get_level_values(0)) and ("Close" in raw.columns.get_level_values(0)):
                prices = raw["Close"].copy()
            elif "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"].copy()
            else:
                prices = raw.xs("Adj Close", axis=1, level=1, drop_level=False)

        prices = self._clean_prices(prices)
        missing_requested = sorted(set(ticker_list) - set(prices.columns))
        if missing_requested:
            logger.warning("Dropped missing tickers: %s", ", ".join(missing_requested))
        if prices.shape[1] == 0:
            raise ValueError("All requested tickers were dropped due to missing data.")
        return prices

    def load_benchmark(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Load benchmark adjusted close series."""
        if not ticker:
            return None
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty:
            logger.warning("Benchmark data not found for %s", ticker)
            return None
        series = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        return series.sort_index().ffill().bfill().dropna()

    def load_market_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        benchmark_ticker: Optional[str] = None,
    ) -> MarketData:
        """Load both asset and benchmark data."""
        prices = self.load_price_data(tickers, start_date, end_date)
        benchmark = self.load_benchmark(benchmark_ticker, start_date, end_date) if benchmark_ticker else None
        if benchmark is not None:
            common_idx = prices.index.intersection(benchmark.index)
            prices = prices.loc[common_idx]
            benchmark = benchmark.loc[common_idx]
        return MarketData(prices=prices, benchmark=benchmark)
