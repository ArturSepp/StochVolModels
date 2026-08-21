"""Stable single-ticker access to optional Yahoo Finance research data."""

from typing import Optional

import pandas as pd
import yfinance as yf


def download_yfinance_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Download one full-history OHLC panel with stable column conventions."""
    download_kwargs = dict(
        tickers=ticker,
        auto_adjust=False,
        ignore_tz=True,
        multi_level_index=False,
        progress=progress,
    )
    if start is None and end is None:
        download_kwargs['period'] = 'max'
    else:
        download_kwargs.update(start=start, end=end)

    data = yf.download(**download_kwargs)
    if data is None or data.empty:
        raise ValueError(f'yfinance returned no history for ticker={ticker!r}')

    if isinstance(data.columns, pd.MultiIndex):
        matching_levels = [
            level
            for level in range(data.columns.nlevels)
            if ticker in data.columns.get_level_values(level)
        ]
        if matching_levels:
            data = data.xs(ticker, axis=1, level=matching_levels[0], drop_level=True)
        while isinstance(data.columns, pd.MultiIndex):
            singleton_levels = [
                level
                for level in range(data.columns.nlevels)
                if len(data.columns.get_level_values(level).unique()) == 1
            ]
            if not singleton_levels:
                break
            data.columns = data.columns.droplevel(singleton_levels[0])
        if isinstance(data.columns, pd.MultiIndex):
            raise ValueError(
                f'cannot normalize yfinance columns for ticker={ticker!r}: {data.columns}'
            )

    data.index = pd.to_datetime(data.index, utc=True)
    return data.sort_index()


def get_yfinance_close(data: pd.DataFrame, adjusted: bool = True) -> pd.Series:
    """Select adjusted close when requested, with close as a provider fallback."""
    candidates = ('Adj Close', 'Close') if adjusted else ('Close', 'Adj Close')
    column = next((candidate for candidate in candidates if candidate in data.columns), None)
    if column is None:
        raise ValueError(
            f'yfinance history contains neither Close nor Adj Close: {data.columns.tolist()}'
        )
    close = data[column]
    if not isinstance(close, pd.Series):
        raise ValueError(f'yfinance close selection is not one-dimensional: {close.columns}')
    return close
