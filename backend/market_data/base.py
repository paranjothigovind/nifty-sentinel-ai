"""
Abstract Market Data Provider interface / protocol for NIFTY Sentinel.
Strategy and evaluation layers depend exclusively on this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Sequence, Optional
from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    ProviderHealth,
)

class MarketDataProvider(ABC):
    """
    Standard protocol for all intraday index and option data providers.
    """
    provider_id: str
    display_name: str

    @abstractmethod
    def get_index_candles(
        self,
        symbol: str = "NIFTY",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Sequence[IndexCandle]:
        """
        Fetch normalized index candles for the requested range.
        Must return sorted, non-overlapping bars.
        """
        pass

    @abstractmethod
    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        """
        Fetch the current option chain quotes for an underlying and expiry.
        Returns empty sequence or structured unavailability when options are unsupported.
        """
        pass

    @abstractmethod
    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        """
        Fetch a normalized quote for a specific option contract.
        Returns None when the contract quote is unavailable.
        """
        pass

    @abstractmethod
    def health(self) -> ProviderHealth:
        """
        Return the operational status, data mode, and capabilities of the provider.
        """
        pass
