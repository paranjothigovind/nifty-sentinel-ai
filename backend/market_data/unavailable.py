"""
Unavailable Options Provider for NIFTY Sentinel.
Returns explicit structured unavailability; NEVER synthesizes, mocks, or guesses option quotes.
Used as the default options provider when no verified real-time options feed is configured.
"""

from datetime import datetime, date, timezone
from typing import Sequence, Optional, Dict, Any

from backend.market_data.base import MarketDataProvider
from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    ProviderHealth,
    DataMode,
)

class UnavailableOptionsProvider(MarketDataProvider):
    """
    Explicit fallback provider representing absence of options market data.
    Strictly forbids fabricating or simulating options bids, asks, or LTPs.
    """
    provider_id = "unavailable_options"
    display_name = "Unavailable Options Provider"

    def __init__(self, reason: str = "No verified options feed configured. Public APIs do not provide free real-time NSE derivatives."):
        self.reason = reason

    def get_index_candles(
        self,
        symbol: str = "NIFTY",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Sequence[IndexCandle]:
        return []

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        # Explicit empty sequence; NEVER synthesize options quotes
        return []

    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        # Explicit None; NEVER fabricate an option quote
        return None

    def get_unavailability_report(self) -> Dict[str, Any]:
        """
        Return structured description of why options data cannot be served.
        """
        return {
            "status": "UNAVAILABLE",
            "provider_id": self.provider_id,
            "reason": self.reason,
            "options_supported": False,
            "synthetic_fallback_permitted": False,
            "guidance": (
                "Real options paper trading requires a configured broker feed (e.g., Zerodha Kite, Upstox, AngelOne) "
                "or deterministic historical CSV replay. Real-time options are never fabricated."
            )
        }

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            display_name=self.display_name,
            is_healthy=False,
            data_mode=DataMode.UNAVAILABLE,
            capabilities=[],
            last_heartbeat_utc=datetime.now(timezone.utc),
            details=self.get_unavailability_report()
        )
