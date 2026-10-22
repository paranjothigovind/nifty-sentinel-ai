"""
Broker Options Provider Interface for NIFTY Sentinel.
Defines the architectural integration interface for licensed Indian brokers
(e.g., Zerodha Kite Connect, Upstox, AngelOne, Fyers).
Strictly kept DISABLED until a provider is deliberately configured and data access is verified.
Never exposes credentials in logs or committed code.
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

class BrokerOptionsProvider(MarketDataProvider):
    """
    Interface specification for authenticated broker market data adapters.
    Remains disabled by default to prevent accidental executions or unauthorized API calls.
    """
    provider_id = "broker_options_interface"
    display_name = "Broker Options Feed (Disabled / Integration Interface)"

    def __init__(
        self,
        broker_name: str = "Unconfigured Broker",
        is_configured: bool = False,
        is_verified: bool = False
    ):
        self.broker_name = broker_name
        self.is_configured = is_configured
        self.is_verified = is_verified

    def get_index_candles(
        self,
        symbol: str = "NIFTY",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Sequence[IndexCandle]:
        if not (self.is_configured and self.is_verified):
            return []
        raise NotImplementedError("Broker live feed requires active verified credentials.")

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        if not (self.is_configured and self.is_verified):
            return []
        raise NotImplementedError("Broker live feed requires active verified credentials.")

    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        if not (self.is_configured and self.is_verified):
            return None
        raise NotImplementedError("Broker live feed requires active verified credentials.")

    def health(self) -> ProviderHealth:
        status_msg = (
            "Broker options integration interface is intentionally DISABLED. "
            "Requires deliberate configuration, verified credentials, and exchange authorization."
            if not self.is_configured else "Broker configured but pending live session verification."
        )
        return ProviderHealth(
            provider_id=self.provider_id,
            display_name=f"{self.broker_name} Options Interface",
            is_healthy=False,
            data_mode=DataMode.UNAVAILABLE,
            capabilities=["option_chain", "option_quote"] if self.is_verified else [],
            last_heartbeat_utc=datetime.now(timezone.utc),
            details={
                "broker_name": self.broker_name,
                "is_configured": self.is_configured,
                "is_verified": self.is_verified,
                "status_reason": status_msg,
                "live_order_execution_supported": False,  # Hardcoded False
            }
        )
