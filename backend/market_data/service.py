"""
Market Data Service Coordinator for NIFTY Sentinel.
Coordinates active providers, caching, freshness evaluation, and safety checks.
"""

from datetime import datetime, date, timezone
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

from backend.config import DATA_DIR, REPLAY_DATA_DIR
from backend.market_data.base import MarketDataProvider
from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    ProviderHealth,
    DataMode,
    DataQualityResult,
    PaperEntrySafetyCheck,
    to_ist_string,
    ensure_utc,
)
from backend.market_data.csv_replay import CsvReplayProvider
from backend.market_data.free_index import FreeIndexProvider
from backend.market_data.unavailable import UnavailableOptionsProvider
from backend.market_data.broker import BrokerOptionsProvider
from backend.market_data.validator import FreshnessAndIntegrityValidator
from backend.market_data.safety import PaperTradingSafetyEngine

class MarketDataService:
    """
    Central service facade for market data access, freshness evaluation, and paper safety.
    """
    def __init__(self):
        self.candles_sample_path = REPLAY_DATA_DIR / "nifty_5m_sample.csv"
        self.options_sample_path = REPLAY_DATA_DIR / "nifty_options_sample.csv"
        
        # Instantiate available providers
        self.replay_provider = CsvReplayProvider(
            candles_csv_path=self.candles_sample_path,
            options_csv_path=self.options_sample_path
        )
        self.free_index_provider = FreeIndexProvider(symbol="^NSEI")
        self.unavailable_options_provider = UnavailableOptionsProvider()
        self.broker_provider = BrokerOptionsProvider()

        # Default configuration: Replay mode if sample data exists, otherwise Free Index + Unavailable Options
        self.active_mode = "replay" if self.candles_sample_path.exists() else "live_unverified"
        
        # Freshness and Safety Engines
        self.validator = FreshnessAndIntegrityValidator()
        self.safety_engine = PaperTradingSafetyEngine(validator=self.validator)

    @property
    def current_index_provider(self) -> MarketDataProvider:
        if self.active_mode == "replay":
            return self.replay_provider
        return self.free_index_provider

    @property
    def current_options_provider(self) -> MarketDataProvider:
        if self.active_mode == "replay":
            return self.replay_provider
        return self.unavailable_options_provider

    def set_active_mode(self, mode: str) -> None:
        """
        Switch between 'replay' (deterministic CSV) and 'free_index' (public Yahoo + unavailable options).
        """
        valid_modes = {"replay", "free_index", "unavailable"}
        if mode in valid_modes:
            self.active_mode = mode

    def get_index_candles(
        self,
        symbol: str = "NIFTY",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50
    ) -> Sequence[IndexCandle]:
        candles = self.current_index_provider.get_index_candles(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end
        )
        if limit > 0 and len(candles) > limit:
            return candles[-limit:]
        return candles

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        return self.current_options_provider.get_option_chain(
            underlying=underlying,
            expiry=expiry,
            as_of=as_of
        )

    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        return self.current_options_provider.get_option_quote(
            contract=contract,
            as_of=as_of
        )

    def evaluate_paper_entry(
        self,
        directional_signal: str,
        target_contract: Optional[OptionContract],
        as_of_utc: Optional[datetime] = None
    ) -> PaperEntrySafetyCheck:
        # Fetch latest index candle
        candles = self.get_index_candles(limit=1)
        latest_candle = candles[-1] if candles else None

        # Fetch option quote if contract provided
        quote = None
        if target_contract is not None:
            quote = self.get_option_quote(target_contract, as_of=as_of_utc)

        return self.safety_engine.evaluate_paper_entry(
            directional_signal=directional_signal,
            target_contract=target_contract,
            index_candle=latest_candle,
            option_quote=quote,
            as_of_utc=as_of_utc
        )

    def get_all_providers_health(self) -> List[ProviderHealth]:
        return [
            self.replay_provider.health(),
            self.free_index_provider.health(),
            self.unavailable_options_provider.health(),
            self.broker_provider.health(),
        ]

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Compact status summary for the UI data-status panel.
        """
        candles = self.get_index_candles(limit=1)
        latest_candle = candles[-1] if candles else None
        
        index_health = self.current_index_provider.health()
        options_health = self.current_options_provider.health()

        chain = self.get_option_chain()
        latest_quote = chain[-1] if chain else None

        now = datetime.now(timezone.utc)
        index_age = (now - latest_candle.market_timestamp_utc).total_seconds() if latest_candle else None
        quote_age = (now - latest_quote.market_timestamp_utc).total_seconds() if latest_quote else None

        # Determine signal blocked reason
        market_open, session_status = FreshnessAndIntegrityValidator.is_market_open(now)
        blocked_reasons = []
        if not options_health.is_healthy:
            blocked_reasons.append("Options data unavailable (No verified broker feed)")
        if latest_candle is None:
            blocked_reasons.append("No index candles available")
        elif index_age is not None and index_age > 60.0 and self.active_mode != "replay":
            blocked_reasons.append(f"Index data stale ({index_age:.0f}s old > 60s limit)")

        return {
            "active_mode": self.active_mode,
            "data_mode": index_health.data_mode.value,
            "index_provider": {
                "id": index_health.provider_id,
                "name": index_health.display_name,
                "healthy": index_health.is_healthy,
                "mode": index_health.data_mode.value,
                "last_market_ts_utc": latest_candle.market_timestamp_utc.isoformat() if latest_candle else None,
                "last_market_ts_ist": to_ist_string(latest_candle.market_timestamp_utc) if latest_candle else None,
                "age_seconds": round(index_age, 1) if (index_age is not None and self.active_mode != "replay") else 0.0,
            },
            "options_provider": {
                "id": options_health.provider_id,
                "name": options_health.display_name,
                "healthy": options_health.is_healthy,
                "mode": options_health.data_mode.value,
                "contracts_count": len(chain),
                "last_market_ts_utc": latest_quote.market_timestamp_utc.isoformat() if latest_quote else None,
                "last_market_ts_ist": to_ist_string(latest_quote.market_timestamp_utc) if latest_quote else None,
                "age_seconds": round(quote_age, 1) if (quote_age is not None and self.active_mode != "replay") else 0.0,
            },
            "market_session": {
                "is_open": market_open,
                "session_note": session_status,
                "trading_hours": "09:15 - 15:30 IST (Mon-Fri)",
            },
            "safety_rules": {
                "completed_5m_bar_required": True,
                "max_index_delay_sec": 60,
                "max_option_quote_age_sec": 15,
                "live_trading_permitted": False,
            },
            "is_paper_trading_ready": len(blocked_reasons) == 0,
            "trade_signal_blocked_reasons": blocked_reasons,
        }

# Global singleton
market_data_service = MarketDataService()
