"""
CSV Replay Provider for NIFTY Sentinel.
Provides deterministic timestamped historical replay of 5-minute index candles
and option chain quotes without requiring live network access.
"""

from datetime import datetime, date, timezone
from pathlib import Path
from typing import Sequence, Optional, List, Dict
import csv

from backend.market_data.base import MarketDataProvider
from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    OptionType,
    ProviderHealth,
    DataMode,
    QualityStatus,
    ensure_utc,
)

class CsvReplayProvider(MarketDataProvider):
    """
    Deterministic CSV replay provider for backtesting and paper trading simulation.
    """
    provider_id = "csv_replay"
    display_name = "Deterministic CSV Replay Provider"

    def __init__(
        self,
        candles_csv_path: Optional[Path] = None,
        options_csv_path: Optional[Path] = None
    ):
        self.candles_csv_path = candles_csv_path
        self.options_csv_path = options_csv_path
        self._cached_candles: List[IndexCandle] = []
        self._cached_quotes: List[OptionQuote] = []
        self._load_data()

    def _load_data(self) -> None:
        # Load Index Candles
        if self.candles_csv_path and self.candles_csv_path.exists():
            with open(self.candles_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
                candles: List[IndexCandle] = []
                for row in reader:
                    ts_str = row.get("timestamp_utc") or row.get("timestamp")
                    if not ts_str:
                        continue
                    ts_start = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_start = ensure_utc(ts_start)
                    # 5-minute bar completion
                    import datetime as dt_mod
                    ts_end = ts_start + dt_mod.timedelta(minutes=5)
                    
                    candle = IndexCandle(
                        symbol=row.get("symbol", "NIFTY"),
                        interval=row.get("interval", "5m"),
                        bar_start_utc=ts_start,
                        bar_end_utc=ts_end,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(float(row["volume"])) if row.get("volume") else None,
                        provider_id=self.provider_id,
                        market_timestamp_utc=ts_end,
                        receipt_timestamp_utc=ts_end,
                        data_mode=DataMode.REPLAY,
                        is_delayed=False,
                    )
                    candles.append(candle)
                # Sort chronologically and deduplicate
                candles.sort(key=lambda c: c.bar_start_utc)
                self._cached_candles = candles

        # Load Options Quotes
        if self.options_csv_path and self.options_csv_path.exists():
            with open(self.options_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
                quotes: List[OptionQuote] = []
                for row in reader:
                    ts_str = row.get("timestamp_utc") or row.get("timestamp")
                    if not ts_str:
                        continue
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts = ensure_utc(ts)
                    
                    contract = OptionContract(
                        underlying=row.get("underlying", "NIFTY"),
                        expiry=date.fromisoformat(row["expiry"]),
                        strike=float(row["strike"]),
                        option_type=OptionType(row["option_type"]),
                        instrument_id=row["instrument_id"]
                    )
                    
                    quote = OptionQuote(
                        contract=contract,
                        bid=float(row["bid"]) if row.get("bid") else None,
                        ask=float(row["ask"]) if row.get("ask") else None,
                        ltp=float(row["ltp"]) if row.get("ltp") else None,
                        volume=int(float(row["volume"])) if row.get("volume") else None,
                        open_interest=int(float(row["open_interest"])) if row.get("open_interest") else None,
                        iv=float(row["iv"]) if row.get("iv") else None,
                        delta=float(row["delta"]) if row.get("delta") else None,
                        gamma=float(row["gamma"]) if row.get("gamma") else None,
                        theta=float(row["theta"]) if row.get("theta") else None,
                        vega=float(row["vega"]) if row.get("vega") else None,
                        market_timestamp_utc=ts,
                        receipt_timestamp_utc=ts,
                        provider_id=self.provider_id,
                        data_mode=DataMode.REPLAY,
                        quality_status=QualityStatus.VALID,
                    )
                    quotes.append(quote)
                self._cached_quotes = quotes

    def get_index_candles(
        self,
        symbol: str = "NIFTY",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Sequence[IndexCandle]:
        candles = self._cached_candles
        if start is not None:
            start_utc = ensure_utc(start)
            candles = [c for c in candles if c.bar_start_utc >= start_utc]
        if end is not None:
            end_utc = ensure_utc(end)
            candles = [c for c in candles if c.bar_end_utc <= end_utc]
        return candles

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        quotes = self._cached_quotes
        if underlying:
            quotes = [q for q in quotes if q.contract.underlying.upper() == underlying.upper()]
        if expiry:
            quotes = [q for q in quotes if q.contract.expiry == expiry]
        if as_of:
            as_of_utc = ensure_utc(as_of)
            quotes = [q for q in quotes if q.market_timestamp_utc <= as_of_utc]
        return quotes

    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        matching = [
            q for q in self._cached_quotes
            if q.contract.instrument_id == contract.instrument_id
            or (
                q.contract.underlying == contract.underlying
                and q.contract.strike == contract.strike
                and q.contract.expiry == contract.expiry
                and q.contract.option_type == contract.option_type
            )
        ]
        if not matching:
            return None
        if as_of:
            as_of_utc = ensure_utc(as_of)
            matching = [q for q in matching if q.market_timestamp_utc <= as_of_utc]
            if not matching:
                return None
        return matching[-1]

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            display_name=self.display_name,
            is_healthy=True,
            data_mode=DataMode.REPLAY,
            capabilities=["index_candles", "option_chain", "option_quote"],
            last_heartbeat_utc=datetime.now(timezone.utc),
            details={
                "loaded_candles": len(self._cached_candles),
                "loaded_option_quotes": len(self._cached_quotes),
                "candles_file": str(self.candles_csv_path) if self.candles_csv_path else None,
                "options_file": str(self.options_csv_path) if self.options_csv_path else None,
            }
        )
