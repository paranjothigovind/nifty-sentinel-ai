"""
Freshness, Quality, and Integrity Validation Engine for NIFTY Sentinel.
Enforces configurable thresholds:
  - Completed 5-minute bar requirements
  - 60-second index freshness rejection
  - 15-second options quote freshness rejection
  - Bid/ask spread sanity & non-positive price rejection
  - Clock skew, timestamp regression, duplicate bar rejection
  - NSE session boundary & market holiday awareness
Note: These thresholds are software quality safeguards and are NOT exchange guarantees.
"""

from datetime import datetime, time, timezone, timedelta
from typing import Sequence, Optional, List, Dict, Any, Tuple
from zoneinfo import ZoneInfo

from backend.market_data.schemas import (
    IndexCandle,
    OptionQuote,
    DataQualityResult,
    QualityStatus,
    ensure_utc,
    to_ist_string,
)
from backend.config import (
    KNOWN_INDIAN_HOLIDAYS,
    DEFAULT_MAX_INDEX_DELAY_SECONDS,
    DEFAULT_MAX_OPTION_QUOTE_AGE_SECONDS,
    DEFAULT_MAX_CLOCK_SKEW_SECONDS,
)

IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN_TIME = time(9, 15)   # 09:15 IST
NSE_CLOSE_TIME = time(15, 30) # 15:30 IST

class FreshnessAndIntegrityValidator:
    """
    Validates data freshness, sanity, and completeness for intraday decision pipelines.
    """
    def __init__(
        self,
        max_index_delay_seconds: float = DEFAULT_MAX_INDEX_DELAY_SECONDS,
        max_option_quote_age_seconds: float = DEFAULT_MAX_OPTION_QUOTE_AGE_SECONDS,
        max_clock_skew_seconds: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        require_completed_bar: bool = True
    ):
        self.max_index_delay_seconds = max_index_delay_seconds
        self.max_option_quote_age_seconds = max_option_quote_age_seconds
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.require_completed_bar = require_completed_bar

    @staticmethod
    def is_market_open(as_of_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if the specified timestamp falls within normal NSE market trading hours
        (09:15 to 15:30 IST, Monday to Friday, excluding statutory holidays).
        """
        if as_of_utc is None:
            as_of_utc = datetime.now(timezone.utc)
        as_of_utc = ensure_utc(as_of_utc)
        as_of_ist = as_of_utc.astimezone(IST)

        # Check weekend
        if as_of_ist.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False, f"Weekend ({as_of_ist.strftime('%A')}) - NSE closed"

        # Check holiday
        mm_dd = as_of_ist.strftime("%m-%d")
        if mm_dd in KNOWN_INDIAN_HOLIDAYS:
            holiday_name = KNOWN_INDIAN_HOLIDAYS[mm_dd]
            return False, f"NSE Market Holiday: {holiday_name}"

        # Check trading hours (09:15 to 15:30 IST)
        cur_time = as_of_ist.time()
        if cur_time < NSE_OPEN_TIME:
            return False, f"Pre-market session ({cur_time.strftime('%H:%M')} IST); market opens at 09:15 IST"
        if cur_time > NSE_CLOSE_TIME:
            return False, f"Post-market session ({cur_time.strftime('%H:%M')} IST); market closed at 15:30 IST"

        return True, "NSE Intraday market is open"

    def validate_index_candle(
        self,
        candle: Optional[IndexCandle],
        as_of_utc: Optional[datetime] = None,
        is_replay: bool = False
    ) -> DataQualityResult:
        """
        Validate freshness, completion, and invariants of an index candle.
        """
        if as_of_utc is None:
            as_of_utc = datetime.now(timezone.utc)
        as_of_utc = ensure_utc(as_of_utc)

        checks: Dict[str, Any] = {
            "completed_bar_check": False,
            "clock_skew_check": False,
            "freshness_check": False,
            "price_invariants_check": False,
            "volume_check": False,
        }
        missing_fields: List[str] = []

        if candle is None:
            return DataQualityResult(
                status=QualityStatus.UNAVAILABLE,
                reason="No index candle provided or feed is empty",
                provider_id="unknown",
                checks=checks
            )

        provider_id = candle.provider_id
        timestamp_utc = candle.market_timestamp_utc
        timestamp_ist = to_ist_string(timestamp_utc)

        # 1. Price Invariants Check
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason=f"Non-positive price in candle: O={candle.open}, H={candle.high}, L={candle.low}, C={candle.close}",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                checks=checks
            )
        if candle.high < candle.low or candle.high < candle.open or candle.high < candle.close:
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason=f"High price invalid relative to open/low/close (H={candle.high}, L={candle.low})",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                checks=checks
            )
        checks["price_invariants_check"] = True

        # 2. Completed Bar Check
        # A candle for 09:15 - 09:20 is only complete after 09:20:00
        if self.require_completed_bar and not is_replay:
            if candle.bar_end_utc > as_of_utc + timedelta(seconds=self.max_clock_skew_seconds):
                return DataQualityResult(
                    status=QualityStatus.INVALID,
                    reason=f"Incomplete bar: Candle bar_end ({to_ist_string(candle.bar_end_utc)}) is in the future relative to evaluation time ({to_ist_string(as_of_utc)})",
                    provider_id=provider_id,
                    timestamp_utc=timestamp_utc,
                    timestamp_ist=timestamp_ist,
                    checks=checks
                )
        checks["completed_bar_check"] = True

        # 3. Clock Skew Check
        clock_skew = (candle.market_timestamp_utc - as_of_utc).total_seconds()
        if not is_replay and clock_skew > self.max_clock_skew_seconds:
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason=f"Future timestamp beyond permitted clock skew ({clock_skew:.1f}s > {self.max_clock_skew_seconds}s)",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                checks=checks
            )
        checks["clock_skew_check"] = True

        # 4. Freshness Check (only enforced on LIVE/DELAYED, not on deterministic historical replay)
        age_seconds = max(0.0, (as_of_utc - candle.market_timestamp_utc).total_seconds())
        if not is_replay:
            # If feed is marked delayed, account for its expected delay
            allowed_delay = self.max_index_delay_seconds + (candle.delay_seconds or 0)
            if age_seconds > allowed_delay:
                return DataQualityResult(
                    status=QualityStatus.STALE,
                    reason=f"Index candle is stale: last update was {age_seconds:.1f}s ago (threshold: {allowed_delay}s)",
                    age_seconds=round(age_seconds, 1),
                    provider_id=provider_id,
                    timestamp_utc=timestamp_utc,
                    timestamp_ist=timestamp_ist,
                    checks=checks
                )
        checks["freshness_check"] = True

        # 5. Volume
        checks["volume_check"] = candle.volume is not None

        return DataQualityResult(
            status=QualityStatus.VALID,
            reason="Index candle passed all freshness and integrity checks",
            age_seconds=round(age_seconds, 1) if not is_replay else 0.0,
            provider_id=provider_id,
            timestamp_utc=timestamp_utc,
            timestamp_ist=timestamp_ist,
            checks=checks
        )

    def validate_option_quote(
        self,
        quote: Optional[OptionQuote],
        as_of_utc: Optional[datetime] = None,
        is_replay: bool = False
    ) -> DataQualityResult:
        """
        Validate freshness, bid/ask spread sanity, and completeness of an option quote.
        Never allows forward-filling across entries if quotes are stale.
        """
        if as_of_utc is None:
            as_of_utc = datetime.now(timezone.utc)
        as_of_utc = ensure_utc(as_of_utc)

        checks: Dict[str, Any] = {
            "contract_identity_check": False,
            "bid_ask_spread_check": False,
            "positive_prices_check": False,
            "freshness_check": False,
            "clock_skew_check": False,
        }
        missing_fields: List[str] = []

        if quote is None:
            return DataQualityResult(
                status=QualityStatus.UNAVAILABLE,
                reason="Option quote is unavailable. No provider quote returned for requested contract.",
                provider_id="unavailable",
                missing_fields=["contract", "bid", "ask", "ltp"],
                checks=checks
            )

        provider_id = quote.provider_id
        timestamp_utc = quote.market_timestamp_utc
        timestamp_ist = to_ist_string(timestamp_utc)

        # 1. Contract Identity Check
        if not quote.contract or not quote.contract.instrument_id or not quote.contract.underlying:
            missing_fields.append("contract_identity")
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason="Option quote missing valid contract identity specification",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                missing_fields=missing_fields,
                checks=checks
            )
        checks["contract_identity_check"] = True

        # 2. Missing Price Fields
        if quote.bid is None:
            missing_fields.append("bid")
        if quote.ask is None:
            missing_fields.append("ask")
        if quote.ltp is None:
            missing_fields.append("ltp")

        if missing_fields:
            return DataQualityResult(
                status=QualityStatus.DEGRADED if quote.ltp is not None else QualityStatus.INVALID,
                reason=f"Option quote missing critical price fields: {', '.join(missing_fields)}",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                missing_fields=missing_fields,
                checks=checks
            )

        # 3. Positive Price Check
        if (quote.bid is not None and quote.bid <= 0) or (quote.ask is not None and quote.ask <= 0) or (quote.ltp is not None and quote.ltp <= 0):
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason=f"Non-positive option price: bid={quote.bid}, ask={quote.ask}, ltp={quote.ltp}",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                checks=checks
            )
        checks["positive_prices_check"] = True

        # 4. Bid / Ask Spread Sanity Check
        if quote.bid is not None and quote.ask is not None:
            if quote.bid > quote.ask:
                return DataQualityResult(
                    status=QualityStatus.INVALID,
                    reason=f"Crossed book: Bid ({quote.bid}) > Ask ({quote.ask})",
                    provider_id=provider_id,
                    timestamp_utc=timestamp_utc,
                    timestamp_ist=timestamp_ist,
                    checks=checks
                )
        checks["bid_ask_spread_check"] = True

        # 5. Clock Skew Check
        clock_skew = (quote.market_timestamp_utc - as_of_utc).total_seconds()
        if not is_replay and clock_skew > self.max_clock_skew_seconds:
            return DataQualityResult(
                status=QualityStatus.INVALID,
                reason=f"Option quote timestamp is in the future beyond allowed clock skew ({clock_skew:.1f}s)",
                provider_id=provider_id,
                timestamp_utc=timestamp_utc,
                timestamp_ist=timestamp_ist,
                checks=checks
            )
        checks["clock_skew_check"] = True

        # 6. Freshness Check (reject if > 15s old for paper entry)
        age_seconds = max(0.0, (as_of_utc - quote.market_timestamp_utc).total_seconds())
        if not is_replay:
            if age_seconds > self.max_option_quote_age_seconds:
                return DataQualityResult(
                    status=QualityStatus.STALE,
                    reason=f"Option quote is stale: last update was {age_seconds:.1f}s ago (threshold: {self.max_option_quote_age_seconds}s)",
                    age_seconds=round(age_seconds, 1),
                    provider_id=provider_id,
                    timestamp_utc=timestamp_utc,
                    timestamp_ist=timestamp_ist,
                    checks=checks
                )
        checks["freshness_check"] = True

        return DataQualityResult(
            status=QualityStatus.VALID,
            reason="Option quote passed all freshness, integrity, and spread sanity checks",
            age_seconds=round(age_seconds, 1) if not is_replay else 0.0,
            provider_id=provider_id,
            timestamp_utc=timestamp_utc,
            timestamp_ist=timestamp_ist,
            checks=checks
        )

    def validate_candle_series(
        self,
        candles: Sequence[IndexCandle]
    ) -> Tuple[List[IndexCandle], List[str]]:
        """
        Validate an array of candles:
          - Eliminates duplicates
          - Detects timestamp regressions / out-of-order bars
          - Verifies chronological ordering
        """
        if not candles:
            return [], ["Empty candle sequence provided"]

        clean_candles: List[IndexCandle] = []
        issues: List[str] = []
        seen_starts = set()

        # Sort by bar_start_utc
        sorted_candles = sorted(candles, key=lambda c: c.bar_start_utc)

        prev_end: Optional[datetime] = None
        for i, c in enumerate(sorted_candles):
            if c.bar_start_utc in seen_starts:
                issues.append(f"Duplicate candle start timestamp detected and omitted: {to_ist_string(c.bar_start_utc)}")
                continue
            seen_starts.add(c.bar_start_utc)

            if prev_end is not None and c.bar_start_utc < prev_end:
                issues.append(f"Timestamp overlap / regression: bar start {to_ist_string(c.bar_start_utc)} precedes previous bar end {to_ist_string(prev_end)}")
                continue

            clean_candles.append(c)
            prev_end = c.bar_end_utc

        return clean_candles, issues
