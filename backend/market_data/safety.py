"""
Safety Engine for NIFTY Sentinel Paper Trading Signals.
Enforces non-negotiable safety rules:
  1. Index-only directional research signals MUST NOT generate contract-specific CALL/PUT paper entries.
  2. Paper entries strictly require valid options contract identity and fresh, usable quotes (< 15s).
  3. Real-money live orders and broker execution endpoints are strictly prohibited.
  4. Explicitly tags and propagates data modes (LIVE, DELAYED, HISTORICAL, REPLAY, UNAVAILABLE).
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    DataMode,
    QualityStatus,
    PaperEntrySafetyCheck,
    DataQualityResult,
    ensure_utc,
)
from backend.market_data.validator import FreshnessAndIntegrityValidator

class PaperTradingSafetyEngine:
    """
    Guarantees that no paper trading entry is initiated without valid, fresh market data,
    and prevents any live execution pathway.
    """
    def __init__(self, validator: Optional[FreshnessAndIntegrityValidator] = None):
        self.validator = validator or FreshnessAndIntegrityValidator()

    def evaluate_paper_entry(
        self,
        directional_signal: str,
        target_contract: Optional[OptionContract],
        index_candle: Optional[IndexCandle],
        option_quote: Optional[OptionQuote],
        as_of_utc: Optional[datetime] = None
    ) -> PaperEntrySafetyCheck:
        """
        Evaluate whether an intraday paper entry is permissible under strict safety rules.
        """
        if as_of_utc is None:
            as_of_utc = datetime.now(timezone.utc)
        as_of_utc = ensure_utc(as_of_utc)

        block_reasons: List[str] = []
        is_replay = (index_candle is not None and index_candle.data_mode == DataMode.REPLAY) or \
                    (option_quote is not None and option_quote.data_mode == DataMode.REPLAY)

        # 1. Validate Index Data Quality
        index_quality = self.validator.validate_index_candle(index_candle, as_of_utc, is_replay=is_replay)
        if index_quality.status != QualityStatus.VALID:
            block_reasons.append(f"Index data rejected: {index_quality.reason}")

        # 2. Rule 1: Index-only signals cannot generate contract-specific CALL/PUT entries
        if target_contract is None:
            block_reasons.append(
                "INDEX_ONLY_SIGNAL_BLOCK: Index directional signal alone cannot trigger a contract-specific "
                "CALL/PUT paper entry without an identified options contract."
            )
            return PaperEntrySafetyCheck(
                is_entry_permitted=False,
                block_reasons=block_reasons,
                underlying_symbol=index_candle.symbol if index_candle else "NIFTY",
                contract_id=None,
                directional_signal=directional_signal,
                data_mode=index_candle.data_mode if index_candle else DataMode.UNAVAILABLE,
                index_quality=index_quality,
                option_quality=None,
            )

        # 3. Rule 2: Options Contract Identity and Fresh Usable Quote
        if option_quote is None:
            block_reasons.append(
                f"MISSING_OPTION_QUOTE: No market quote available for target contract {target_contract.instrument_id}."
            )
            option_quality = DataQualityResult(
                status=QualityStatus.UNAVAILABLE,
                reason="Option quote is None; no feed available",
                provider_id="none",
                missing_fields=["quote", "bid", "ask", "ltp"]
            )
        else:
            option_quality = self.validator.validate_option_quote(option_quote, as_of_utc, is_replay=is_replay)
            if option_quality.status != QualityStatus.VALID:
                block_reasons.append(f"Option quote rejected ({option_quality.status}): {option_quality.reason}")

            # Contract matching verification
            if option_quote.contract.instrument_id != target_contract.instrument_id:
                block_reasons.append(
                    f"Contract mismatch: Target {target_contract.instrument_id} != Quote {option_quote.contract.instrument_id}"
                )

        # 4. Check Active Data Mode
        effective_mode = DataMode.UNAVAILABLE
        if option_quote is not None:
            effective_mode = option_quote.data_mode
        elif index_candle is not None:
            effective_mode = index_candle.data_mode

        is_permitted = len(block_reasons) == 0

        return PaperEntrySafetyCheck(
            is_entry_permitted=is_permitted,
            block_reasons=block_reasons,
            underlying_symbol=target_contract.underlying if target_contract else "NIFTY",
            contract_id=target_contract.instrument_id if target_contract else None,
            directional_signal=directional_signal,
            data_mode=effective_mode,
            index_quality=index_quality,
            option_quality=option_quality,
        )

    @staticmethod
    def execute_live_order(*args, **kwargs):
        """
        Hard safety assertion: Real trading and broker live execution are strictly forbidden.
        """
        raise NotImplementedError(
            "LIVE_TRADING_FORBIDDEN: Live brokerage order execution is permanently disabled in NIFTY Sentinel. "
            "This application is strictly an educational quantitative research and paper evaluation terminal."
        )
