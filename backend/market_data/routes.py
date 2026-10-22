"""
FastAPI Routes for NIFTY Sentinel Intraday Market Data Layer.
Preserves existing endpoints while adding normalized intraday index and options data.
"""

from datetime import datetime, date, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Body
from pydantic import BaseModel, Field

from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    OptionType,
    ProviderHealth,
    DataQualityResult,
    PaperEntrySafetyCheck,
    DataMode,
)
from backend.market_data.service import market_data_service

router = APIRouter(prefix="/market-data", tags=["Market Data Layer"])

class EvaluateEntryRequest(BaseModel):
    directional_signal: str = Field(..., description="Directional forecast: UP, DOWN, or NEUTRAL")
    underlying: str = Field("NIFTY", description="Underlying symbol")
    expiry: Optional[date] = Field(None, description="Contract expiry date")
    strike: Optional[float] = Field(None, description="Strike price")
    option_type: Optional[OptionType] = Field(None, description="CE or PE")
    instrument_id: Optional[str] = Field(None, description="Optional specific provider instrument ID")

class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., description="Provider mode: replay or free_index")

@router.get("/health", response_model=List[ProviderHealth])
def get_providers_health():
    """
    Return health and capabilities of all configured market data providers.
    """
    return market_data_service.get_all_providers_health()

@router.get("/status")
def get_market_data_status():
    """
    Return compact status summary for the UI data-status panel.
    Shows active provider, data mode, latest market timestamp, quote age, coverage, and trade signal block reasons.
    """
    return market_data_service.get_status_summary()

@router.get("/candles", response_model=List[IndexCandle])
def get_index_candles(
    symbol: str = Query("NIFTY", description="Index symbol"),
    interval: str = Query("5m", description="Candle interval (5m, 15m, 1d)"),
    limit: int = Query(50, ge=1, le=500, description="Max candles to return")
):
    """
    Fetch normalized intraday index candles.
    """
    return market_data_service.get_index_candles(symbol=symbol, interval=interval, limit=limit)

@router.get("/option-chain")
def get_option_chain(
    underlying: str = Query("NIFTY", description="Underlying symbol"),
    expiry: Optional[date] = Query(None, description="Expiry date filter")
):
    """
    Fetch normalized options chain quotes.
    Returns structured unavailability if options provider is unavailable.
    """
    quotes = market_data_service.get_option_chain(underlying=underlying, expiry=expiry)
    if not quotes:
        provider = market_data_service.current_options_provider
        return {
            "status": "UNAVAILABLE",
            "provider_id": provider.provider_id,
            "data_mode": DataMode.UNAVAILABLE.value,
            "message": "Options chain data is currently unavailable. No verified real-time options feed is active.",
            "quotes": []
        }
    return {
        "status": "AVAILABLE",
        "provider_id": market_data_service.current_options_provider.provider_id,
        "data_mode": market_data_service.current_options_provider.health().data_mode.value,
        "count": len(quotes),
        "quotes": [q.model_dump() for q in quotes]
    }

@router.get("/quote")
def get_option_quote(
    underlying: str = Query("NIFTY", description="Underlying symbol"),
    strike: float = Query(..., description="Strike price"),
    option_type: OptionType = Query(..., description="CE or PE"),
    expiry: date = Query(..., description="Contract expiry date"),
    instrument_id: Optional[str] = Query(None, description="Optional instrument ID")
):
    """
    Fetch single contract quote with bid/ask/LTP.
    """
    contract = OptionContract(
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        instrument_id=instrument_id or f"{underlying}{expiry.strftime('%y%b').upper()}{int(strike)}{option_type.value}"
    )
    quote = market_data_service.get_option_quote(contract)
    if quote is None:
        raise HTTPException(
            status_code=404,
            detail=f"Quote for {contract.instrument_id} is unavailable from current provider."
        )
    return quote

@router.get("/quality", response_model=DataQualityResult)
def get_data_quality_report():
    """
    Evaluate and return current index data quality and freshness report.
    """
    candles = market_data_service.get_index_candles(limit=1)
    latest_candle = candles[-1] if candles else None
    return market_data_service.validator.validate_index_candle(
        latest_candle,
        is_replay=(market_data_service.active_mode == "replay")
    )

@router.post("/evaluate-paper-entry", response_model=PaperEntrySafetyCheck)
def evaluate_paper_entry(payload: EvaluateEntryRequest):
    """
    Enforce safety rules before initiating an intraday CALL/PUT paper entry:
      1. Index-only signals cannot generate contract entries without an identified contract.
      2. Rejects stale index (>60s) or stale options quote (>15s).
      3. Verifies non-crossed book and positive prices.
      4. Live orders are permanently disabled.
    """
    contract = None
    if payload.strike and payload.expiry and payload.option_type:
        contract = OptionContract(
            underlying=payload.underlying,
            expiry=payload.expiry,
            strike=payload.strike,
            option_type=payload.option_type,
            instrument_id=payload.instrument_id or f"{payload.underlying}{payload.expiry.strftime('%y%b').upper()}{int(payload.strike)}{payload.option_type.value}"
        )

    return market_data_service.evaluate_paper_entry(
        directional_signal=payload.directional_signal,
        target_contract=contract
    )

@router.post("/mode")
def set_provider_mode(payload: ModeSwitchRequest):
    """
    Switch active provider mode between 'replay' (deterministic CSV) and 'free_index' (Yahoo Finance).
    """
    market_data_service.set_active_mode(payload.mode)
    return {
        "status": "success",
        "active_mode": market_data_service.active_mode,
        "summary": market_data_service.get_status_summary()
    }
