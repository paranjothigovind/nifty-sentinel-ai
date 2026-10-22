"""
Normalized schemas for NIFTY Sentinel Intraday Market Data Layer.
Enforces UTC timezone-aware storage with Asia/Kolkata display formatting,
null-safe field representations, and strict validation models.
"""

from datetime import datetime, date, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, model_validator, field_validator

IST_ZONE = ZoneInfo("Asia/Kolkata")

class DataMode(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    HISTORICAL = "HISTORICAL"
    REPLAY = "REPLAY"
    UNAVAILABLE = "UNAVAILABLE"

class QualityStatus(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"

class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"

def to_ist_string(dt: Optional[datetime]) -> Optional[str]:
    """Convert a UTC datetime to formatted Asia/Kolkata string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST_ZONE).strftime("%Y-%m-%d %H:%M:%S IST")

def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

class IndexCandle(BaseModel):
    """
    Normalized OHLCV candle for index (NIFTY 50).
    """
    symbol: str = Field(..., description="Index symbol (e.g., ^NSEI, NIFTY)")
    interval: str = Field(..., description="Candle interval (e.g., 5m, 15m, 1d)")
    bar_start_utc: datetime = Field(..., description="Bar start timestamp in UTC")
    bar_end_utc: datetime = Field(..., description="Bar completion timestamp in UTC")
    open: float = Field(..., gt=0.0, description="Opening price")
    high: float = Field(..., gt=0.0, description="High price")
    low: float = Field(..., gt=0.0, description="Low price")
    close: float = Field(..., gt=0.0, description="Closing price")
    volume: Optional[int] = Field(None, ge=0, description="Session volume (nullable if unavailable)")
    provider_id: str = Field(..., description="Source provider identifier")
    market_timestamp_utc: datetime = Field(..., description="Original exchange quote/bar timestamp in UTC")
    receipt_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Local receipt timestamp in UTC")
    data_mode: DataMode = Field(..., description="Data mode (LIVE, DELAYED, HISTORICAL, REPLAY, UNAVAILABLE)")
    is_delayed: bool = Field(False, description="Whether the feed carries an exchange delay")
    delay_seconds: Optional[int] = Field(None, ge=0, description="Feed delay in seconds if known")
    bar_start_ist: Optional[str] = None
    bar_end_ist: Optional[str] = None
    market_timestamp_ist: Optional[str] = None

    @field_validator("bar_start_utc", "bar_end_utc", "market_timestamp_utc", "receipt_timestamp_utc", mode="before")
    @classmethod
    def parse_and_localize_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return ensure_utc(dt)
        if isinstance(v, datetime):
            return ensure_utc(v)
        raise ValueError(f"Invalid datetime: {v}")

    @model_validator(mode="after")
    def validate_invariants_and_populate_ist(self) -> "IndexCandle":
        # Validate OHLC invariants
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open}) and Close ({self.close})")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open}) and Close ({self.close})")
        if self.bar_end_utc <= self.bar_start_utc:
            raise ValueError(f"Bar end ({self.bar_end_utc}) must be strictly after bar start ({self.bar_start_utc})")
        
        # Populate IST representation
        self.bar_start_ist = to_ist_string(self.bar_start_utc)
        self.bar_end_ist = to_ist_string(self.bar_end_utc)
        self.market_timestamp_ist = to_ist_string(self.market_timestamp_utc)
        return self

class OptionContract(BaseModel):
    """
    Standardized NIFTY Options Contract specification.
    """
    underlying: str = Field(..., description="Underlying asset symbol (e.g. NIFTY)")
    expiry: date = Field(..., description="Option expiry date")
    strike: float = Field(..., gt=0.0, description="Strike price")
    option_type: OptionType = Field(..., description="Option type (CE = Call, PE = Put)")
    instrument_id: str = Field(..., description="Provider instrument identifier (e.g., NIFTY24OCT24500CE)")

    @model_validator(mode="after")
    def validate_contract(self) -> "OptionContract":
        if self.strike % 50 != 0 and self.strike % 100 != 0:
            # Standard NIFTY strikes are multiples of 50
            pass
        return self

class OptionQuote(BaseModel):
    """
    Normalized options quote. Unavailable fields are represented as null, never zero.
    """
    contract: OptionContract
    bid: Optional[float] = Field(None, gt=0.0, description="Best bid price (null if unavailable)")
    ask: Optional[float] = Field(None, gt=0.0, description="Best ask price (null if unavailable)")
    ltp: Optional[float] = Field(None, gt=0.0, description="Last traded price (null if unavailable)")
    volume: Optional[int] = Field(None, ge=0, description="Contracts traded (null if unavailable)")
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest (null if unavailable)")
    iv: Optional[float] = Field(None, ge=0.0, description="Implied Volatility (annualized, null if unavailable)")
    delta: Optional[float] = Field(None, description="Delta Greek (null if unavailable)")
    gamma: Optional[float] = Field(None, description="Gamma Greek (null if unavailable)")
    theta: Optional[float] = Field(None, description="Theta Greek (null if unavailable)")
    vega: Optional[float] = Field(None, description="Vega Greek (null if unavailable)")
    market_timestamp_utc: datetime = Field(..., description="Market timestamp in UTC")
    receipt_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Local receipt timestamp in UTC")
    market_timestamp_ist: Optional[str] = None
    provider_id: str = Field(..., description="Options data provider ID")
    data_mode: DataMode = Field(..., description="Data mode (LIVE, DELAYED, HISTORICAL, REPLAY, UNAVAILABLE)")
    quality_status: QualityStatus = Field(QualityStatus.VALID, description="Quote quality verification status")

    @field_validator("market_timestamp_utc", "receipt_timestamp_utc", mode="before")
    @classmethod
    def parse_and_localize_quote_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return ensure_utc(dt)
        if isinstance(v, datetime):
            return ensure_utc(v)
        raise ValueError(f"Invalid datetime: {v}")

    @model_validator(mode="after")
    def validate_quotes_and_populate_ist(self) -> "OptionQuote":
        # Validate bid/ask spread sanity if both are present
        if self.bid is not None and self.ask is not None:
            if self.bid > self.ask:
                self.quality_status = QualityStatus.INVALID
        self.market_timestamp_ist = to_ist_string(self.market_timestamp_utc)
        return self

class DataQualityResult(BaseModel):
    """
    Structured data freshness, integrity, and safety check report.
    """
    status: QualityStatus
    reason: Optional[str] = None
    age_seconds: Optional[float] = None
    provider_id: str
    timestamp_utc: Optional[datetime] = None
    timestamp_ist: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    checks: Dict[str, Any] = Field(default_factory=dict)

class ProviderHealth(BaseModel):
    """
    Health and operational status of a market data provider.
    """
    provider_id: str
    display_name: str
    is_healthy: bool
    data_mode: DataMode
    capabilities: List[str] = Field(default_factory=list)
    last_heartbeat_utc: Optional[datetime] = None
    last_heartbeat_ist: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class PaperEntrySafetyCheck(BaseModel):
    """
    Strict safety check before allowing an intraday CALL/PUT paper trading entry.
    Ensures paper trade is blocked if quotes are stale, missing, or unsupported.
    """
    is_entry_permitted: bool
    block_reasons: List[str] = Field(default_factory=list)
    underlying_symbol: str
    contract_id: Optional[str] = None
    directional_signal: str
    data_mode: DataMode
    index_quality: DataQualityResult
    option_quality: Optional[DataQualityResult] = None
    disclaimer: str = (
        "Educational paper trading only. No live trading, broker execution endpoints, "
        "or automated real-money orders are supported or permitted."
    )
