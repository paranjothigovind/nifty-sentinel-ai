"""
Free Index Provider for NIFTY Sentinel.
Experimental adapter for publicly accessible index data (e.g., Yahoo Finance ^NSEI).
Strictly documents limitations: 15-minute delayed data, rate limits, no options data.
Never claims real-time exchange guarantees.
"""

from datetime import datetime, timezone, timedelta
from typing import Sequence, Optional, List
import json
import urllib.request
import urllib.error

from backend.market_data.base import MarketDataProvider
from backend.market_data.schemas import (
    IndexCandle,
    OptionContract,
    OptionQuote,
    ProviderHealth,
    DataMode,
    ensure_utc,
)

class FreeIndexProvider(MarketDataProvider):
    """
    Experimental public index data provider.
    Limitations:
      1. Free index feeds for ^NSEI carry a 15-minute statutory exchange delay.
      2. No options chain or options quote data is provided by public index endpoints.
      3. Subject to unannounced IP rate limiting by public endpoints.
      4. Intraday 5m/15m bars are available only for recent rolling trading sessions (~60 days).
    """
    provider_id = "free_index_yahoo"
    display_name = "Experimental Free Index Provider (Yahoo Finance ^NSEI)"

    def __init__(self, symbol: str = "^NSEI"):
        self.symbol = symbol

    def get_index_candles(
        self,
        symbol: str = "^NSEI",
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Sequence[IndexCandle]:
        # Supported intraday intervals: 5m, 15m, 1h, 1d
        valid_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"}
        if interval not in valid_intervals:
            interval = "5m"

        now = datetime.now(timezone.utc)
        if end is None:
            end = now
        if start is None:
            # Default to last 5 trading days for intraday
            start = end - timedelta(days=5)

        period1 = int(start.timestamp())
        period2 = int(end.timestamp())

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
            f"?period1={period1}&period2={period2}&interval={interval}&includePrePost=false"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NiftySentinel/1.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                chart = data.get("chart", {})
                result = chart.get("result")
                if not result:
                    return []
                chart_data = result[0]
                timestamps = chart_data.get("timestamp", [])
                indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]
                opens = indicators.get("open", [])
                highs = indicators.get("high", [])
                lows = indicators.get("low", [])
                closes = indicators.get("close", [])
                volumes = indicators.get("volume", [])

                interval_minutes = 5
                if interval.endswith("m"):
                    interval_minutes = int(interval[:-1])
                elif interval.endswith("h"):
                    interval_minutes = int(interval[:-1]) * 60
                elif interval == "1d":
                    interval_minutes = 375

                candles: List[IndexCandle] = []
                for i, ts in enumerate(timestamps):
                    o = opens[i] if i < len(opens) else None
                    h = highs[i] if i < len(highs) else None
                    l = lows[i] if i < len(lows) else None
                    c = closes[i] if i < len(closes) else None
                    v = volumes[i] if i < len(volumes) else None

                    if None in (o, h, l, c):
                        continue

                    # Sanity check for corrupt negative or zero prices
                    if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l:
                        continue

                    bar_start = datetime.fromtimestamp(ts, tz=timezone.utc)
                    bar_end = bar_start + timedelta(minutes=interval_minutes)

                    candle = IndexCandle(
                        symbol=self.symbol,
                        interval=interval,
                        bar_start_utc=bar_start,
                        bar_end_utc=bar_end,
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=int(v) if v is not None and v > 0 else None,
                        provider_id=self.provider_id,
                        market_timestamp_utc=bar_end,
                        receipt_timestamp_utc=now,
                        data_mode=DataMode.DELAYED,
                        is_delayed=True,
                        delay_seconds=900,  # 15 minutes exchange delay
                    )
                    candles.append(candle)

                candles.sort(key=lambda c: c.bar_start_utc)
                return candles

        except urllib.error.URLError as e:
            # Network or DNS error
            return []
        except Exception:
            # Empty response or unexpected JSON format
            return []

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[object] = None,
        as_of: Optional[datetime] = None
    ) -> Sequence[OptionQuote]:
        # Free public Yahoo index feed has no real-time NIFTY options chain
        return []

    def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionQuote]:
        # Free public Yahoo index feed has no option quotes
        return None

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            display_name=self.display_name,
            is_healthy=True,
            data_mode=DataMode.DELAYED,
            capabilities=["index_candles"],
            last_heartbeat_utc=datetime.now(timezone.utc),
            details={
                "symbol": self.symbol,
                "is_delayed": True,
                "delay_minutes": 15,
                "options_supported": False,
                "notes": "Experimental public feed. Subject to 15-minute exchange delay and rate limits."
            }
        )
