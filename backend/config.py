import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "nifty_sentinel.db"

# Quantitative Market Parameters
TICKER_SYMBOL = "^NSEI"  # NIFTY 50 Index on Yahoo Finance
LOOKBACK_YEARS = 5
DEFAULT_MIN_TRAIN_252 = 252  # ~1 trading year
DEFAULT_MIN_TRAIN_750 = 750  # ~3 trading years
# Refit interval: 60 trading sessions corresponds to an approximately quarterly trading cadence (~63 sessions per calendar quarter)
DEFAULT_REFIT_INTERVAL = 60
FASTAPI_PORT = 8050

# Known Indian Market Holidays (NSE) for session audit reference
KNOWN_INDIAN_HOLIDAYS = {
    "01-26": "Republic Day",
    "03-08": "Mahashivratri",
    "03-25": "Holi",
    "04-11": "Id-Ul-Fitr",
    "04-14": "Dr. Ambedkar Jayanti",
    "04-17": "Ram Navami",
    "05-01": "Maharashtra Day",
    "06-17": "Bakri Id",
    "07-17": "Muharram",
    "08-15": "Independence Day",
    "10-02": "Mahatma Gandhi Jayanti",
    "10-12": "Dussehra",
    "11-01": "Diwali Laxmi Pujan",
    "11-15": "Gurunanak Jayanti",
    "12-25": "Christmas",
}
