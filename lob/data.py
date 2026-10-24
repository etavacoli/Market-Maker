"""
data.py - load one day of real Nasdaq order-book data (LOBSTER).

LOBSTER (https://lobsterdata.com) rebuilds the Nasdaq limit order book from the
exchange's own message feed. Its free sample covers five stocks on
21 June 2012, from 09:30 to 16:00: AAPL, AMZN, GOOG, INTC and MSFT.

Every row is one event in the order book, in time order:

    time        seconds after midnight (34200.0 = 09:30:00), nanosecond precision
    type        1 = new limit order           4 = execution of a visible order
                2 = partial cancellation      5 = execution of a hidden order
                3 = full cancellation
    order_id    the exchange's ID for the resting order the event refers to
    size        number of shares
    price       dollars x 10,000   ($583.17 is stored as 5,831,700)
    direction   side of the RESTING order: +1 = buy order (bid), -1 = sell order (ask)

So an execution (type 4) with direction +1 means a resting BUY order was hit:
somebody sold into the bids with a market order.

Each row also carries the best bid and ask AFTER that event:

    bid, bid_size, ask, ask_size

Prices stay in LOBSTER's integer units throughout the code (one cent = 100),
which avoids floating-point rounding when comparing prices. They are converted
to dollars only when results are reported.

Two input formats are accepted:

  1. The official LOBSTER sample files: a message file and an orderbook file,
     e.g. AAPL_2012-06-21_34200000_57600000_message_10.csv and
          AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
     (any level works; only the best bid and ask are used).
  2. A "compact" CSV with both joined into one file, which is what
     download_samples() fetches: LOBSTER_<SYMBOL>_compact.csv.gz.
"""

from dataclasses import dataclass
from pathlib import Path
import glob
import urllib.request

import numpy as np
import pandas as pd

SYMBOLS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]

PRICE_SCALE = 10_000   # LOBSTER price units per dollar
TICK = 100             # one cent, the Nasdaq tick size for these stocks

OPEN = 34_200.0        # 09:30:00 in seconds after midnight
CLOSE = 57_600.0       # 16:00:00

# Compact copies of the LOBSTER sample day (all messages + best bid/ask),
# published by the GitHub project m4rpg/algo-trading-liquidity. For AAPL the
# message columns were checked row-for-row against the official LOBSTER file.
COMPACT_URL = ("https://raw.githubusercontent.com/m4rpg/algo-trading-liquidity/"
               "main/raw_data/LOBSTER_{symbol}_compact.csv.gz")

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]


@dataclass
class Market:
    """One stock-day of order-book events, as numpy arrays (one entry per event)."""

    symbol: str
    time: np.ndarray        # float64, seconds after midnight
    type: np.ndarray        # int64, event type 1-5
    order_id: np.ndarray    # int64
    size: np.ndarray        # int64, shares
    price: np.ndarray       # int64, LOBSTER price units
    direction: np.ndarray   # int64, +1 bid side, -1 ask side
    bid: np.ndarray         # int64, best bid AFTER the event
    bid_size: np.ndarray    # int64, shares at the best bid
    ask: np.ndarray         # int64, best ask AFTER the event
    ask_size: np.ndarray    # int64, shares at the best ask
    tick: int = TICK

    def __post_init__(self):
        # Derived series used everywhere. Mid is the plain average of the best
        # bid and ask; spread is how far apart they are.
        self.mid = 0.5 * (self.bid + self.ask)
        self.spread = self.ask - self.bid

    def __len__(self):
        return len(self.time)

    def mid_at(self, times):
        """Mid price in force at each of `times` (the state after the last event at or before it)."""
        idx = np.searchsorted(self.time, times, side="right") - 1
        idx = np.clip(idx, 0, len(self.time) - 1)
        return self.mid[idx]


def market_from_frame(symbol, df):
    """Build a Market from a DataFrame with the message columns plus bid1/bidsize1/ask1/asksize1."""
    df = df.sort_values("time", kind="stable")
    # Keep only rows where both sides of the book exist. (LOBSTER marks an
    # empty side with dummy prices of +/-9,999,999,999.) The sample day has none,
    # but other days can.
    ok = (df["bid1"] > 0) & (df["ask1"] < 9_000_000_000) & (df["ask1"] > df["bid1"])
    df = df[ok]
    arr = lambda col: df[col].to_numpy()
    return Market(
        symbol=symbol,
        time=arr("time").astype(np.float64),
        type=arr("type").astype(np.int64),
        order_id=arr("order_id").astype(np.int64),
        size=arr("size").astype(np.int64),
        price=arr("price").astype(np.int64),
        direction=arr("direction").astype(np.int64),
        bid=arr("bid1").astype(np.int64),
        bid_size=arr("bidsize1").astype(np.int64),
        ask=arr("ask1").astype(np.int64),
        ask_size=arr("asksize1").astype(np.int64),
    )


def load_market(symbol, data_dir="data"):
    """Load one stock from data_dir, in either format (see the module docstring)."""
    data_dir = Path(data_dir)

    compact = data_dir / f"LOBSTER_{symbol}_compact.csv.gz"
    if compact.exists():
        return market_from_frame(symbol, pd.read_csv(compact))

    # Official LOBSTER files, possibly inside an unzipped sample folder.
    messages = sorted(glob.glob(str(data_dir / "**" / f"{symbol}_*_message_*.csv"), recursive=True))
    if messages:
        message_file = messages[0]
        book_file = message_file.replace("_message_", "_orderbook_")
        msg = pd.read_csv(message_file, header=None, usecols=range(6), names=MESSAGE_COLUMNS)
        book = pd.read_csv(book_file, header=None, usecols=range(4),
                           names=["ask1", "asksize1", "bid1", "bidsize1"])
        return market_from_frame(symbol, pd.concat([msg, book], axis=1))

    raise FileNotFoundError(
        f"No data for {symbol} in {data_dir}/. Run `python run_lob.py --download`, "
        "or put the official LOBSTER sample files there.")


def download_samples(data_dir="data", symbols=SYMBOLS):
    """Fetch the five compact sample files (about 25 MB in total) into data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        target = data_dir / f"LOBSTER_{symbol}_compact.csv.gz"
        if target.exists():
            continue
        print(f"downloading {symbol} ...")
        urllib.request.urlretrieve(COMPACT_URL.format(symbol=symbol), target)
