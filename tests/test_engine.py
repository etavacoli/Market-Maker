"""
Tests for the backtest engine and the walk-forward signal.

Each engine test builds a tiny hand-made order book where the right answer is
obvious, then checks the engine gets it. Run from the repo root:

    python -m pytest tests -q
"""

import numpy as np
import pandas as pd
import pytest

from lob.data import CLOSE, OPEN, market_from_frame
from lob.engine import run_backtest
from lob.signal import walk_forward
from lob.strategies import strategy_a

BID, ASK = 1_000_000, 1_000_100     # $100.00 / $100.01, a one-tick spread
WIDE_ASK = 1_000_500                # $100.05, a five-tick spread
T0 = OPEN + 60.0


def make_market(rows):
    """rows: (seconds after T0, type, order_id, size, price, direction, bid, bid_size, ask, ask_size)."""
    df = pd.DataFrame(rows, columns=["time", "type", "order_id", "size", "price", "direction",
                                     "bid1", "bidsize1", "ask1", "asksize1"])
    df["time"] += T0
    return market_from_frame("TEST", df)


class Scripted:
    """A strategy that always wants the same quotes (or follows a function of time)."""

    name = "scripted"

    def __init__(self, market, want):
        self.market = market
        self.want = want  # (bid, ask) or a function of event time -> (bid, ask)

    def decide(self, i, bid, ask, inventory, current_bid, current_ask):
        return self.want(self.market.time[i]) if callable(self.want) else self.want


def run(market, want, latency=0.001):
    return run_backtest(market, Scripted(market, want), T0, T0 + 100, latency=latency)


def test_joining_the_bid_waits_for_the_queue_ahead():
    # 300 shares are at the bid when we join, so the first 300 traded are not ours.
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, ASK, 300),
        (1, 1, 2, 100, ASK + 100, -1, BID, 300, ASK, 300),   # a quiet event; our order arrives before it
        (2, 4, 1, 200, BID, 1, BID, 100, ASK, 300),          # 200 trade: all ahead of us
        (3, 4, 1, 150, BID, 1, BID, 100, ASK, 300),          # 100 more ahead, then 50 to us
        (4, 4, 9, 100, BID, 1, BID, 100, ASK, 300),          # the rest of our 100
    ])
    r = run(m, (BID, None))
    assert list(r.fill_qty) == [50, 50]
    assert r.final_inventory == 100


def test_cancels_ahead_move_us_up_but_cancels_behind_do_not():
    m = make_market([
        (0, 1, 5, 300, BID, 1, BID, 300, ASK, 300),
        (1, 1, 6, 100, ASK + 100, -1, BID, 300, ASK, 300),
        (2, 1, 99, 100, BID, 1, BID, 500, ASK, 300),         # joins after us: behind
        (3, 3, 99, 100, BID, 1, BID, 400, ASK, 300),         # cancelled: doesn't help us
        (4, 3, 5, 200, BID, 1, BID, 200, ASK, 300),          # 200 ahead of us cancelled
        (5, 4, 7, 150, BID, 1, BID, 100, ASK, 300),          # 100 still ahead, then 50 to us
    ])
    r = run(m, (BID, None))
    assert list(r.fill_qty) == [50]


def test_improved_bid_is_filled_by_a_trade_at_a_worse_price():
    # Our bid at $100.02 sits above the $100.00 best bid. A sell that trades at
    # $100.00 would have hit us first.
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, WIDE_ASK, 300),
        (1, 1, 2, 100, WIDE_ASK + 100, -1, BID, 300, WIDE_ASK, 300),
        (2, 4, 1, 300, BID, 1, BID, 100, WIDE_ASK, 300),
    ])
    r = run(m, (BID + 200, None))
    assert list(r.fill_qty) == [100]
    assert r.fill_price[0] == BID + 200
    assert r.spread_capture == pytest.approx(((BID + WIDE_ASK) / 2 - (BID + 200)) * 100)


def test_an_order_cannot_fill_before_it_reaches_the_exchange():
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, WIDE_ASK, 300),
        (0.5, 4, 1, 100, BID, 1, BID, 200, WIDE_ASK, 300),   # our order is still in flight
        (2.0, 4, 1, 100, BID, 1, BID, 100, WIDE_ASK, 300),   # now it has arrived
    ])
    r = run(m, (BID + 200, None), latency=1.0)
    assert list(r.fill_time - T0) == [2.0]


def test_a_cancel_in_flight_does_not_stop_an_earlier_fill():
    # We decide to cancel at t=1, but the cancel only arrives at t=2; a trade at t=1.5 still fills us.
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, WIDE_ASK, 300),
        (0.1, 1, 2, 100, WIDE_ASK + 100, -1, BID, 300, WIDE_ASK, 300),
        (1.0, 1, 3, 100, WIDE_ASK + 200, -1, BID, 300, WIDE_ASK, 400),
        (1.5, 4, 1, 100, BID, 1, BID, 200, WIDE_ASK, 400),
        (2.5, 4, 1, 100, BID, 1, BID, 100, WIDE_ASK, 400),
    ])
    want = lambda t: (BID + 200, None) if t < T0 + 1.0 else (None, None)
    r = run(m, want, latency=0.05)
    assert r.n_fills == 0  # with fast cancels the t=1.5 trade misses us
    r = run(m, want, latency=1.0)
    assert list(r.fill_time - T0) == [1.5]  # with slow cancels it doesn't


def test_post_only_rejects_an_order_that_would_cross():
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, ASK, 300),
        (1, 1, 2, 100, ASK + 100, -1, BID, 300, ASK, 300),
        (2, 4, 1, 300, BID, 1, BID - 100, 300, ASK, 300),
    ])
    r = run(m, (ASK, None))          # a bid at the best ask would trade immediately
    assert r.orders_rejected == 1 and r.n_fills == 0


def test_a_new_order_that_crosses_ours_trades_with_us():
    # Our ask improves to $100.03; a buyer then posts a limit order at $100.03. It would have bought from us.
    m = make_market([
        (0, 1, 1, 300, BID, 1, BID, 300, WIDE_ASK, 300),
        (1, 1, 2, 100, BID - 100, 1, BID, 300, WIDE_ASK, 300),
        (2, 1, 3, 100, BID + 300, 1, BID + 300, 100, WIDE_ASK, 300),
    ])
    r = run(m, (None, BID + 300))
    assert list(r.fill_side) == [-1] and list(r.fill_qty) == [100]


def test_inventory_limit_stops_buying():
    rows = [(0, 1, 1, 100, BID, 1, BID, 100, ASK, 300)]
    for k in range(1, 30):
        rows.append((k, 4, 1, 1000, BID, 1, BID, 100, ASK, 300))   # sellers keep hitting the bid
    m = make_market(rows)
    r = run_backtest(m, strategy_a(max_inventory=200), T0, T0 + 100, latency=0.001)
    assert r.final_inventory == 200


def random_market(seed=0, n=40_000):
    """A random but internally consistent book: a mid that random-walks on a 1-cent grid."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(OPEN, CLOSE, n))
    mid_ticks = 10_000 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n))
    spread = rng.choice([1, 1, 2, 3], n)
    bid = (mid_ticks * 100).astype(np.int64)
    ask = bid + spread * 100
    bsz = rng.integers(100, 2000, n)
    asz = rng.integers(100, 2000, n)
    types = rng.choice([1, 3, 4], n, p=[0.5, 0.4, 0.1])
    direction = rng.choice([1, -1], n)
    price = np.where(direction == 1, bid, ask)
    size = rng.integers(1, 300, n)
    df = pd.DataFrame({"time": times, "type": types, "order_id": np.arange(n), "size": size,
                       "price": price, "direction": direction, "bid1": bid, "bidsize1": bsz,
                       "ask1": ask, "asksize1": asz})
    return market_from_frame("RAND", df)


def test_pnl_adds_up_from_the_fills_alone():
    # Recompute P&L and its two parts from the list of fills, independently of the engine's running totals.
    m = random_market()
    r = run_backtest(m, strategy_a(), OPEN, CLOSE, latency=0.01)
    assert r.n_fills > 50
    s, q, p, mid0 = r.fill_side, r.fill_qty, r.fill_price, r.fill_mid
    final_mid = m.mid[np.searchsorted(m.time, CLOSE) - 1]
    assert r.pnl == pytest.approx(np.sum(s * q * (final_mid - p)))
    assert r.spread_capture == pytest.approx(np.sum(s * q * (mid0 - p)))
    assert r.final_inventory == np.sum(s * q)


def test_micro_price_never_learns_from_the_future():
    # Scramble everything after 12:00. The tables used up to 12:00 must not change.
    m = random_market(seed=1)
    before = walk_forward(m, horizon=5.0)
    cut = 43_200.0
    scrambled = random_market(seed=2)
    late = m.time > cut
    for name in ("bid", "ask", "bid_size", "ask_size"):
        getattr(m, name)[late] = getattr(scrambled, name)[late]
    m.__post_init__()  # recompute mid and spread
    after = walk_forward(m, horizon=5.0)
    folds_before_cut = int(np.searchsorted(before.edges, cut, side="right"))  # folds that start at or before 12:00
    assert np.allclose(before.tables[:folds_before_cut], after.tables[:folds_before_cut])
    assert not np.allclose(before.tables, after.tables)  # sanity: the scrambling did change later folds
