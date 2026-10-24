"""
calibrate.py - measure Part 1's fill model on real trades instead of assuming it.

Part 1 assumed the rate at which a quote gets filled decays exponentially with
its distance from the mid,

    fill rate at distance delta = A * exp(-k * delta),

with A and k taken from Avellaneda & Stoikov (2008). Here we estimate them.

For every market order (one or more executions with the same timestamp and side,
in consecutive messages) we record how far from the mid it reached:

    depth = mid before - lowest price traded     for a sell market order (hits bids)
    depth = highest price traded - mid before    for a buy market order (lifts asks)

A quote at distance delta from the mid is reached by every market order with
depth >= delta, so the fill rate at delta is

    lambda(delta) = (number of market orders with depth >= delta) / (seconds of trading).

Taking logs turns the exponential into a straight line,

    log lambda(delta) = log A - k * delta,

which we fit by least squares. How straight that line really is tells us whether
the textbook model describes the stock at all.
"""

from dataclasses import dataclass

import numpy as np

from .data import CLOSE, OPEN, PRICE_SCALE


def market_order_depths(market):
    """Depth reached by each market order, in dollars, and its side (+1 sell order hitting bids, -1 buy lifting asks)."""
    m = market
    idx = np.nonzero((m.type == 4) | (m.type == 5))[0]
    idx = idx[idx > 0]
    same = (m.time[idx[1:]] == m.time[idx[:-1]]) \
        & (m.direction[idx[1:]] == m.direction[idx[:-1]]) \
        & (idx[1:] == idx[:-1] + 1)
    starts = np.r_[0, np.nonzero(~same)[0] + 1]
    first = idx[starts]
    side = m.direction[first]
    mid_before = m.mid[first - 1]
    prices = m.price[idx].astype(float)
    lowest = np.minimum.reduceat(prices, starts)
    highest = np.maximum.reduceat(prices, starts)
    depth = np.where(side == 1, mid_before - lowest, highest - mid_before)
    return depth / PRICE_SCALE, side


@dataclass
class IntensityFit:
    A: float            # fill rate at the mid, market orders per second (both sides together)
    k: float            # decay per dollar of distance
    r2: float           # fit quality in log space
    delta: np.ndarray   # grid of distances, dollars
    rate: np.ndarray    # measured lambda(delta), per second
    n_orders: int


def fit_intensity(market, seconds=CLOSE - OPEN, min_count=20):
    depth, _ = market_order_depths(market)
    step = market.tick / PRICE_SCALE / 2          # half a cent
    top = np.quantile(depth, 0.995)
    delta = np.arange(step, top + step / 2, step)
    counts = np.array([(depth >= d - 1e-12).sum() for d in delta])
    ok = counts >= min_count
    delta, rate = delta[ok], counts[ok] / seconds

    slope, intercept = np.polyfit(delta, np.log(rate), 1)
    fitted = intercept + slope * delta
    resid = np.log(rate) - fitted
    r2 = 1 - resid.var() / np.log(rate).var() if len(delta) > 2 else float("nan")
    return IntensityFit(A=float(np.exp(intercept)), k=float(-slope), r2=float(r2),
                        delta=delta, rate=rate, n_orders=len(depth))
