"""
analysis.py - turn a backtest into numbers we can trust (or not).

Markouts: how adverse selection shows up.
    For every fill, imagine closing the position at the mid h seconds later.
        markout(h) = (mid at t+h - price) per share bought
                     (price - mid at t+h) per share sold
    markout(0) is the edge we captured at the moment of the fill (half the
    spread, roughly). If markouts FALL as h grows, the price tended to move
    against us right after we traded: the people trading with us knew something.
    That is adverse selection, and it's how real market makers grade their quotes.

P&L per 5-minute interval, with a block bootstrap.
    One afternoon is a small sample, and 5-minute P&Ls are correlated with their
    neighbours. The block bootstrap resamples whole runs of consecutive intervals
    (15 minutes at a time) to keep that correlation, and gives a 95% confidence
    interval for the average. If the interval includes 0, we can't claim the
    strategy makes or loses money.
"""

import numpy as np

from .data import PRICE_SCALE

HORIZONS = [0, 0.1, 1, 5, 10, 30, 60, 300]   # seconds after the fill


def markouts(result, market, horizons=HORIZONS):
    """Average markout per share at each horizon, in cents."""
    q = result.fill_qty
    if q.sum() == 0:
        return np.full(len(horizons), np.nan)
    s, p, t = result.fill_side, result.fill_price, result.fill_time
    out = []
    for h in horizons:
        mid_h = result.fill_mid if h == 0 else market.mid_at(t + h)
        out.append(np.sum(s * (mid_h - p) * q) / q.sum())
    return np.array(out) / PRICE_SCALE * 100


def interval_pnl(result, minutes=5):
    """P&L earned in each interval, in dollars."""
    edges = np.arange(result.start, result.end + 1e-9, minutes * 60.0)
    idx = np.searchsorted(result.snap_time, edges, side="right") - 1
    pnl = np.where(idx >= 0, result.snap_pnl[np.clip(idx, 0, None)], 0.0)
    pnl[0] = 0.0
    pnl[-1] = result.pnl
    return np.diff(pnl) / PRICE_SCALE


def block_bootstrap_ci(x, block=3, reps=20_000, seed=0, level=0.95):
    """Mean of x and a moving-block bootstrap confidence interval for it."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return x.mean(), lo, hi


def summarize(result, market, rebate_cents=0.0):
    """Headline numbers for one run, in dollars unless stated."""
    shares = result.shares_traded
    pnl = result.pnl / PRICE_SCALE
    intervals = interval_pnl(result)
    mean, lo, hi = block_bootstrap_ci(intervals)
    marks = markouts(result, market)
    return {
        "symbol": result.symbol,
        "strategy": result.strategy,
        "fills": result.n_fills,
        "shares": shares,
        "spread_capture": result.spread_capture / PRICE_SCALE,
        "inventory_pnl": result.inventory_pnl / PRICE_SCALE,
        "pnl": pnl,
        "pnl_with_rebate": pnl + rebate_cents / 100 * shares,
        "pnl_per_share_cents": 100 * pnl / shares if shares else np.nan,
        "breakeven_rebate_cents": max(0.0, -100 * pnl / shares) if shares else np.nan,
        "max_abs_inventory": result.max_abs_inventory,
        "mean_5min": mean, "ci_low": lo, "ci_high": hi,
        "sharpe_5min": mean / intervals.std(ddof=1) if intervals.std() > 0 else np.nan,
        "markout_0s": marks[0], "markout_5s": marks[3], "markout_60s": marks[6],
        "orders_sent": result.orders_sent, "orders_rejected": result.orders_rejected,
    }
