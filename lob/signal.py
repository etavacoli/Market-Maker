"""
signal.py - a short-term fair value from order-book imbalance (the micro-price).

The mid treats the best bid and the best ask as equally important. But if 9,000
shares are waiting to buy at the bid and only 300 to sell at the ask, the next
move is more likely up: the ask queue is much closer to running out.
Imbalance measures that:

    imbalance = (bid size - ask size) / (bid size + ask size)        between -1 and 1

Two ways to turn it into a fair value:

  weighted mid = mid + imbalance x spread / 2
                 A fixed formula. Simple, but it trusts imbalance completely.

  micro-price  = mid + g(imbalance, spread)
                 g is LEARNED from data: the average change in the mid over the next
                 few seconds, for each bucket of (imbalance, spread). This follows the
                 idea in Stoikov (2018), "The micro-price: a high-frequency estimator
                 of future prices", in a simpler empirical form (bucket averages
                 instead of his Markov-chain construction).

Walk-forward fitting, so nothing is learned from the future:
  - The day is cut into 30-minute folds. The g used in fold k is fitted only on
    folds before k.
  - Training samples whose label (the mid `horizon` seconds later) would land
    inside fold k are dropped. This is called purging: without it, the last few
    seconds of training labels would quietly contain test-period prices.
  - Fold 0 has no history, so it uses g = 0, i.e. the plain mid.

Small buckets are shrunk toward 0 (g = sum / (count + shrink)), so a bucket seen
only a handful of times can't produce a large, noisy adjustment.
"""

from dataclasses import dataclass

import numpy as np

from .data import CLOSE, OPEN

IMBALANCE_BINS = 10
SPREAD_EDGES = [1.5, 2.5, 4.5, 9.5]   # spread in ticks -> buckets {1}, {2}, {3-4}, {5-9}, {10+}
N_BUCKETS = IMBALANCE_BINS * (len(SPREAD_EDGES) + 1)


def imbalance(bid_size, ask_size):
    return (bid_size - ask_size) / (bid_size + ask_size)


def bucket_of(bid, ask, bid_size, ask_size, tick):
    """Bucket number (0 .. N_BUCKETS-1) for each book state."""
    imb = imbalance(bid_size, ask_size)
    imb_bin = np.clip(((imb + 1.0) / 2.0 * IMBALANCE_BINS).astype(int), 0, IMBALANCE_BINS - 1)
    spread_bin = np.digitize((ask - bid) / tick, SPREAD_EDGES)
    return spread_bin * IMBALANCE_BINS + imb_bin


def fold_edges(start=OPEN, end=CLOSE, minutes=30):
    return np.arange(start, end + 1e-9, minutes * 60.0)


def fit_table(buckets, labels, shrink):
    sums = np.bincount(buckets, weights=labels, minlength=N_BUCKETS)
    counts = np.bincount(buckets, minlength=N_BUCKETS)
    return sums / (counts + shrink)


@dataclass
class WalkForward:
    horizon: float
    edges: np.ndarray
    tables: np.ndarray          # (n_folds, N_BUCKETS): g for each fold, price units
    event_adjust: np.ndarray    # g looked up for every event (what Strategy C uses)
    sample_time: np.ndarray     # evaluation grid (every `step` seconds)
    sample_fold: np.ndarray
    sample_imbalance: np.ndarray
    label: np.ndarray           # realised change in mid over `horizon`, price units
    pred_micro: np.ndarray      # walk-forward micro-price prediction of that change
    pred_weighted: np.ndarray   # weighted-mid prediction of that change


def walk_forward(market, horizon=5.0, fold_minutes=30, step=0.1, shrink=200.0):
    m = market
    edges = fold_edges(minutes=fold_minutes)
    n_folds = len(edges) - 1

    # Samples on a regular time grid, so busy seconds don't count more than quiet ones.
    times = np.arange(OPEN, CLOSE - horizon, step)
    j = np.searchsorted(m.time, times, side="right") - 1
    keep = j >= 0
    times, j = times[keep], j[keep]
    bid, ask = m.bid[j], m.ask[j]
    bsz, asz = m.bid_size[j], m.ask_size[j]
    buckets = bucket_of(bid, ask, bsz, asz, m.tick)
    label = m.mid_at(times + horizon) - m.mid[j]
    imb = imbalance(bsz, asz)
    weighted = imb * (ask - bid) / 2.0

    tables = np.zeros((n_folds, N_BUCKETS))
    for k in range(1, n_folds):
        train = times + horizon <= edges[k]          # purged: label ends before fold k starts
        tables[k] = fit_table(buckets[train], label[train], shrink)

    sample_fold = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, n_folds - 1)
    pred_micro = tables[sample_fold, buckets]

    event_buckets = bucket_of(m.bid, m.ask, m.bid_size, m.ask_size, m.tick)
    event_fold = np.clip(np.searchsorted(edges, m.time, side="right") - 1, 0, n_folds - 1)
    event_adjust = tables[event_fold, event_buckets]

    return WalkForward(horizon, edges, tables, event_adjust, times, sample_fold, imb,
                       label, pred_micro, weighted)


def out_of_sample_r2(label, prediction):
    """1 - (squared error of the prediction) / (squared error of predicting no change).
    Positive = better than the plain mid; negative = worse than doing nothing."""
    return 1.0 - np.sum((label - prediction) ** 2) / np.sum(label ** 2)
