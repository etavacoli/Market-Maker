"""
engine.py - the backtester.

We replay the real trading day event by event and pretend we also had orders
resting in the book. The historical messages tell us what everyone else did;
the engine works out what would have happened to OUR orders.

The key ideas, in plain language:

1. Price-time priority (FIFO). Nasdaq fills resting orders best price first and,
   at the same price, first come first served. When we join the best bid we go
   to the BACK of the queue: everyone already there is ahead of us.

2. Queue ahead. For each of our orders we track how many shares are ahead of us.
   It starts as the displayed size at our price when our order arrives, then
   shrinks when
     - shares at our price trade (the front of the queue is filled first), or
     - an order that was ahead of us is cancelled.
   Orders that join our price after us are behind us, so their cancellations
   don't move us up. We can tell which orders those are from their order IDs.

3. How our orders get filled. There are three ways:
     a. Queue: a market order trades at our price after the queue ahead of us is used up.
     b. Trade-through: a market order trades at a WORSE price than ours (for our
        bid, a lower price). Our better-priced order would have been hit first.
        This is how we get filled after improving on the best price.
     c. Marketable arrival: a new limit order arrives at a price that crosses
        ours (a sell at or below our bid). It would have traded with us.

4. Latency. Our decisions take `latency` seconds to reach the exchange. A cancel
   still in flight can't stop a fill that happens first, and a new order can't
   be filled before it arrives.

5. Post-only. A new order that would cross the spread when it arrives (our bid
   at or above the best ask) is rejected, so we never pay to trade.
   We also only place new orders AT or INSIDE the best price. Deeper in the book
   we would not know how many shares are queued ahead of us (the data only show
   the best level), so we don't pretend to.

What a replay like this can't capture: our orders aren't in the historical data,
so other traders never react to them, and a market order that fills us still
fills the historical orders too. These are the standard limits of backtesting on
replayed history.

Accounting (same idea as Part 1):

    total P&L      = spread capture + inventory P&L
    spread capture = sum over fills of (mid - price) per share bought,
                     and (price - mid) per share sold, using the mid just before the fill
    inventory P&L  = everything else: the mid moving while we hold shares

Exchange fees and rebates are left out here and applied afterwards
(they are just rebate-per-share x shares traded).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

PLACE, CANCEL = 0, 1


@dataclass
class Order:
    """One of our resting orders at the exchange."""

    side: int          # +1 = bid (we buy), -1 = ask (we sell)
    price: int         # LOBSTER price units
    remaining: int     # shares not yet filled
    queue_ahead: int   # shares ahead of us at our price (price-time priority)
    behind: set = field(default_factory=set)  # IDs of orders that joined our price after us


@dataclass
class BacktestResult:
    """Everything a run produces. Money is in LOBSTER price units x shares (divide by 10,000 for dollars)."""

    symbol: str
    strategy: str
    start: float
    end: float
    latency: float
    # one row per fill
    fill_time: np.ndarray
    fill_side: np.ndarray      # +1 we bought, -1 we sold
    fill_price: np.ndarray
    fill_qty: np.ndarray
    fill_mid: np.ndarray       # mid just before the fill
    # snapshots every `snapshot_every` seconds
    snap_time: np.ndarray
    snap_pnl: np.ndarray       # mark-to-market P&L
    snap_inventory: np.ndarray
    # totals at the end of the window
    pnl: float
    spread_capture: float
    final_inventory: int
    shares_traded: int
    orders_sent: int
    orders_rejected: int
    max_abs_inventory: int

    @property
    def inventory_pnl(self):
        return self.pnl - self.spread_capture

    @property
    def n_fills(self):
        return len(self.fill_time)


def run_backtest(market, strategy, start, end, latency=0.010, order_size=100, snapshot_every=1.0):
    """
    Replay `market` between `start` and `end` (seconds after midnight) with our
    orders managed by `strategy`. We start flat (no shares, no cash).

    strategy.decide(i, bid, ask, inventory, current_bid, current_ask) is called
    whenever the best bid/ask changes or we get filled, and returns the prices we
    want our bid and ask at (or None for "no order on that side").
    """
    m = market
    i_start = max(1, int(np.searchsorted(m.time, start, side="left")))
    i_end = int(np.searchsorted(m.time, end, side="left"))
    lo = i_start - 1  # we also need the book state just before the first event

    # Plain Python lists are much faster to index inside a loop than numpy arrays.
    T = m.time[lo:i_end].tolist()
    TYP = m.type[lo:i_end].tolist()
    OID = m.order_id[lo:i_end].tolist()
    SZ = m.size[lo:i_end].tolist()
    PX = m.price[lo:i_end].tolist()
    DIR = m.direction[lo:i_end].tolist()
    BID = m.bid[lo:i_end].tolist()
    BSZ = m.bid_size[lo:i_end].tolist()
    ASK = m.ask[lo:i_end].tolist()
    ASZ = m.ask_size[lo:i_end].tolist()
    MID = m.mid[lo:i_end].tolist()
    decide = strategy.decide

    live = {1: None, -1: None}       # our orders that have reached the exchange
    intended = {1: None, -1: None}   # the price we last asked for on each side (live or in flight)
    pending = deque()                # requests in flight: (arrival time, PLACE/CANCEL, side, price)

    inv = 0
    cash = 0.0
    spread_capture = 0.0
    traded = 0
    sent = 0
    rejected = 0
    max_abs_inv = 0
    fills = []
    snaps = []
    next_snap = start
    first = True

    def fill(order, qty, t, mid_before):
        """Book a fill of `qty` shares of `order`. Returns the changes to cash, inventory and spread capture."""
        order.remaining -= qty
        fills.append((t, order.side, order.price, qty, mid_before))
        if order.remaining == 0:
            live[order.side] = None
            if intended[order.side] == order.price:
                intended[order.side] = None
        s = order.side
        return -s * order.price * qty, s * qty, s * (mid_before - order.price) * qty

    for k in range(1, len(T)):
        t = T[k]

        # 1. Requests that reach the exchange before this event.
        #    (Strictly before: events sharing a timestamp are one instant, e.g. one
        #    market order sweeping several resting orders.)
        while pending and pending[0][0] < t:
            _, kind, side, price = pending.popleft()
            current = live[side]
            if kind == CANCEL:
                if current is not None and current.price == price:
                    live[side] = None
                continue
            best_bid, best_ask = BID[k - 1], ASK[k - 1]
            if side == 1:
                ok = best_bid <= price < best_ask          # at or inside the best bid, not crossing
                ahead = BSZ[k - 1] if price == best_bid else 0
            else:
                ok = best_bid < price <= best_ask
                ahead = ASZ[k - 1] if price == best_ask else 0
            if ok:
                live[side] = Order(side, price, order_size, ahead)
            else:
                rejected += 1
                if intended[side] == price:
                    intended[side] = None

        # 2. P&L snapshots for the quiet time before this event.
        while next_snap < t and next_snap < end:
            snaps.append((next_snap, cash + inv * MID[k - 1], inv))
            next_snap += snapshot_every

        # 3. The market event, and what it does to our orders.
        typ, d, px, sz = TYP[k], DIR[k], PX[k], SZ[k]
        mid_before = MID[k - 1]
        got_filled = False

        order = live[d]  # our order on the same side as the resting order in this event
        if order is not None:
            if typ == 4 or typ == 5:  # a resting order on our side was executed
                qty = 0
                if (px < order.price) if d == 1 else (px > order.price):
                    # Trade-through: it traded at a worse price than ours, so we were hit first.
                    qty = min(order.remaining, sz)
                elif px == order.price and typ == 4:
                    # Same price: the queue ahead of us is filled first.
                    used = min(order.queue_ahead, sz)
                    order.queue_ahead -= used
                    qty = min(order.remaining, sz - used)
                if qty > 0:
                    dc, di, ds = fill(order, qty, t, mid_before)
                    cash += dc; inv += di; spread_capture += ds; traded += qty
                    got_filled = True
            elif typ == 1:
                if px == order.price:
                    order.behind.add(OID[k])  # joined our price after us
            else:  # 2 = partial cancel, 3 = full cancel
                if px == order.price:
                    if OID[k] in order.behind:
                        if typ == 3:
                            order.behind.discard(OID[k])
                    else:
                        order.queue_ahead = max(0, order.queue_ahead - sz)

        if typ == 1:
            # A new limit order that crosses our order on the OTHER side trades with us.
            other = live[-d]
            if other is not None and ((px <= other.price) if d == -1 else (px >= other.price)):
                qty = min(other.remaining, sz)
                dc, di, ds = fill(other, qty, t, mid_before)
                cash += dc; inv += di; spread_capture += ds; traded += qty
                got_filled = True

        if inv > max_abs_inv or -inv > max_abs_inv:
            max_abs_inv = abs(inv)

        # 4. Decide our quotes when something relevant changed.
        if first or got_filled or BID[k] != BID[k - 1] or ASK[k] != ASK[k - 1] \
                or BSZ[k] != BSZ[k - 1] or ASZ[k] != ASZ[k - 1]:
            first = False
            want_bid, want_ask = decide(lo + k, BID[k], ASK[k], inv, intended[1], intended[-1])
            for side, want in ((1, want_bid), (-1, want_ask)):
                current = intended[side]
                if want == current:
                    continue
                arrive = t + latency
                if current is not None:
                    pending.append((arrive, CANCEL, side, current))
                if want is not None:
                    pending.append((arrive, PLACE, side, want))
                    sent += 1
                intended[side] = want

    final_mid = MID[-1]
    while next_snap < end:
        snaps.append((next_snap, cash + inv * final_mid, inv))
        next_snap += snapshot_every

    f = np.array(fills, dtype=float).reshape(-1, 5)
    s = np.array(snaps, dtype=float).reshape(-1, 3)
    return BacktestResult(
        symbol=m.symbol, strategy=getattr(strategy, "name", "strategy"),
        start=start, end=end, latency=latency,
        fill_time=f[:, 0], fill_side=f[:, 1].astype(int), fill_price=f[:, 2],
        fill_qty=f[:, 3].astype(int), fill_mid=f[:, 4],
        snap_time=s[:, 0], snap_pnl=s[:, 1], snap_inventory=s[:, 2].astype(int),
        pnl=cash + inv * final_mid, spread_capture=spread_capture,
        final_inventory=inv, shares_traded=traded, orders_sent=sent,
        orders_rejected=rejected, max_abs_inventory=max_abs_inv,
    )
