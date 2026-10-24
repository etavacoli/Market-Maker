"""
strategies.py - where to put our bid and ask, given the book and our inventory.

All three strategies share one rule set; they differ only in their settings.
This mirrors Part 1, where Strategy B with skew = 0 is exactly Strategy A.

    fair value   = mid                         (Strategies A and B)
                 = mid + micro-price adjustment (Strategy C, see signal.py)
    reservation  = fair value - skew x (inventory in round lots of 100 shares)
    target bid   = reservation - theta x (spread / 2)
    target ask   = reservation + theta x (spread / 2)

theta ("aggressiveness") sets how far from the reservation price we quote,
as a fraction of the current half-spread:
    theta = 1   quote exactly at the best bid and ask when flat
    theta < 1   quote inside the spread (better prices, more fills, less edge)

Turning a target into an order (prices snap to the 1-cent tick):
    - target at or better than the best price -> join or improve it
      (but never cross the spread; post-only)
    - target behind the best price -> no new order; keep an existing order only
      if it is still at or below the target (we know our queue position there)
    - inventory limit reached -> no order on the side that would add to it

How the skew works on a one-tick-spread stock (INTC, MSFT): improving is
impossible, so the only choices are "join the best price" or "step back".
Long inventory pushes the reservation price down, so the bid target falls
behind the best bid and we stop buying; the ask stays at the touch.
"""

import math

LOT = 100  # shares per order (one round lot)


class QuotingStrategy:
    def __init__(self, name, tick=100, theta=1.0, skew=0.0, max_inventory=500,
                 fair_adjust=None, lot=LOT):
        self.name = name
        self.tick = tick
        self.theta = theta                  # fraction of the half-spread to quote away from reservation
        self.skew = skew                    # ticks of reservation shift per round lot held
        self.max_inventory = max_inventory  # hard position limit in shares
        self.fair_adjust = fair_adjust      # per-event fair value adjustment (price units), or None
        self.lot = lot

    def decide(self, i, bid, ask, inventory, current_bid, current_ask):
        tick = self.tick
        fair = 0.5 * (bid + ask)
        if self.fair_adjust is not None:
            fair += self.fair_adjust[i]
        reservation = fair - self.skew * tick * inventory / self.lot
        half = self.theta * 0.5 * (ask - bid)

        target_bid = math.floor((reservation - half) / tick + 1e-9) * tick
        target_ask = math.ceil((reservation + half) / tick - 1e-9) * tick

        if inventory + self.lot > self.max_inventory:
            want_bid = None
        elif target_bid >= bid:
            want_bid = min(target_bid, ask - tick)          # join or improve, never cross
        elif current_bid is not None and current_bid <= target_bid:
            want_bid = current_bid                          # keep our place behind the touch
        else:
            want_bid = None

        if inventory - self.lot < -self.max_inventory:
            want_ask = None
        elif target_ask <= ask:
            want_ask = max(target_ask, bid + tick)
        elif current_ask is not None and current_ask >= target_ask:
            want_ask = current_ask
        else:
            want_ask = None

        if want_bid is not None and want_ask is not None and want_bid >= want_ask:
            # Both sides improved into the same price: keep the side that reduces inventory.
            if inventory > 0:
                want_bid = None
            else:
                want_ask = None
        return want_bid, want_ask


def strategy_a(tick=100, max_inventory=500):
    """A: quote at the best bid and ask, ignore inventory (except the hard limit)."""
    return QuotingStrategy("A: best bid/ask", tick=tick, theta=1.0, skew=0.0,
                           max_inventory=max_inventory)


def strategy_b(theta, skew, tick=100, max_inventory=500):
    """B: inventory skew around the mid (Part 1's reservation price, on a real book)."""
    return QuotingStrategy("B: inventory skew", tick=tick, theta=theta, skew=skew,
                           max_inventory=max_inventory)


def strategy_c(theta, skew, fair_adjust, tick=100, max_inventory=500):
    """C: inventory skew around the micro-price instead of the mid."""
    return QuotingStrategy("C: skew + micro-price", tick=tick, theta=theta, skew=skew,
                           max_inventory=max_inventory, fair_adjust=fair_adjust)
