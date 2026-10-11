"""
strategy.py - the two quoting strategies we compare.

A strategy has one job: given the current mid price and how many shares we
hold (our inventory), decide where to put our bid (the price we'll buy at)
and our ask (the price we'll sell at).
"""


class FixedSpreadMaker:
    """
    Strategy A: the "dumb" market maker.

    Always quotes the same distance either side of the mid, no matter what it
    holds. With half_spread = 0.50 and mid = 100: bid 99.50, ask 100.50.

    Its weakness: it ignores inventory. If it happens to get filled on the bid
    several times in a row it just keeps buying, so it can drift into a large
    long (or short) position and is then exposed to the price moving against it.
    """

    name = "A: fixed spread"

    def __init__(self, half_spread=0.50):
        self.half_spread = half_spread

    def quote(self, mid, inventory):
        bid = mid - self.half_spread
        ask = mid + self.half_spread
        return bid, ask


class InventoryAwareMaker:
    """
    Strategy B: the inventory-aware market maker.

    Instead of centring its quotes on the mid, it centres them on a
    "reservation price": the price at which it would be happy to trade,
    given what it already holds.

        reservation = mid - skew * inventory

    - Long (inventory > 0): reservation is BELOW the mid, so both quotes shift
      down. The bid gets further from the mid (less likely to buy more) and the
      ask gets closer (more likely to sell). Inventory tends to shrink back to 0.
    - Short (inventory < 0): the opposite. Both quotes shift up, so it's more
      likely to buy back and less likely to sell more.
    - Flat (inventory = 0): identical to Strategy A.

    `skew` is how many dollars to shift per share held. skew = 0 turns this
    back into Strategy A; a bigger skew pulls inventory back to 0 harder.
    """

    def __init__(self, half_spread=0.50, skew=0.05):
        self.half_spread = half_spread
        self.skew = skew
        self.name = f"B: inventory-aware (skew {skew})"

    def quote(self, mid, inventory):
        reservation = mid - self.skew * inventory
        bid = reservation - self.half_spread
        ask = reservation + self.half_spread
        return bid, ask
