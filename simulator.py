"""
simulator.py - run one market maker over one price path.

Each time step goes like this:

  1. Look at the current mid price and our current inventory.
  2. Ask the strategy for a bid and an ask.
  3. Roll the dice for each side, independently:
       - does someone sell to us at our bid?
       - does someone buy from us at our ask?
     So in one step we can get 0, 1 or 2 fills (always 1 share each).
  4. Update cash and inventory for any fills:
       bid fill:  inventory += 1, cash -= bid   (we paid `bid` for a share)
       ask fill:  inventory -= 1, cash += ask   (we received `ask` for a share)
  5. The price moves to the next value on the path.
  6. Mark to market:  P&L = cash + inventory * new_mid
     i.e. "if we valued everything we hold at today's fair price, how much
     money have we made since the start?" (We start with 0 cash and 0 shares.)

Splitting the P&L into two parts helps explain where money comes from:

  - Spread P&L: on every fill, how far from the mid we traded. Buying at 99.50
    when the mid is 100 earns 0.50 of "edge" straight away. This is the market
    maker's payment for providing liquidity, and it is always >= 0 as long as
    our quotes stay on the right side of the mid.
  - Inventory P&L: everything else, i.e. gains and losses from the price moving
    while we hold shares. Holding +10 shares while the price drops 1.00 loses 10.

  Total P&L = spread P&L + inventory P&L  (exactly, not approximately).
"""

from dataclasses import dataclass

import numpy as np

from market import fill_probability


@dataclass
class SimulationResult:
    pnl: np.ndarray        # mark-to-market P&L after each step (pnl[0] = 0, the start)
    inventory: np.ndarray  # shares held after each step (inventory[0] = 0, the start)
    n_fills: int           # total number of trades (bid fills + ask fills)
    spread_pnl: float      # total edge earned from trading away from the mid (see above)


def simulate(prices, strategy, rng, base_fill_prob, k):
    """
    Run `strategy` over one price path.

    prices          list of mid prices, e.g. from price_action(); length = steps + 1
    strategy        an object with a quote(mid, inventory) -> (bid, ask) method
    rng             numpy random generator used for the fill dice rolls
    base_fill_prob  fill probability for a quote right at the mid
    k               how fast fill probability decays with distance from the mid
    """
    n_steps = len(prices) - 1

    cash = 0.0
    inventory = 0
    n_fills = 0
    spread_pnl = 0.0

    pnl_history = [0.0]
    inventory_history = [0]

    for t in range(n_steps):
        mid = prices[t]
        bid, ask = strategy.quote(mid, inventory)

        # How far each quote is from the mid. Positive = we're buying below /
        # selling above fair value, which is good for us.
        bid_delta = mid - bid
        ask_delta = ask - mid

        # Both dice are rolled every step, whatever the strategy does. That way
        # two strategies run on the same seed get the same random numbers in the
        # same order, so differences in results come from the strategy, not luck.
        bid_filled = rng.random() < fill_probability(bid_delta, base_fill_prob, k)
        ask_filled = rng.random() < fill_probability(ask_delta, base_fill_prob, k)

        if bid_filled:  # someone sold 1 share to us at our bid
            inventory += 1
            cash -= bid
            spread_pnl += bid_delta
            n_fills += 1

        if ask_filled:  # someone bought 1 share from us at our ask
            inventory -= 1
            cash += ask
            spread_pnl += ask_delta
            n_fills += 1

        # The market moves; value what we hold at the new mid.
        new_mid = prices[t + 1]
        pnl_history.append(cash + inventory * new_mid)
        inventory_history.append(inventory)

    return SimulationResult(
        pnl=np.array(pnl_history),
        inventory=np.array(inventory_history),
        n_fills=n_fills,
        spread_pnl=spread_pnl,
    )
