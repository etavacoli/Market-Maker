"""
experiment.py - compare Strategy A (fixed spread) with Strategy B (inventory-aware).

Research question: does inventory-aware quoting reduce inventory risk without
sacrificing too much expected profit?

Run it from this folder:
    python experiment.py

What it prints:
  1. A comparison table of both strategies over N_PATHS simulated markets.
  2. Inventory risk up close: the single path where Strategy A did worst,
     replayed step by step for both strategies.
  3. A skew sweep: Strategy B with different skew values, to show the
     risk vs profit trade-off.
"""

import numpy as np

from market import price_action
from metrics import format_table, path_metrics, summarize
from simulator import simulate
from strategy import FixedSpreadMaker, InventoryAwareMaker

# ---- Settings ------------------------------------------------------------
# Market parameters follow the Avellaneda-Stoikov (2008) market-making paper:
# volatility 2 per unit of time, 200 steps of length dt = 0.005, and an order
# arrival rate of A = 140 per unit time with decay k = 1.5.

SEED = 42                       # master seed: same seed -> identical results every run
N_PATHS = 2000                  # simulated price paths per strategy
N_STEPS = 200                   # time steps per path
START_PRICE = 100.0             # mid price at the start of every path
DRIFT = 0.0                     # average price change per step (0 = no trend)
SIGMA = 2.0 * np.sqrt(0.005)    # std dev of the price change per step (~0.14)
BASE_FILL_PROB = 140 * 0.005    # fill probability for a quote right at the mid (0.7)
K = 1.5                         # how fast fill probability decays with distance
HALF_SPREAD = 0.50              # mid 100 -> bid 99.50, ask 100.50
SKEW = 0.05                     # Strategy B: shift quotes 0.05 per share held
SKEWS_TO_TRY = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def make_price_path(path_index):
    """The mid-price path for simulation number `path_index` (always the same for a given seed)."""
    rng = np.random.default_rng([SEED, path_index, 0])
    return price_action(DRIFT, SIGMA, START_PRICE, N_STEPS, rng=rng)


def run_on_path(strategy, prices, path_index):
    """Run one strategy on one price path."""
    # Every strategy gets the same fill seed on a given path, so they all face
    # the same price moves AND the same dice rolls. This is a fair comparison:
    # any difference in results is caused by the quotes, not by luck.
    fill_rng = np.random.default_rng([SEED, path_index, 1])
    return simulate(prices, strategy, fill_rng, BASE_FILL_PROB, K)


def run_experiment(strategies, n_paths):
    """Run every strategy on n_paths price paths. Returns {name: [path_metrics, ...]}."""
    stats = {s.name: [] for s in strategies}
    for i in range(n_paths):
        prices = make_price_path(i)
        for strategy in strategies:
            result = run_on_path(strategy, prices, i)
            stats[strategy.name].append(path_metrics(result))
    return stats


def show_worst_path(strategy_a, strategy_b, stats):
    """
    Stage 5, inventory risk up close: replay the path where Strategy A lost the
    most and show inventory and P&L over time for both strategies.
    """
    a_final = [s["final_pnl"] for s in stats[strategy_a.name]]
    worst = int(np.argmin(a_final))

    prices = make_price_path(worst)
    result_a = run_on_path(strategy_a, prices, worst)
    result_b = run_on_path(strategy_b, prices, worst)

    print(f"Path #{worst}: the path where Strategy A did worst")
    print(f"{'step':>5} {'mid':>8} {'A inv':>7} {'A P&L':>8} {'B inv':>7} {'B P&L':>8}")
    for t in range(0, N_STEPS + 1, 20):
        print(
            f"{t:>5} {prices[t]:>8.2f} "
            f"{result_a.inventory[t]:>7d} {result_a.pnl[t]:>8.2f} "
            f"{result_b.inventory[t]:>7d} {result_b.pnl[t]:>8.2f}"
        )

    print()
    print("Where the final P&L came from:")
    for label, result in [("A", result_a), ("B", result_b)]:
        final = result.pnl[-1]
        inventory_pnl = final - result.spread_pnl
        print(
            f"  {label}: spread P&L {result.spread_pnl:7.2f}  +  "
            f"inventory P&L {inventory_pnl:7.2f}  =  total {final:7.2f}"
        )


def skew_sweep():
    """Run Strategy B with several skew values to show the risk vs profit trade-off."""
    strategies = [InventoryAwareMaker(HALF_SPREAD, skew) for skew in SKEWS_TO_TRY]
    stats = run_experiment(strategies, N_PATHS)

    print(f"{'skew':>6} {'mean P&L':>10} {'std P&L':>9} {'worst P&L':>10} "
          f"{'Sharpe-like':>12} {'avg max|inv|':>13}")
    for strategy in strategies:
        s = summarize(stats[strategy.name])
        print(
            f"{strategy.skew:>6.2f} "
            f"{s['Mean final P&L']:>10.2f} "
            f"{s['Std dev of final P&L']:>9.2f} "
            f"{s['Worst final P&L']:>10.2f} "
            f"{s['Sharpe-like (mean / std)']:>12.2f} "
            f"{s['Avg max |inventory|']:>13.2f}"
        )
    print("(skew 0.00 is Strategy A, since with no skew B quotes exactly like A.)")


def main():
    strategy_a = FixedSpreadMaker(HALF_SPREAD)
    strategy_b = InventoryAwareMaker(HALF_SPREAD, SKEW)

    print("=" * 72)
    print(f"Market making simulator: {N_PATHS} price paths x {N_STEPS} steps, seed {SEED}")
    print(f"Start price {START_PRICE}, price-change std dev {SIGMA:.3f} per step, "
          f"half-spread {HALF_SPREAD}")
    print(f"Fill model: P(fill) = {BASE_FILL_PROB:.2f} * exp(-{K} * distance from mid)")
    print("=" * 72)

    stats = run_experiment([strategy_a, strategy_b], N_PATHS)
    summaries = {name: summarize(path_stats) for name, path_stats in stats.items()}

    print("\n1) Strategy comparison\n")
    print(format_table(summaries))

    print("\n2) Inventory risk up close\n")
    show_worst_path(strategy_a, strategy_b, stats)

    print("\n3) Skew sweep (Strategy B with different skews)\n")
    skew_sweep()


if __name__ == "__main__":
    main()
