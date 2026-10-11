"""
metrics.py - turn simulation results into numbers we can compare.

Two levels:
  path_metrics()  numbers for ONE simulated path (one price path, one strategy)
  summarize()     combines those numbers across MANY paths into one summary
"""

import numpy as np


def path_metrics(result):
    """Describe one SimulationResult with a few numbers."""
    final_pnl = result.pnl[-1]
    return {
        "final_pnl": final_pnl,
        "spread_pnl": result.spread_pnl,
        "inventory_pnl": final_pnl - result.spread_pnl,
        "n_fills": result.n_fills,
        # Biggest position held at any point, long or short.
        "max_abs_inventory": np.max(np.abs(result.inventory)),
        # Typical position size over the path. We average |inventory| rather than
        # inventory itself, because +10 and -10 would cancel out to a misleading 0.
        "avg_abs_inventory": np.mean(np.abs(result.inventory)),
    }


def summarize(path_stats):
    """
    Combine a list of path_metrics() dicts (one per path) into one summary dict.

    The "Sharpe-like" ratio is mean P&L divided by its standard deviation across
    paths: profit per unit of risk. It isn't a real (annualised) Sharpe ratio,
    just a quick way to compare strategies on the same scale.
    """
    final = np.array([s["final_pnl"] for s in path_stats])
    spread = np.array([s["spread_pnl"] for s in path_stats])
    inv_pnl = np.array([s["inventory_pnl"] for s in path_stats])
    fills = np.array([s["n_fills"] for s in path_stats])
    max_inv = np.array([s["max_abs_inventory"] for s in path_stats])
    avg_inv = np.array([s["avg_abs_inventory"] for s in path_stats])

    return {
        "Mean final P&L": final.mean(),
        "Std dev of final P&L": final.std(),
        "Worst final P&L": final.min(),
        "Sharpe-like (mean / std)": final.mean() / final.std(),
        "Mean spread P&L": spread.mean(),
        "Mean inventory P&L": inv_pnl.mean(),
        "Std dev of inventory P&L": inv_pnl.std(),
        "Avg fills per path": fills.mean(),
        "Avg max |inventory|": max_inv.mean(),
        "Worst max |inventory|": max_inv.max(),
        "Avg |inventory|": avg_inv.mean(),
    }


def format_table(summaries):
    """
    Build a printable comparison table.

    summaries: dict mapping strategy name -> summary dict from summarize().
    Rows are metrics, columns are strategies.
    """
    names = list(summaries)
    metric_names = list(summaries[names[0]])

    label_width = max(len(m) for m in metric_names) + 2
    col_width = max(len(n) for n in names) + 2

    lines = []
    header = "Metric".ljust(label_width) + "".join(n.rjust(col_width) for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for metric in metric_names:
        row = metric.ljust(label_width)
        for name in names:
            row += f"{summaries[name][metric]:.2f}".rjust(col_width)
        lines.append(row)
    return "\n".join(lines)
