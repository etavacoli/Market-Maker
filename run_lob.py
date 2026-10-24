"""
run_lob.py - Part 2: market making on a real day of Nasdaq order-book data.

    python run_lob.py --download   # first time only: fetch the data (~25 MB) into data/
    python run_lob.py              # run everything and write results/ (about 3 minutes)

The design, fixed before looking at any afternoon result:

  - 09:30-12:30 is the CALIBRATION window. For Strategies B and C we try every
    combination of theta (how far inside the spread to quote) and skew (how hard
    to lean against inventory) and keep the one with the best morning P&L.
  - 12:30-16:00 is the TEST window. Every headline number comes from here, using
    the morning's choices unchanged.
  - The micro-price (Strategy C) is fitted walk-forward over the whole day, only
    ever on data from earlier folds (see lob/signal.py).
  - 10 ms latency, 100-share orders, 500-share position limit, no fees or rebates
    in the P&L (a rebate column shows what a typical 2012 rebate would add).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lob import plots
from lob.analysis import HORIZONS, block_bootstrap_ci, interval_pnl, markouts, summarize
from lob.calibrate import fit_intensity
from lob.data import CLOSE, OPEN, PRICE_SCALE, SYMBOLS, download_samples, load_market
from lob.engine import run_backtest
from lob.signal import out_of_sample_r2, walk_forward
from lob.strategies import strategy_a, strategy_b, strategy_c

SPLIT = 45_000.0                 # 12:30:00
LATENCY = 0.010                  # seconds
HORIZON = 5.0                    # micro-price forecast horizon, seconds
THETAS = [1.0, 0.75, 0.5, 0.25]
SKEWS = [0.0, 0.5, 1.0, 2.0, 4.0]   # ticks of reservation shift per 100 shares held
REBATE_CENTS = 0.20              # illustrative maker rebate, cents per share
LATENCIES = [0.0001, 0.001, 0.01, 0.1, 1.0]


def dollars(x):
    return x / PRICE_SCALE


def main(data_dir="data", out_dir="results"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    started = time.time()

    signal_rows, intensity_rows, grid_rows, test_rows = [], [], [], []
    markout_rows, latency_rows, diff_rows, choice_rows = [], [], [], []
    curves, tables, intensity_fits = {}, {}, {}

    for symbol in SYMBOLS:
        m = load_market(symbol, data_dir)
        print(f"\n== {symbol}: {len(m):,} events, median spread "
              f"{np.median(m.spread) / m.tick:.0f} tick(s), median best-bid size {np.median(m.bid_size):,.0f} shares")

        # Fair value signal, fitted walk-forward.
        wf = walk_forward(m, horizon=HORIZON)
        afternoon = wf.sample_time >= SPLIT
        signal_rows.append({
            "symbol": symbol,
            "r2_weighted_mid": out_of_sample_r2(wf.label[afternoon], wf.pred_weighted[afternoon]),
            "r2_micro_price": out_of_sample_r2(wf.label[afternoon], wf.pred_micro[afternoon]),
        })
        tables[symbol] = (wf, m)

        # Fill-intensity model, A * exp(-k * delta).
        fit = fit_intensity(m)
        intensity_fits[symbol] = fit
        intensity_rows.append({"symbol": symbol, "market_orders": fit.n_orders, "A_per_sec": fit.A,
                               "k_per_dollar": fit.k, "r2_log": fit.r2, "grid_points": len(fit.delta)})

        # Calibration: morning only.
        makers = {
            "B": lambda th, sk: strategy_b(th, sk),
            "C": lambda th, sk: strategy_c(th, sk, wf.event_adjust),
        }
        best = {}
        for key, make in makers.items():
            for theta in THETAS:
                for skew in SKEWS:
                    r = run_backtest(m, make(theta, skew), OPEN, SPLIT, LATENCY)
                    pnl = dollars(r.pnl)
                    grid_rows.append({"symbol": symbol, "strategy": key, "theta": theta, "skew": skew,
                                      "morning_pnl": pnl, "morning_shares": r.shares_traded})
                    if key not in best or pnl > best[key][2]:
                        best[key] = (theta, skew, pnl)
            print(f"   {key}: chose theta={best[key][0]}, skew={best[key][1]} "
                  f"(morning P&L ${best[key][2]:,.0f})")
        a_morning = run_backtest(m, strategy_a(), OPEN, SPLIT, LATENCY)

        # Test: afternoon, once.
        strategies = {
            "A": strategy_a(),
            "B": strategy_b(best["B"][0], best["B"][1]),
            "C": strategy_c(best["C"][0], best["C"][1], wf.event_adjust),
        }
        results = {key: run_backtest(m, s, SPLIT, CLOSE, LATENCY) for key, s in strategies.items()}
        curves[symbol] = results

        for key, r in results.items():
            row = summarize(r, m, REBATE_CENTS)
            row["key"] = key
            row["theta"] = best[key][0] if key in best else 1.0
            row["skew"] = best[key][1] if key in best else 0.0
            test_rows.append(row)
            for h, value in zip(HORIZONS, markouts(r, m)):
                markout_rows.append({"symbol": symbol, "key": key, "horizon_s": h, "markout_cents": value})
            morning = best[key][2] if key in best else dollars(a_morning.pnl)
            choice_rows.append({"symbol": symbol, "key": key, "morning_pnl": morning,
                                "afternoon_pnl": dollars(r.pnl)})
            print(f"   test {key}: P&L ${row['pnl']:>9,.0f}  spread ${row['spread_capture']:>9,.0f}  "
                  f"inventory ${row['inventory_pnl']:>9,.0f}  shares {row['shares']:>7,}  "
                  f"markout 0s/5s/60s {row['markout_0s']:.2f}/{row['markout_5s']:.2f}/{row['markout_60s']:.2f}c")

        # Is B or C really better than A? Paired comparison on the same 5-minute intervals.
        base = interval_pnl(results["A"])
        for key in ("B", "C"):
            mean, lo, hi = block_bootstrap_ci(interval_pnl(results[key]) - base)
            diff_rows.append({"symbol": symbol, "comparison": f"{key} - A", "mean_5min_diff": mean,
                              "ci_low": lo, "ci_high": hi})

        # Latency: how much does speed matter?
        for key in ("A", "C"):
            for lat in LATENCIES:
                s = strategies[key] if key == "A" else strategy_c(best["C"][0], best["C"][1], wf.event_adjust)
                r = run_backtest(m, s, SPLIT, CLOSE, lat)
                latency_rows.append({"symbol": symbol, "key": key, "latency_s": lat,
                                     "pnl": dollars(r.pnl), "shares": r.shares_traded,
                                     "markout_5s": markouts(r, m)[3]})

    frames = {
        "signal": pd.DataFrame(signal_rows),
        "fill_intensity": pd.DataFrame(intensity_rows),
        "calibration_grid": pd.DataFrame(grid_rows),
        "test_summary": pd.DataFrame(test_rows),
        "markouts": pd.DataFrame(markout_rows),
        "paired_differences": pd.DataFrame(diff_rows),
        "in_vs_out_of_sample": pd.DataFrame(choice_rows),
        "latency": pd.DataFrame(latency_rows),
    }
    for name, df in frames.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")

    plots.make_all(frames, curves, tables, intensity_fits, out)
    print(f"\nWrote {out}/ in {time.time() - started:.0f}s")
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch the LOBSTER sample data first")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.download:
        download_samples(args.data)
    main(args.data, args.out)
