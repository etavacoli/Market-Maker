"""
plots.py - the charts in results/.

Colors come from a palette checked for color-blind safety, and every line is
also labelled directly (with a leader line when labels had to be nudged apart),
so no chart relies on color alone.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from .analysis import HORIZONS  # noqa: E402
from .data import PRICE_SCALE, SYMBOLS  # noqa: E402
from .signal import IMBALANCE_BINS, SPREAD_EDGES  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STOCK_COLOR = dict(zip(SYMBOLS, SERIES))
STRATEGY_COLOR = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
STRATEGY_LABEL = {"A": "A: best bid/ask", "B": "B: inventory skew", "C": "C: skew + micro-price"}
DPI = 200

dollar_format = FuncFormatter(lambda v, _: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")


def _figure(width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    return fig, axes


def _titles(fig, title, subtitle):
    fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=8.8, color=INK2, linespacing=1.45)


def _end_labels(ax, items, min_gap=0.06, x_offset=12):
    """Label each line at its right end. Labels are nudged apart so they never
    overlap, and a thin leader line in the series color connects each label to
    its line. Call this AFTER the figure layout is final.
    items: (x, y, text, color) in data coordinates."""
    if not items:
        return
    ax.set_xlim(ax.get_xlim())  # freeze limits so positions can't shift later
    ax.set_ylim(ax.get_ylim())
    to_axes = ax.transData + ax.transAxes.inverted()
    pts = sorted(([to_axes.transform((x, y))[1], x, y, text, color] for x, y, text, color in items),
                 key=lambda p: p[0])
    for k in range(1, len(pts)):
        pts[k][0] = max(pts[k][0], pts[k - 1][0] + min_gap)
    if pts[-1][0] > 0.99:
        pts[-1][0] = 0.99
        for k in range(len(pts) - 2, -1, -1):
            pts[k][0] = min(pts[k][0], pts[k + 1][0] - min_gap)
    axes_height_pt = ax.bbox.height * 72 / ax.figure.dpi
    for label_y, x, y, text, color in pts:
        true_y = to_axes.transform((x, y))[1]
        dy = (label_y - true_y) * axes_height_pt
        ax.plot([x], [y], marker="o", ms=4.5, color=color, mec=SURFACE, mew=1.2, zorder=5, clip_on=False)
        ax.annotate(text, xy=(x, y), xytext=(x_offset, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=1, shrinkB=3,
                                    relpos=(0, 0.5)))  # attach to the label's left edge, not its middle


def plot_markouts(markouts, out):
    labels = ["at fill", "0.1 s", "1 s", "5 s", "10 s", "30 s", "1 min", "5 min"]
    fig, ax = _figure(8.4, 5.0)
    _titles(fig, "The edge from each fill is gone within seconds",
            "Strategy A (quote at the best bid and ask), out-of-sample afternoon, 21 June 2012. "
            "A markout is the profit per share if we\nclosed each fill at the mid a little later. "
            "Below 0, the price moved against us by more than the spread we earned.")
    fig.subplots_adjust(left=0.09, right=0.89, top=0.8, bottom=0.12)
    x = np.arange(len(HORIZONS))
    ends, lowest = [], 0
    for symbol in SYMBOLS:
        d = markouts[(markouts.symbol == symbol) & (markouts.key == "A")].sort_values("horizon_s")
        y = 100 * d.markout_cents.to_numpy() / d.markout_cents.iloc[0]
        lowest = min(lowest, y.min())
        ax.plot(x, y, color=STOCK_COLOR[symbol], lw=2, zorder=3)
        ends.append((x[-1], y[-1], symbol, STOCK_COLOR[symbol]))
    ax.axhline(0, color=INK2, lw=1, zorder=2)
    ax.axhline(100, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(0.08, 103, "edge earned at the fill (about half the spread)", fontsize=8, color=INK2, va="bottom")
    ax.text(4.1, 3, "0 = break-even", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, len(x) - 0.75)
    ax.set_ylim(np.floor((lowest - 12) / 25) * 25, 118)
    ax.set_xlabel("Time after our order was filled", color=INK2, fontsize=9.5)
    ax.set_ylabel("Markout, % of the edge earned at the fill", color=INK2, fontsize=9.5)
    _end_labels(ax, ends)
    fig.savefig(out / "markouts.png", facecolor=SURFACE)
    plt.close(fig)


def plot_cumulative_pnl(curves, out):
    fig, axes = _figure(13.5, 4.0, ncols=len(SYMBOLS))
    _titles(fig, "Afternoon P&L by strategy, out of sample",
            "Mark-to-market P&L from 12:30 to 16:00, no fees or rebates. B and C use settings chosen on "
            "the morning only. Each panel has its own dollar scale.\nOn AAPL the morning picked settings "
            "for B that are identical to A, so those two lines coincide.")
    handles = [plt.Line2D([], [], color=STRATEGY_COLOR[k], lw=2.2) for k in ("A", "B", "C")]
    fig.legend(handles, [STRATEGY_LABEL[k] for k in ("A", "B", "C")], loc="upper right",
               ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.995, 0.995))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.73, bottom=0.1, wspace=0.45)
    for ax, symbol in zip(axes, SYMBOLS):
        ends = []
        for key in ("B", "C", "A"):
            r = curves[symbol][key]
            hours = 12.5 + (r.snap_time - r.start) / 3600
            pnl = r.snap_pnl / PRICE_SCALE
            ax.plot(hours, pnl, color=STRATEGY_COLOR[key], lw=1.6, zorder=3)
            ends.append((hours[-1], pnl[-1], key, STRATEGY_COLOR[key]))
        ax.axhline(0, color=INK2, lw=0.9, zorder=2)
        ax.set_title(symbol, loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_xticks([13, 14, 15, 16], ["1pm", "2pm", "3pm", "4pm"])
        ax.yaxis.set_major_formatter(dollar_format)
        _end_labels(ax, ends, min_gap=0.08, x_offset=8)
    fig.savefig(out / "cumulative_pnl.png", facecolor=SURFACE)
    plt.close(fig)


def plot_latency(latency, out):
    fig, ax = _figure(8.4, 5.0)
    _titles(fig, "On wide-spread stocks, speed decides who gets picked off",
            "Strategy A, out-of-sample afternoon, identical rules at different latencies. A slow quote is "
            "still in the book after the\nprice has moved, so it gets filled at a stale price. "
            "On the one-tick stocks (INTC, MSFT), queue position matters more than speed.")
    fig.subplots_adjust(left=0.11, right=0.89, top=0.8, bottom=0.12)
    ends = []
    for symbol in SYMBOLS:
        d = latency[(latency.symbol == symbol) & (latency.key == "A")].sort_values("latency_s")
        ax.plot(d.latency_s, d.pnl, color=STOCK_COLOR[symbol], lw=2, zorder=3)
        ax.plot(d.latency_s[:-1], d.pnl[:-1], "o", color=STOCK_COLOR[symbol], ms=4, mec=SURFACE, mew=1, zorder=4)
        ends.append((d.latency_s.iloc[-1], d.pnl.iloc[-1], symbol, STOCK_COLOR[symbol]))
    ax.axhline(0, color=INK2, lw=1, zorder=2)
    ax.axvline(0.01, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1], ["0.1 ms", "1 ms", "10 ms", "100 ms", "1 s"])
    ax.minorticks_off()
    ax.set_xlim(0.00007, 1.4)
    ax.text(0.0108, ax.get_ylim()[1], "default used in the other results", fontsize=8, color=INK2, va="top")
    ax.yaxis.set_major_formatter(dollar_format)
    ax.set_xlabel("Latency: delay between deciding and the order reaching the exchange", color=INK2, fontsize=9.5)
    ax.set_ylabel("Afternoon P&L", color=INK2, fontsize=9.5)
    _end_labels(ax, ends)
    fig.savefig(out / "latency.png", facecolor=SURFACE)
    plt.close(fig)


def micro_price_curves(tables):
    """Learned adjustment vs imbalance from the last fold, at each stock's most common spread,
    as a percentage of that spread."""
    centers = -1 + (np.arange(IMBALANCE_BINS) + 0.5) * 2 / IMBALANCE_BINS
    curves = {}
    for symbol, (wf, m) in tables.items():
        spread_bins = np.digitize(m.spread / m.tick, SPREAD_EDGES)
        spread_bin = np.bincount(spread_bins).argmax()
        typical_spread = np.median(m.spread[spread_bins == spread_bin])
        g = wf.tables[-1][spread_bin * IMBALANCE_BINS:(spread_bin + 1) * IMBALANCE_BINS]
        curves[symbol] = (centers, 100 * g / typical_spread, typical_spread / m.tick)
    return curves


def plot_micro_price(curves, r2, out):
    fig, ax = _figure(8.4, 5.0)
    _titles(fig, "Imbalance predicts the next move only when the spread is one tick",
            "Learned micro-price adjustment at each stock's typical spread, fitted walk-forward on data before 15:30. "
            "On INTC and MSFT\na heavy bid queue means the price is about to rise; on the wide-spread stocks "
            "the curve is noise around zero.\nR\u00b2 = out-of-sample forecast skill on the afternoon "
            "(0% = no better than the plain mid).")
    fig.subplots_adjust(left=0.1, right=0.72, top=0.76, bottom=0.12)
    ends = []
    for symbol in SYMBOLS:
        x, g, spread_ticks = curves[symbol]
        ax.plot(x, g, color=STOCK_COLOR[symbol], lw=2, zorder=3)
        skill = f"{100 * r2[symbol]:+.1f}%".replace("-0.0%", "0.0%").replace("+0.0%", "0.0%")
        label = f"{symbol}: {spread_ticks:.0f}-tick spread, R² {skill}"
        ends.append((x[-1], g[-1], label, STOCK_COLOR[symbol]))
    ax.axhline(0, color=INK2, lw=1, zorder=2)
    ax.set_xlim(-1, 0.92)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    ax.set_xlabel("Order-book imbalance = (bid size - ask size) / (bid size + ask size)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Expected change in mid over next 5 s, % of spread", color=INK2, fontsize=9.5)
    _end_labels(ax, ends, min_gap=0.065, x_offset=24)
    fig.savefig(out / "micro_price.png", facecolor=SURFACE)
    plt.close(fig)


def plot_fill_intensity(fits, out):
    fig, axes = _figure(13.5, 4.0, ncols=len(SYMBOLS))
    _titles(fig, "Part 1's fill model, measured: rate = A · exp(-k · distance)",
            "Dots: how often a market order reached at least that far from the mid. Line: least-squares fit "
            "of the exponential. It fits the wide-spread stocks;\non INTC and MSFT almost every market order "
            "stops at the best price, so there is almost nothing to fit.")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.15, wspace=0.32)
    for ax, symbol in zip(axes, SYMBOLS):
        f = fits[symbol]
        cents = f.delta * 100
        ax.plot(cents, f.rate, "o", ms=3.5, color=SERIES[0], zorder=3)
        line = np.linspace(cents.min(), cents.max(), 50)
        ax.plot(line, f.A * np.exp(-f.k * line / 100), color=INK2, lw=1.4, zorder=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(symbol, loc="left", fontsize=10.5, fontweight="bold", color=INK)
        r2 = "n/a (2 points)" if np.isnan(f.r2) else f"{f.r2:.2f}"
        ax.text(0.97, 0.95, f"k = {f.k:,.0f} per $\nfit R² = {r2}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.3, color=INK2)
        ax.set_xlabel("Distance from mid (cents)", color=INK2, fontsize=8.8)
    axes[0].set_ylabel("Market orders reaching it, per second", color=INK2, fontsize=8.8)
    fig.savefig(out / "fill_intensity.png", facecolor=SURFACE)
    plt.close(fig)


def make_all(frames, curves, tables, intensity_fits, out):
    r2 = dict(zip(frames["signal"].symbol, frames["signal"].r2_micro_price))
    plot_markouts(frames["markouts"], out)
    plot_cumulative_pnl(curves, out)
    plot_latency(frames["latency"], out)
    plot_micro_price(micro_price_curves(tables), r2, out)
    plot_fill_intensity(intensity_fits, out)
