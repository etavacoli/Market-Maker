"""
lob - Part 2 of the project: market making on real Nasdaq order-book data.

Part 1 (the files in the repo root) simulates a market with random numbers.
Part 2 replays one real trading day, message by message, and asks how the same
quoting ideas hold up once fills, queues and other traders are real.

Modules, in the order data flows through them:

    data.py       load LOBSTER order-book messages for one stock-day
    signal.py     order-book imbalance -> short-term fair value (micro-price),
                  fitted walk-forward so it never sees the future
    strategies.py the three quoting strategies (A, B, C)
    engine.py     the backtester: our orders, queue position, latency, fills
    calibrate.py  fit the fill-intensity model A * exp(-k * delta) to real trades
    analysis.py   markouts (adverse selection), P&L decomposition, bootstrap CIs
    plots.py      the charts in results/
"""
