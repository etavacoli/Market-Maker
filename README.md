# Market-Making Simulator

A small Python simulator of a market maker: someone who constantly offers to buy
slightly below fair value and sell slightly above it, earning the gap. It
simulates a random-walk price, quotes bids and asks around it, fills those quotes
at random (more often when they're attractive), and tracks cash, inventory and
mark-to-market P&L. It then compares two quoting strategies over thousands of
simulated markets.

**Research question:** does inventory-aware quoting reduce inventory risk without
sacrificing too much expected profit?

**Short answer (details below):** yes. In this model it cut the standard deviation
of P&L by 55% and the largest typical position by more than half, while giving up
only about 1.7% of average profit.

## How to run

The project uses a local virtual environment (`.venv`, one folder up), so NumPy
is installed only for this project.

```bash
cd ~/python_quant
source .venv/bin/activate          # once per terminal session
cd market_maker
python experiment.py
```

It takes a few seconds. The seed is fixed (`SEED = 42` in `experiment.py`), so
you get exactly the same numbers every time. Change the seed to see a different
set of random markets.

To set the project up from scratch somewhere else:
`python3 -m venv .venv && .venv/bin/pip install -r market_maker/requirements.txt`.

## Files

| File | What it does |
|---|---|
| `market.py` | The market. `price_action()` (your Stage 1 random walk) moves the mid price; `fill_probability()` decides how likely a quote is to get filled. |
| `strategy.py` | The two quoting strategies: `FixedSpreadMaker` (A) and `InventoryAwareMaker` (B). |
| `simulator.py` | Runs one strategy over one price path, step by step, tracking cash, inventory and P&L. |
| `metrics.py` | Turns simulation results into numbers (mean P&L, worst loss, max inventory, ...) and prints the comparison table. |
| `experiment.py` | All the settings, plus the experiment itself: thousands of paths per strategy, the worst-path replay and the skew sweep. |

`price_action()` is copied from `~/market making simulator/price_movement.py`
unchanged except for one optional `rng` argument (marked `[Added]` in the code).
Passing a seeded generator makes the price path reproducible. Leaving it out
(`rng=None`) behaves exactly like the original.

---

## The concepts, in plain language

### 1. Random walk (the mid price)

The "mid" is the market's current idea of fair value. Each step, it moves by a
random amount:

    P(t+1) = P(t) + eps,    eps ~ Normal(0, sigma^2)

Each move is independent of the last one, so knowing the past doesn't help you
predict the next move. On average the price goes nowhere (mean 0), but over time
it can wander far from where it started. Here sigma is about 0.14 per step, so a
200-step path typically ends within about +/-2 of 100, though some go further.

### 2. Bid, ask and spread

A market maker posts two prices at once:

- **Bid**: the price it will buy at (below the mid).
- **Ask**: the price it will sell at (above the mid).
- **Spread**: the gap between them. **Half-spread** is the distance from the mid to either side.

With mid = 100 and half-spread 0.50: bid 99.50, ask 100.50. If someone sells to
us at 99.50 and someone else buys from us at 100.50, we've made 1.00 without
having any opinion on where the price is going. That's the business.

### 3. Fill probability: why attractive quotes fill more

Nobody is forced to trade with us. A seller would happily take a bid of 99.99 when
fair value is 100, but a bid of 95 is a bad deal. So the further our quote is from
the mid, the less likely anyone takes it. We model that as:

    P(fill in one step) = A * e^(-k * delta)

- **delta** is how far the quote is from the mid (0.50 for our default quotes).
- **A** is the fill chance for a quote right at the mid (0.7 here).
- **k** is how price-sensitive other traders are (1.5 here). Each extra dollar of
  distance multiplies the fill chance by e^(-k). With k = 1.5, each extra 10 cents
  cuts it by about 14%.

With delta = 0.50, P(fill) = 0.7 x e^(-0.75) ≈ 0.33 per side per step.

This captures the market maker's core trade-off: **wide quotes earn more per trade
but trade less; tight quotes trade more but earn less each time.** The bid and
the ask are checked separately each step, so a step can have 0, 1 or 2 fills.

### 4. Inventory and cash

**Inventory** is how many shares we hold: positive means long, negative means short.
**Cash** is money in or out. Each fill is one share:

- **Bid fill** (someone sells to us): `inventory += 1`, `cash -= bid`
- **Ask fill** (someone buys from us): `inventory -= 1`, `cash += ask`

We start with 0 cash and 0 shares. Cash goes negative after a buy; that's fine,
because we own the share instead.

### 5. Mark-to-market P&L

Cash alone is misleading: after buying a share our cash is -99.50, but we own
something worth about 100. **Mark-to-market** values everything we hold at the
current mid:

    P&L = cash + inventory * mid

Worked example:

1. Mid 100, our bid at 99.50 fills: inventory 1, cash -99.50, P&L = -99.50 + 1 x 100 = **+0.50**.
2. If our ask at 100.50 fills next (mid still 100): inventory 0, cash +1.00, P&L = **+1.00**.
3. If instead the price drops to 98 before we sell: P&L = -99.50 + 1 x 98 = **-1.50**.

The simulator also splits P&L into two parts, which is the key to understanding
the results:

- **Spread P&L**: the edge earned on each fill, measured from the mid at the time of
  the trade (buying at 99.50 when the mid is 100 earns 0.50). This is what the
  market maker is paid for providing liquidity.
- **Inventory P&L**: everything else, i.e. gains or losses from the price moving
  while we hold a position.

**Total P&L = spread P&L + inventory P&L**, exactly.

### 6. Inventory risk

Fills are random, so a fixed-spread maker can get hit on the bid several times in a
row and end up holding +20 shares without choosing to. Now every 1.00 drop in the
price costs 20. In this model the price has no trend, so on average inventory P&L
is about zero. **Inventory adds risk but no expected return.** It just makes results
noisier and occasionally very bad. That's inventory risk: exposure to directional
moves you never meant to bet on.

### 7. Reservation price (the inventory-aware fix)

Strategy B centres its quotes on a **reservation price** instead of the mid. The
reservation price is the price it would be happy to trade at, given what it already
holds:

    reservation = mid - skew * inventory
    bid = reservation - half_spread
    ask = reservation + half_spread

- **Long** (inventory > 0): both quotes shift down. The bid moves further from the mid,
  so buying more is less likely. The ask moves closer, so selling is more likely.
  Inventory drifts back toward 0.
- **Short**: both quotes shift up, so it's more likely to buy back.
- **Flat**: identical to Strategy A.

With skew = 0.05, holding +10 shares moves both quotes down by 0.50: bid 99.00,
ask 100.00 when the mid is 100. In effect B is saying "I'll give up a little edge
to get rid of a position I don't want."

---

## Experiment setup

| Setting | Value | Meaning |
|---|---|---|
| Paths | 2,000 per strategy | independent simulated markets |
| Steps per path | 200 | |
| Start price | 100 | |
| sigma per step | ~0.141 | 2 x sqrt(0.005) |
| A (fill prob at mid) | 0.7 | 140 x 0.005 |
| k | 1.5 | |
| Half-spread | 0.50 | both strategies |
| Skew (Strategy B) | 0.05 per share | |
| Seed | 42 | |

The market parameters (sigma = 2 per unit time, dt = 0.005, A = 140, k = 1.5) are
taken from Avellaneda & Stoikov (2008), *High-frequency trading in a limit order
book*, a classic market-making paper. That makes the setup comparable to a known
benchmark.

**Fair comparison:** both strategies see the **same 2,000 price paths and the same
random numbers for fills**. Any difference in results comes from how they quote,
not from one getting luckier. (Check: with skew = 0, Strategy B's numbers match
Strategy A's exactly.)

## Results

### 1) Strategy comparison (2,000 paths each)

| Metric | A: fixed spread | B: inventory-aware (skew 0.05) |
|---|---:|---:|
| Mean final P&L | 65.90 | 64.78 |
| Std dev of final P&L | 13.82 | **6.22** |
| Worst final P&L | -5.25 | **43.97** |
| Sharpe-like (mean / std) | 4.77 | **10.41** |
| Mean spread P&L | 66.19 | 64.83 |
| Mean inventory P&L | -0.29 | -0.05 |
| Std dev of inventory P&L | 12.95 | **4.12** |
| Avg fills per path | 132.39 | 133.94 |
| Avg max \|inventory\| | 11.20 | **5.01** |
| Worst max \|inventory\| | 34 | **9** |
| Avg \|inventory\| | 4.95 | 1.59 |

("Avg |inventory|" averages the size of the position over each path. It uses
absolute values because +10 and -10 would otherwise cancel to a misleading 0.)

What it shows:

- **Almost all of the profit comes from the spread**, for both strategies (about 65
  out of about 65). Mean inventory P&L is about 0, as expected with a trendless price.
- **Almost all of the risk comes from inventory.** For A, the inventory P&L std dev
  (12.95) is nearly the whole P&L std dev (13.82).
- **B cuts that risk sharply:** P&L std dev down 55%, average largest position down
  from 11 to 5 shares, worst position from 34 to 9.
- **The cost is small:** mean P&L falls by 1.12 (1.7%). Skewing makes one side tighter
  than 0.50 at times, so those fills earn less edge. B actually trades slightly
  more, but earns a little less per trade.
- **Tail outcomes improve the most:** A's worst path lost money (-5.25). B's worst
  path still made 43.97.

### 2) Inventory risk up close: Strategy A's worst path

On path #65 the price slid from 100 to about 95.4. Strategy A kept buying on the
way down and built up as many as +24 shares; Strategy B never held more than 6.
(The script prints every 20th step; a selection is shown here.)

| step | mid | A inventory | A P&L | B inventory | B P&L |
|---:|---:|---:|---:|---:|---:|
| 0 | 100.00 | 0 | 0.00 | 0 | 0.00 |
| 40 | 100.37 | 9 | 14.41 | 3 | 14.10 |
| 100 | 98.55 | 12 | 21.25 | 4 | 32.42 |
| 140 | 97.24 | 21 | 16.64 | 4 | 42.88 |
| 160 | 95.82 | 20 | -7.22 | 0 | 46.74 |
| 200 | 95.36 | 21 | -5.25 | -1 | 57.10 |

    A: spread P&L 69.50  +  inventory P&L -74.75  =  total -5.25
    B: spread P&L 69.90  +  inventory P&L -12.80  =  total 57.10

Both earned about the same from the spread. A gave all of it back, and more, by
holding a large long position into a falling market. That's inventory risk in one picture.

### 3) Skew sweep: how much skew?

| skew | mean P&L | std P&L | worst P&L | Sharpe-like | avg max \|inv\| |
|---:|---:|---:|---:|---:|---:|
| 0.00 (= A) | 65.90 | 13.82 | -5.25 | 4.77 | 11.20 |
| 0.01 | 65.88 | 9.31 | 30.03 | 7.08 | 8.06 |
| 0.02 | 65.61 | 7.75 | 37.15 | 8.47 | 6.74 |
| 0.05 | 64.78 | 6.22 | 43.97 | 10.41 | 5.01 |
| 0.10 | 63.39 | 5.43 | 44.58 | 11.68 | 3.85 |
| 0.20 | 60.23 | 4.86 | 44.33 | 12.39 | 2.88 |

- **The first bit of skew is almost free:** 0.01 cuts std dev by a third and costs 0.02 of mean P&L.
- **Returns diminish:** going from 0.10 to 0.20 costs 3.16 of mean P&L but only reduces std dev by 0.57.
- With large skews the maker sometimes quotes on the wrong side of the mid. For
  example, with skew 0.20 and +3 shares, the ask is 0.10 *below* the mid. It is
  effectively paying to get flat.
- Which skew is "best" depends on how much you dislike risk. Sharpe-like keeps
  rising here, but mean profit keeps falling.

### Answer to the research question

**Yes.** In this model, inventory-aware quoting dramatically reduces inventory
risk for a small cost in expected profit. With skew 0.05: about half the P&L
volatility, less than half the typical position, no losing paths, and only 1.7%
less average profit. The reason is that inventory risk is *uncompensated* here:
holding a position adds variance but no expected return. So getting rid of it is
cheap, and you only pay a little edge to do so.

## Limitations and possible next steps

- **No adverse selection.** Fills here are independent of where the price goes next.
  In real markets, the trader hitting your bid often knows something (they sell
  just before a drop). That makes inventory more dangerous and spreads more important.
- **Constant skew.** Avellaneda-Stoikov uses `reservation = mid - q * gamma * sigma^2 * (T - t)`.
  The skew depends on risk aversion (gamma), volatility, and time left, and the paper
  also derives an optimal spread. That's the natural next upgrade.
- **Fixed half-spread.** Both strategies always quote 0.50. Optimising the spread
  (e.g. widening when volatility is high) is another lever.
- **No hard inventory limits, fees, latency, or order sizes beyond 1 share.**
- **Charts.** Plotting inventory and P&L paths and a P&L histogram per strategy
  (e.g. with matplotlib) would make the comparison easier to see.
