"""
market.py - the simulated market.

Two pieces live here:

  1. price_action()      How the "true" mid price moves over time (a random walk).
                         This is Stage 1, written by Eamon. It is copied here
                         unchanged except for one optional `rng` argument, so
                         experiments can be made reproducible.

  2. fill_probability()  How likely it is that some other trader accepts one of
                         our quotes, depending on how far that quote is from the mid.
"""

import math

import numpy as np


def price_action(mean: float, standard_dev: float, fair_price, iterations: int, rng=None):


    # need to save price at each timestep, take inventory of price
    #formula for price after timestep: P(t+1) = P(t) + (randomely generated price change calculated from N ~ (0, standard_dev)

    prices = [fair_price]
    price_changes = []
    # [Added] Use the caller's random number generator if one is passed in, so a
    # fixed seed gives the same path every run. With rng=None this line does
    # exactly what the original did: a fresh, unseeded generator.
    random_price = np.random.default_rng() if rng is None else rng
    samples = random_price.normal(loc=mean, scale=standard_dev, size=iterations)
    for sample in samples:
        price_changes.append(sample)
        prices.append(fair_price + sample)
        fair_price = fair_price + sample



    return prices


def fill_probability(delta, base_prob, k):
    """
    Chance that a quote sitting `delta` dollars away from the mid gets filled
    during one time step:

        P(fill) = base_prob * e^(-k * delta)

    What this means in plain language:
      - Traders who want to trade right now are much more willing to accept a
        price close to fair value than one far away. If the mid is 100, a bid
        at 99.99 is a great deal for a seller; a bid at 95 is a terrible one.
      - So the further our quote is from the mid (bigger delta), the less likely
        anyone takes it. The exponential makes the drop-off smooth: every extra
        dollar of distance multiplies the fill chance by the same factor e^(-k).
        With k = 1.5, each extra 10 cents of distance cuts the chance by ~14%.
      - base_prob is the fill chance for a quote sitting right at the mid (delta = 0).
      - k measures how price-sensitive the other traders are. Big k: fills dry up
        quickly as we quote wider. Small k: we can quote wide and still get filled.

    Why it matters: this is the market maker's core trade-off. A wide quote earns
    more per trade (buy cheaper, sell dearer) but trades less often. A tight quote
    trades often but earns little each time.

    If delta is negative (our quote is on the "wrong" side of the mid, e.g. we bid
    above fair value), the formula gives more than base_prob. We cap it at 1
    because a probability can't be more than 100%.
    """
    return min(1.0, base_prob * math.exp(-k * delta))
