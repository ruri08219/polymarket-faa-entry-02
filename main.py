#!/usr/bin/env python3
"""Minute-by-minute quantile forecasts for Polymarket Yes prices.

Each run handles one case file: a checkpoint plus the one-minute Yes
prices of every market of a single family for the 72 hours leading up
to it. The run writes one forecast file: for every market, and for
each of the 240 minutes after the checkpoint, five quantiles of that
minute's Yes price -- the 5%, 25%, 50%, 75% and 95% levels.

The forecasting idea
--------------------
A prediction-market Yes price behaves like a sticky random walk. Most
minutes it does not move at all. Then it ticks a little. Every so
often it jumps a long way on news. That shape fixes the whole design:

* The centre of every forecast row is the last price seen. The
  market's best guess about the next minute is, to first order, where
  it already stands; chasing the last few minutes of drift only adds
  noise, so the median is persistence, not momentum.
* The spread around that centre is a random-walk cone. Uncertainty
  compounds the further ahead you look, so the cone widens with the
  square root of the horizon.
* The cone's width depends on where the price sits. A market hovering
  near one half can swing much further than one pinned near a few
  cents, so the per-minute spread is scaled by the square root of
  p * (1 - p), measured against a fixed reference level.
* The two outer quantiles are stretched by half again. Real jumps are
  heavier tailed than a Gaussian cone admits, so the 5% and 95% bands
  are widened to pay less when a jump lands outside them.

The program is deterministic: the same case file always yields the
same forecast file, byte for byte. It uses only the standard library
and never touches the network.
"""

import json
import math
import os
import sys


# Quantile levels, in the order the forecast file requires them.
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)

# Points of the standard normal distribution at those levels, written
# out to full double precision.
NORMAL_POINTS = (
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
)

# How much wider the two outer quantiles are than the cone alone would
# make them, in the same order as LEVELS.
TAIL_FACTORS = (1.5, 1.0, 1.0, 1.0, 1.5)

# Minutes ahead that every forecast row covers.
LOOKAHEAD_MINUTES = 240

# Per-minute spread of the cone at the reference price level, in price
# units per square-root minute.
REFERENCE_SPREAD = 0.0012

# Reference value of p * (1 - p), taken where p is 0.05 (the same
# value where p is 0.95). The level factor is the square root of
# p * (1 - p) over this, so the spread is exactly REFERENCE_SPREAD at
# those two price levels and grows towards the middle of the range.
REFERENCE_VARIANCE = 0.0475


def usable_prices(raw_series):
    """Keep only the finite numbers of a price series, as floats.

    A series can carry gaps or glitches -- a null, or a value that is
    not finite. Those say nothing about where the market stands, so
    they are dropped before anything else touches the series. Boolean
    values are not prices either, so they are dropped too.
    """
    cleaned = []
    for entry in raw_series:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, (int, float)) and math.isfinite(entry):
            cleaned.append(float(entry))
    return cleaned


def level_spread(anchor):
    """Per-minute spread of the cone for a market sitting at `anchor`.

    The spread grows with the price level's own variance: a market
    near one half moves further per minute than one pinned near the
    edges, so the reference spread is scaled by the square root of
    p * (1 - p) over the reference variance. At an exact edge the
    variance is zero and the cone would collapse to a point, which is
    never an honest forecast, so the variance is floored at a small
    positive value there.
    """
    variance = anchor * (1.0 - anchor)
    if variance <= 0.0:
        variance = 0.0025
    factor = math.sqrt(variance / REFERENCE_VARIANCE)
    return REFERENCE_SPREAD * factor


def quantile_row(anchor, spread, minute):
    """One forecast row: the five quantiles `minute` minutes ahead.

    The centre is the anchor. Each level sits `point * factor *
    spread * sqrt(minute)` away from it, where `point` is the normal
    quantile and `factor` the tail widening for that level. The row is
    clamped into [0, 1] and rounded to six decimals; clamping keeps
    the row ordered because every level moves the same centre by a
    monotone amount.
    """
    root = math.sqrt(minute)
    row = []
    for point, factor in zip(NORMAL_POINTS, TAIL_FACTORS):
        shifted = anchor + point * factor * spread * root
        bounded = min(1.0, max(0.0, shifted))
        row.append(round(bounded, 6))
    return row


def market_forecast(market, horizon):
    """The full forecast for one market: rows of five quantiles.

    The anchor is the market's last usable price; a market with no
    usable price at all is forecast around one half, the neutral
    level, with the reference spread.
    """
    series = usable_prices(market.get("prices") or [])
    anchor = series[-1] if series else 0.5
    # A stray price outside [0, 1] is bad data, not a market view: pin
    # the anchor to the valid range before it feeds the level factor,
    # so one corrupt tick cannot collapse or blow up the cone.
    anchor = min(1.0, max(0.0, anchor))
    spread = level_spread(anchor)
    rows = []
    for minute in range(1, horizon + 1):
        rows.append(quantile_row(anchor, spread, minute))
    return {"id": market["id"], "q": rows}


def read_case(path):
    """Load one case file and return its parsed JSON object."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_forecast(case, output_path):
    """Write the forecast file for one case.

    The output holds exactly one key, `forecasts`, with one entry per
    market of the case, each entry holding that market's id and its
    rows of five quantiles.
    """
    horizon = case.get("horizon_minutes") or LOOKAHEAD_MINUTES
    forecasts = []
    for market in case.get("markets") or []:
        forecasts.append(market_forecast(market, horizon))
    payload = {"forecasts": forecasts}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")


def input_directory():
    """Where case files are read from."""
    return os.environ.get("MAKEFUN_INPUT_DIR") or "/input"


def output_directory():
    """Where forecast files are written to."""
    return os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"


def main(argv):
    """Forecast every case file found in the input directory."""
    source = input_directory()
    target = output_directory()
    os.makedirs(target, exist_ok=True)
    names = sorted(
        name for name in os.listdir(source) if name.endswith(".in")
    )
    for name in names:
        case = read_case(os.path.join(source, name))
        stem = name[:-3]
        write_forecast(case, os.path.join(target, stem + ".out"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
