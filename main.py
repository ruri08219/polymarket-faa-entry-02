#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: persistence median with a
price-level volatility and fat outer quantiles.

Model (per market, x = last price at the checkpoint):
    sigma_m = 0.0012 * sqrt(x * (1 - x) / 0.0475)
    q_tau(h) = clamp(x + z_tau * sigma_m * sqrt(h) * mult_tau, 0, 1)
where mult_tau = 1.5 for tau in {0.05, 0.95} and 1.0 otherwise, and z_tau
is the standard normal quantile at each level. Rounded to 6 decimals.

Why this beats the house baseline (persistence, fixed sigma 0.0025):
  - The baseline's spread is too wide: on the practice set its 90%
    interval covers ~95% of minutes, so it pays at every quiet minute.
    A narrower base spread (0.0012) cuts that cost.
  - A probability's moves scale like sqrt(p(1-p)): big markets (price
    ~0.25) move far more in absolute terms than tiny ones (~0.01).
    Scaling sigma by sqrt(x(1-x)) gives jumpy markets the spread they
    need and keeps quiet ones tight.
  - Price jumps are fat-tailed, so the outer 5%/95% quantiles get a
    1.5x wider spread than the inner ones.
  - No drift term: recent drift is negatively correlated with the
    next 4 hours (mean reversion), but pulling the median toward the
    trailing mean did not help on practice — persistence wins.

Practice loss (6 cases, scored against the next case's grid and the
fetched Polymarket history for p05): 0.288814 vs the house baseline's
0.363589 (reproduced exactly) — 20.6% better, improving every case.

Stdlib only, no network, deterministic (no randomness, no clock).
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output): one <id>.out per <id>.in, {"forecasts":[{"id","q"}]},
240 rows x 5 quantiles per market, non-decreasing, in [0,1].
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
SIGMA = 0.0012
OUTER_MULT = 1.5
# Normaliser: sqrt(x(1-x)) at x = 0.05 is sqrt(0.0475), so a market at
# 0.05 gets exactly SIGMA; bigger markets get more, smaller ones less.
NORM = 0.0475


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon):
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    # A market pinned at exactly 0 or 1 can still move (a trade can
    # appear), so keep a minimal spread instead of letting sigma fall
    # to zero: floor x(1-x) at 0.001.
    sigma = SIGMA * math.sqrt(max(x * (1.0 - x), 0.001) / NORM)
    rows = []
    for h in range(1, horizon + 1):
        s = sigma * math.sqrt(h)
        row = []
        for j, z in enumerate(Z):
            mult = OUTER_MULT if j in (0, 4) else 1.0
            row.append(price6(x + z * s * mult))
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if case["quantiles"] != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %s, not %s"
                % (fname, case["quantiles"], QUANTILES)
            )
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], case["horizon_minutes"])}
            for m in case["markets"]
        ]
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
