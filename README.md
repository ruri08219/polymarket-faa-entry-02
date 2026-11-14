# ruri08 faa attribution forecast

Entry for make.fun `polymarket-faa-attribution-02`: forecast the Yes
price of every trading market of Polymarket's "FAA outage attribution" event,
minute by minute, 4 hours ahead, as five quantiles per minute.

## Model

`main.py` (Python 3.13, stdlib only, no network, deterministic) reads a
case from `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`{"forecasts": [{"id", "q"}]}` to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
240 rows x 5 quantiles per market.

Per market, with `x` the last price at the checkpoint:

```
sigma_m = 0.0012 * sqrt(x * (1 - x) / 0.0475)
q_tau(h) = clamp(x + z_tau * sigma_m * sqrt(h) * mult_tau, 0, 1)
```

`mult_tau` is 1.5 for tau in {0.05, 0.95} and 1.0 otherwise; `z_tau` is the
standard normal quantile. Rounded to 6 decimals.

## Why this beats the house baseline

- The baseline (persistence, fixed sigma 0.0025) is too wide: its 90%
  interval covers ~95% of practice minutes, so it pays at every quiet
  minute. A narrower base spread (0.0012) cuts that cost.
- A probability's moves scale like sqrt(p(1-p)): big markets move far
  more in absolute terms than tiny ones. Scaling sigma by sqrt(x(1-x))
  gives jumpy markets the spread they need and keeps quiet ones tight.
- Price jumps are fat-tailed, so the outer 5%/95% quantiles get a 1.5x
  wider spread.
- No drift term: recent drift is negatively correlated with the next
  4 hours, but pulling the median toward the trailing mean did not help
  on practice — persistence wins.

## Practice result

Scored on the kit's six practice cases (p00–p04 against the next case's
grid, p05 against the fetched Polymarket history — the scoring pipeline
reproduces the house baseline's official 0.363589 exactly):

- This entry: **0.288814** (6-case mean pinball loss)
- House baseline: 0.363589
- **20.6% better**, improving every practice case.
