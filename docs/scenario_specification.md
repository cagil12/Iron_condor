# Scenario Specification: Synthetic 0DTE IC Backtester

This document formalizes how the current engine in `src/research/synthetic_backtest.py` computes the four scenarios:

- `hold_to_expiry`
- `tp50_or_expiry`
- `tp50_sl_capped`
- `worst_case`

Scope: this is a behavior specification of current code, not a redesign.

## Shared Inputs

For each trading day:

- `S_open`, `S_high`, `S_low`, `S_close`
- `K_short_put`, `K_long_put`
- `K_short_call`, `K_long_call`
- `credit_received` (per-share net IC credit, after haircut)
- `wing_width` (`K_short_put - K_long_put = K_long_call - K_short_call`)
- `tp_pct`
- `sl_mult`

Derived:

- `credit_usd = credit_received * 100`
- `max_loss_usd = max(0, (wing_width - credit_received) * 100)`
- `tp_cap_usd = tp_pct * credit_usd`
- `sl_cap_usd = sl_mult * credit_usd`

Per-share settlement value decomposition:

```text
put_spread_value = max(0, K_short_put - S_close) - max(0, K_long_put - S_close)
call_spread_value = max(0, S_close - K_short_call) - max(0, S_close - K_long_call)
settlement_pnl_per_share = credit_put - put_spread_value + credit_call - call_spread_value
settlement_pre_commission = settlement_pnl_per_share * 100
```

## Shared Touch Detection

Current implementation (`_simulate_dataframe`, lines ~547-550):

```text
put_breached  = (S_low <= K_short_put)
call_breached = (S_high >= K_short_call)
touched_short = put_breached OR call_breached
```

- Touch is inclusive (`<=` on put side, `>=` on call side).
- Wings are also flagged (`S_low <= K_long_put`, `S_high >= K_long_call`) but not directly used by scenario payouts.

ASSUMPTION: OHLC gives extrema but not intraday ordering of high vs low.
ASSUMPTION: Barrier touches are inferred from index OHLC, not from option spread marks.
ASSUMPTION: Close-to-close settlement pricing is used for all non-worst-case scenarios.
ASSUMPTION: Exact strike equality counts as touched.

---

## Scenario 1: `hold_to_expiry`

### Touch Logic

- Intraday touches are computed but do not alter hold-to-expiry PnL formula.
- Outcome is driven by settlement payoff (`S_close`) and scenario commission rule.

### PnL Formula

```text
pre_commission = settlement_pre_commission
```

Commission (`conditional` model):

```text
if (S_close >= K_short_put) and (S_close <= K_short_call):
    commission = open_commission   # $2.60 in fixed plan
else:
    commission = round_trip_commission  # $5.20
```

Final:

```text
outcome_hold_to_expiry = pre_commission - commission
```

### Outcome Paths

- No touch, close between shorts: max profit at settlement, open-only commission.
- Put touch/call touch/both touch but close between shorts: still settlement win, open-only commission.
- Close outside short strikes: settlement loss/profit on breached side, round-trip commission.

### Limitations

ASSUMPTION: If close is between short strikes, the position is treated as expired worthless for commission, even if intraday touch occurred.

---

## Scenario 2: `worst_case`

### Touch Logic

```text
if touched_short:
    pre_commission = -max_loss_usd
else:
    pre_commission = credit_usd
```

### PnL Formula

```text
commission = round_trip_commission
outcome_worst_case = pre_commission - commission
```

### Outcome Paths

- No short touch: full credit.
- Put touch / call touch / both touch: immediate max-loss assumption.

### Limitations

ASSUMPTION: Any short touch is treated as terminal max loss regardless of recovery by close.

---

## Scenario 3: `tp50_or_expiry`

### Touch Logic

```text
untouched = (not touched_short) and isfinite(settlement_pre_commission)
```

### TP Inference from OHLC (critical)

Current code uses:

```text
if untouched and settlement_pre_commission > 0:
    pre_commission = min(settlement_pre_commission, tp_cap_usd)
else:
    pre_commission = settlement_pre_commission
```

This is not time-based and not option-mark based. It is an index-touch + settlement heuristic:

- If neither short strike is touched and settlement is profitable, cap profit to TP amount.
- If any short strike is touched, no TP cap is applied; scenario reverts to settlement payoff.

So TP detection is best described as:

- (b) distance/touch-based proxy, combined with
- settlement-based cap (`min(settlement, tp_cap)`),
- not an explicit intraday fill model.

### PnL Formula

```text
commission = round_trip_commission
outcome_tp50_or_expiry = pre_commission - commission
```

### Outcome Paths

- No touch + profitable settlement: capped to TP.
- No touch + non-profitable settlement: settlement value.
- Put/call/both touched: settlement value (uncapped), then round-trip commission.

### Limitations

ASSUMPTION: No explicit SL precedence is enforced in this scenario when touch happens.

FINDING: `tp50_or_expiry` does not implement "SL before TP" ordering on ambiguous OHLC bars. It uses settlement when touched. Impact: medium (can overstate performance on touch-and-recover days).

---

## Scenario 4: `tp50_sl_capped`

### Touch Logic

```text
if touched_short:
    pre_commission = max(settlement_pre_commission, -sl_cap_usd)
else:
    if settlement_pre_commission > 0:
        pre_commission = min(settlement_pre_commission, tp_cap_usd)
    else:
        pre_commission = settlement_pre_commission
```

### PnL Formula

```text
commission = round_trip_commission
outcome_tp50_sl_capped = pre_commission - commission
```

### Outcome Paths

- No touch + profitable settlement: TP cap.
- No touch + non-profitable settlement: settlement.
- Touch on either side: settlement floored at `-sl_mult * credit`.

### Limitations

ASSUMPTION: On touched days, TP cap is not applied; only SL floor is applied.

FINDING: `tp50_sl_capped` is not guaranteed to lie between `worst_case` and `hold_to_expiry`. On loss days it can be better than hold (SL cap), and on untouched wins it can be below worst due to TP cap + round-trip commissions. Impact: low-to-medium (important for interpretation of "bounds").

---

## Ambiguity Resolution: High/Low Ordering

If both `S_low <= K_short_put` and `S_high >= K_short_call` occur the same day:

- `worst_case`: max loss.
- `hold_to_expiry`: settlement only.
- `tp50_or_expiry`: settlement only.
- `tp50_sl_capped`: settlement with SL floor.

There is no intraday sequencing model in current code.

FINDING: Global rule "assume SL hit before TP when ambiguous" is only strictly represented by `worst_case`, not by `tp50_or_expiry` or `hold_to_expiry`. Impact: medium.
