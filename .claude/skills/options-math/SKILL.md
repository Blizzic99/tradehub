---
name: options-math
description: >-
  Authoritative reference and verification workflow for every option-pricing formula in this
  project. Invoke this skill BEFORE writing or editing any code that computes option Greeks (Delta,
  Gamma), Black-Scholes price, implied volatility, IV Rank / IV Percentile, max pain / magnet pin,
  GEX (gamma exposure), volatility skew, put/call ratio, or a DTE-scaled stop / profit-target -- in
  alpha_scanner.py or alpha_backtest.py. It gives each formula its cited source (Hull for the core
  math; SqueezeMetrics / tastytrade / Maximum-Pain theory for market conventions) and a known-correct
  worked example, and defines the mandatory check (verify_formulas.py, run automatically by the
  PostToolUse hook, plus the quant-reviewer subagent) that must pass before such code is done.
---

# Options math — formulas, citations, and verification

**When this applies:** any change to option-pricing / Greeks / IV / IV-rank / max-pain-magnet /
GEX / skew / DTE stop-target math, in `alpha_scanner.py` or `alpha_backtest.py`. Per the project's
hard constraints, no such formula changes outside this workflow.

## How to use it (the workflow)

1. **Before editing:** read the relevant entry below. Keep the cited formula and the worked example
   in view — they are the definition of correct.
2. **After editing:** the `PostToolUse` hook (`.claude/settings.json`) runs
   `verify_formulas.py` automatically whenever you Edit/Write `alpha_scanner.py` or
   `alpha_backtest.py`. A mismatch exits 2 and **blocks** the edit, printing which formula drifted.
3. **If the change is intentional** (e.g. Phase 4 DTE tuning): re-derive the new expected value
   against the cited source, then update **both** this SKILL.md worked example **and**
   `verify_formulas.py` in the same change, so they stay in lock-step.
4. **Before "done":** hand the change to the **quant-reviewer** subagent (read-only) to check the
   code against this skill.

Run the check manually:
```
C:/Users/santw/AppData/Local/Python/bin/python.exe .claude/skills/options-math/verify_formulas.py
```

**Citations note:** Hull references are given by chapter title (stable across editions) — pin the
exact edition/page against your own copy of *Options, Futures, and Other Derivatives* (John C. Hull).
Convention sources are named where no textbook applies. Constant `RISK_FREE_RATE = 0.045`.

---

## Core Black-Scholes-Merton — cited to Hull

All anchored to Hull's canonical example: **S = K = 100, T = 1 yr, r = 0.05, σ = 0.20 → call ≈ 10.4506.**

Common terms: `d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)`, `d2 = d1 − σ·√T`, `N(·)` = standard-normal CDF.

### 1. Standard-normal CDF — `alpha_backtest._norm_cdf(x)`
- Formula: `N(x) = 0.5·(1 + erf(x/√2))`.
- Source: Hull, *The Black–Scholes–Merton Model* (cumulative normal `N(·)`); `erf` identity is standard.
- **Worked example:** `_norm_cdf(0.35) = 0.636831`.

### 2. Black-Scholes call price — `alpha_backtest._bs_call(S, K, T, r, sig)`
- Formula: `c = S·N(d1) − K·e^(−rT)·N(d2)`. (Puts via parity `p = c − S + K·e^(−rT)`, used inside `_bs_iv`.)
- Source: Hull, *The Black–Scholes–Merton Model* (the BSM pricing formulas).
- **Worked example:** `_bs_call(100, 100, 1, 0.05, 0.20) = 10.45058` (Hull's canonical value).

### 3. Implied volatility (bisection) — `alpha_backtest._bs_iv(price, S, K, T, r, is_call=True)`
- Method: bisection on σ ∈ [1e-3, 5.0], 64 iterations, inverting `_bs_call` (put via parity); returns
  `None` if `price ≤ intrinsic` (no time value) or the result hits the search bounds.
- Source: Hull, *The Black–Scholes–Merton Model*, "Implied Volatilities" (iterative search — bisection
  is the elementary bracketing method).
- **Worked example:** `_bs_iv(10.4506, 100, 100, 1, 0.05, True) = 0.200000` (inverts example 2).

### 4. Delta — `alpha_scanner._bs_delta(S, K, T, sigma, r, is_call)`
- Formula: call `Δ = N(d1)`; put `Δ = N(d1) − 1`. Returns `None` on degenerate inputs (S/K/T/σ ≤ 0).
- Source: Hull, *The Greek Letters* (Delta of European options).
- **Worked examples:** call `_bs_delta(100,100,1,0.20,0.05,True) = 0.636831`; put `= −0.363169`.

### 5. Gamma — `alpha_scanner._bs_gamma(S, K, T, sigma)`
- Formula: `Γ = N'(d1) / (S·σ·√T)`, where `N'` is the standard-normal PDF; identical for calls & puts.
- **Note:** this function uses the module constant `RISK_FREE_RATE = 0.045` for `r` internally (it
  takes no `r` argument), so worked examples must use r = 0.045.
- Source: Hull, *The Greek Letters* (Gamma).
- **Worked example:** `_bs_gamma(100, 100, 1, 0.20) = 0.018921` (with internal r = 0.045).

### 6. Expected 1-sigma move (DTE-scaled stop/target basis) — inline in `alpha_scanner.compute_trade_mechanics`
- Formula: `E = price · ATM_IV · √(DTE_years)`. Stop `= level ∓ STOP_K·E` (STOP_K = 0.5); target
  `= level ± TARGET_K·E` (TARGET_K = 1.0). Falls back to flat `STOP_BUFFER = 0.0075` when IV/DTE
  are unavailable. `STOP_K`/`TARGET_K` are **house risk multipliers** (per spec), not from Hull.
- Source (of the √T basis): Hull, *Volatility* — the standard deviation of the return over horizon T
  scales as `σ·√T`; `E` is the 1σ price move over the option's life.
- **Worked example:** `E(price=100, ATM_IV=0.20, DTE_years=0.25) = 100·0.20·√0.25 = 10.00`; for a
  long entry at level 100 → stop 95.00, target 110.00. *(Documented + reviewed; not yet in the
  automated hook — see "Coverage" below. When Phase 4 changes this, add it to verify_formulas.py.)*

---

## Market conventions — cited to their originators (no textbook home)

### 7. GEX / gamma exposure — `alpha_scanner._net_gex_at`, `compute_gex`
- Formula: net dealer GEX at spot S = `Σ_strikes  Γ(S,K,T,IV_K)·OI_K·100·S`, with **calls POSITIVE,
  puts NEGATIVE** (dealers assumed net long call gamma / short put gamma). `flip` = spot where net
  GEX crosses zero (swept ±20%, linearly interpolated); `regime` = POSITIVE if spot > flip
  (dealers dampen → magnet-pin thesis holds), else NEGATIVE.
- Source: SqueezeMetrics, *Gamma Exposure (GEX)* white paper (squeezemetrics.com) — the
  dealer-long-call/short-put convention and the price-dampening interpretation. It is an assumption
  (dealer books are not observable), already flagged in-code.
- **Worked example:** asymmetric one-strike book (K=100, T=1, IV=0.20, call OI 2000, put OI 1000,
  S=100): net GEX `= Γ·(2000−1000)·100·S = 0.018921·1000·100·100 = 189209.9`; regime `POSITIVE`.

### 8. Volatility skew (25-delta) — `alpha_scanner.compute_skew`
- Formula: `skew = IV(nearest −0.25-delta put) − IV(nearest +0.25-delta call), ×100` vol points;
  positive = puts richer (downside-hedging demand). Strikes selected by `_bs_delta` nearest ±0.25.
- Source: standard risk-reversal / 25-delta skew convention (Natenberg-style vertical skew); Delta
  from Hull *The Greek Letters*.
- **Worked example:** single put strike IV 0.30, single call strike IV 0.25 → `(0.30−0.25)·100 = 5.0`.

### 9. Put/Call ratio — `alpha_scanner.compute_pcr`
- Formula: `ratio = total put volume / total call volume` (nearest expiry). Flags: `EXTREME-PUT` if
  `ratio ≥ 1.5`, `EXTREME-CALL` if `ratio ≤ 0.5`; `None` when call volume is 0.
- Source: standard put/call-ratio sentiment convention (CBOE).
- **Worked example:** `compute_pcr({call_vol_total:1000, put_vol_total:1500}) = (1.5, "EXTREME-PUT")`.

### 10. Max pain / magnet pin — `alpha_scanner.get_max_pain_strike`
- Formula: over strikes carrying OI, the max-pain strike **minimizes** total in-the-money payout to
  option holders: `pain(c) = Σ_{K>c}(K−c)·callOI_K + Σ_{K<c}(c−K)·putOI_K`. The magnet-pin target is
  that strike (dealer-hedging attractor).
- Source: Maximum Pain theory (a.k.a. max-pain / pinning; Optionistics and standard options-market
  definition).
- **Worked example:** callOI {100:1000, 105:500}, putOI {95:800, 100:1200} → pain(95)=10000,
  **pain(100)=6500**, pain(105)=14000 → max-pain strike **100**. *(Documented + reviewed; add to the
  hook via a pure `_max_pain_from_oi` helper when this is next modified.)*

### 11. IV Rank / IV Percentile — `alpha_scanner._iv_stats_from_history`, `compute_iv_metrics`
- Formula: over the logged ~52-week IV series (≥ `IV_HISTORY_MIN_REAL = 60` days, else realized-vol
  proxy marked `~`): `IV Rank = (cur − min)/(max − min)·100`; `IV Percentile = 100·(#days < cur)/n`.
- Source: tastytrade published definitions of IV Rank and IV Percentile.
- **Worked example:** series min 0.10, max 0.30, cur 0.25 → Rank `= (0.25−0.10)/0.20·100 = 75`;
  with two days [0.10, 0.30] → Percentile `= 100·1/2 = 50`. *(Documented + reviewed; add to the hook
  via a pure `_rank_and_pctl` helper when this is next modified.)*

---

## Coverage: what the automated hook checks today

`verify_formulas.py` (run by the hook on every edit of the two modules) numerically verifies the
**pure, network-free** formula functions: examples **1–5** (BS price, N(·), IV solver, Delta, Gamma)
and **7–9** (GEX net + regime, skew, P/C). Examples **6, 10, 11** (expected-move/stop-target, max
pain, IV rank) live inside network-bound or file-bound functions; they are documented here with
worked examples and are covered by the **quant-reviewer** subagent. When any of those three is next
modified (e.g. Phase 4 touches the DTE stop/target), extract the pure core into a helper
(`_expected_move`, `_max_pain_from_oi`, `_rank_and_pctl`) and add it to `verify_formulas.py` in the
same change — that extraction goes through this same workflow.
