# LLM & Model Accuracy Fixes — Verification Report

**Date:** 2026-08-07
**Scope:** The reported bug cluster (breaking-trades prices, DCF auto-fill crash,
PowerPoint accuracy, financial-model formula/values, LLM entity pollution and
35-part output explosion) plus the requested 10k+ prompt stress testing.

---

## 1. Breaking Trades — Prices, Targets, Alerts

### Reported bug
> Suggested assets priced at $0, all targets and levels priced at $0.
> Trade alerts and notes should be specific, dynamic, tailored per asset/user.

### Root causes
1. `_calculate_price_levels` had **no zero/NaN guard** — a `0.0` or `NaN`
   current price flowed through and produced `$0.00` entry/stop/TP levels that
   were displayed as real setups.
2. Alerts were generic boilerplate; position sizing was a flat percent.

### Fixes (`breaking_trades_generator.py`)
- `_calculate_price_levels` now returns `None` for `0 / NaN / inf / <= 0`
  prices. The caller skips the symbol entirely — **no more $0 setups**.
- Levels dict now carries the `price` used, and `risk_reward` is computed from
  the real entry/stop spread.
- `_generate_reasoning` now emits **asset-specific, price-referenced alerts**:
  - `ENTRY: LONG NVDA on trigger near $99.80 (stop $96.80, 3.0% risk from entry)`
  - `TP1 $105.80 — scale out ~30%` / `TP2` / `TP3` with real computed levels
  - `INVALIDATION: exit flat if price breaks $96.32`
  - `SIZING: suggested X% of portfolio at Y% confidence` — the % comes from the
    user's **trader profile** via `_calculate_position_size` when available.
- Defensive `.get()` on setup fields so partial setup dicts cannot KeyError.

### Verification
- `test_breaking_trades_zero_price_never_emits_setup` — 0/NaN/inf/-5 → None
- `test_breaking_trades_levels_are_positive_and_ordered` — entry/stop/TP sane
  for both BULLISH and BEARISH
- `test_breaking_trades_reasoning_is_asset_specific_with_real_levels` — alerts
  embed the real computed `$entry`/`$stop_loss`, no `$0.00` anywhere
- Existing integration test (`test_breaking_trades_pipeline_with_mocked_providers`)
  still passes.

---

## 2. DCF Auto-Fill — NoneType Crash + AAPL Revenue Scale

### Reported bug
> Auto-fill partial (argument of type 'NoneType' is not iterable). Enter
> assumptions manually. ... I tested AAPL for the DCF and it is not good.

### Root causes
1. `t.info` is a known yfinance failure mode that can be `None` (or a list/
   scalar). The old code iterated/`.items()`d it directly →
   `TypeError: argument of type 'NoneType' is not iterable`.
2. When `totalRevenue` was absent, the code fell back to `revenuePerShare`
   (~$25 for AAPL), which failed the `>1e6` sanity check and **clamped revenue
   to $100M instead of ~$390B** — a 3,900x error that corrupted every DCF.

### Fixes (`financial_model_generator.py`)
- `_safe_info_fetch(ticker_obj)` — defensive wrapper: `None` → `{}`, non-dict →
  coerced or `{}`, drops `None` values. The NoneType crash can no longer escape.
- `_derive_revenue_millions(info, shares)` — per-share fallback now multiplies
  by shares outstanding: `revenuePerShare × shares` ($25 × 15.5B ≈ $387B).
  Falls back conservatively to 0 (never a fabricated floor).
- `_extract_ebitda_millions` — removed the fabricated `500.0` fallback; missing
  EBITDA now yields 0 (callers clamp) instead of inventing $500M.
- `fetch_ticker_fundamentals` — whole function rewritten defensively with a
  last-ditch `fast_info` price/shares recovery and a clear `ValueError` only
  when genuinely no market data exists.

### Verification (mocked-yfinance unit tests, offline-safe)
- `test_autofill_no_none_crash_when_info_missing` — `info=None` returns a dict
- `test_autofill_derives_real_revenue_scale` — $390B revenue → `revenue_m`
  ≈ 390,000, market cap ≈ $3.45T
- `test_autofill_does_not_fabricate_ebitda` — no EBITDA row → `ebitda_m == 0`
  (never 500)

### Live AAPL verification (2026-08-07, real yfinance pull)
```
AUTO-FILL OK:
  ticker = AAPL
  price = 313.33
  revenue_m = 466823.0        # $466.8B — was clamped to ~$100M before the fix
  revenue_growth = 16.4       # %
  ebit_margin_pct = 32.6
  tax_rate_pct = 21.0
  market_cap_m = 4572794.0    # $4.57T
  shares_m = 14594.2          # 14.59B
REVENUE SCALE OK
```
The exact reported case ("I tested AAPL for the DCF") now auto-fills with
correct-scale fundamentals — a 4,600x revenue improvement over the old bug.

---

## 3. Streamlit Duplicate-ID Crash (Precedents Tab)

### Reported bug
> Precedents Engine Error: multiple text_input elements with the same
> auto-generated ID ... pass a unique key.

### Root cause
Widgets in the new tabs used identical labels+defaults (e.g. "Subject Sector"
= "Technology" in both Comps and Precedents), so Streamlit generated identical
auto-IDs across tabs → `StreamlitDuplicateElementId`.

### Fix (`financial_model_generator_ui.py`)
Added unique `key=` to every un-keyed widget in the Precedents, Comps, IPO,
and Valuation-Bridge tabs (`key="prec_subj_sector"`, `key="comps_subj_sector"`,
etc.). Verified no un-keyed widget remains in the colliding label set.

### Verification
- All 26 UI walkthrough tests pass (exercises every tab end-to-end).

---

## 4. PowerPoint Accuracy — Fabricated Fallbacks Removed

### Reported bug
> Please ensure all information in the PowerPoint presentations are always
> 100% true and accurate; some are incorrect.

### Root causes (`presentation_generator.py`)
1. DCF pitchbook read `getattr(r, 'base_revenue', 5000)` — but those values
   live on `r.assumptions`, so **every deck showed the hardcoded fallbacks**
   ($5,000M revenue, 8% growth, 20% EBIT margin) regardless of the model.
2. EV/equity/net-debt were divided by `1e9` but are in $M → "$0.00B".
3. LBO deck used hardcoded 45/15/40 Sources & Uses splits; M&A deck had
   fabricated synergy/earnout figures.

### Fixes
- Added the `_a(r, name, default)` central reader — resolves from
  `r.assumptions` first, then the result, then a default. All three pitchbooks
  now read real model inputs through `_a` / real result fields.
- DCF: `_bn()` divides $M by 1000 → correct $B display; real scenarios,
  real WACC build-up, real catalyst list from the result.
- LBO: Sources & Uses table built from the model's real `sources_uses` /
  `debt_structure` / `total_sources` / `total_uses` — never hardcoded splits.
- M&A: real offer price, premium, pro forma EPS, new shares, after-tax
  synergies from result fields; honest "not modeled" states instead of
  invented numbers.

### Verification
- `test_dcf_pitchbook_shows_real_values` — generated deck contains the real
  `$12,345M` base revenue, `~14%` growth, `32.0%` EBIT margin, and contains
  **no** `$0.00B` / `$5,000M` / `~8%` fallbacks.
- `test_pitchbook_builds_all_deck_types` (existing) + manual byte-level
  inspection of all three decks (see `.temp_trash/verify_*` scripts).

---

## 5. LLM Entity Resolution — the "CAPEX/FCF/GPU as tickers" Bug

### Reported bug (excerpt)
> Live snapshot for CAPEX ... FCF ... GPU ... CUDA ... MOAT ... LOSS ... UNIT
> ... ASPS ... GROSS ... BUILD ... FACTS ... "Unable to fetch current data.
> Please verify the ticker symbol." (35 such sections)

### Root cause
`expand_query_intents` uppercased the **entire query** and regex-matched every
`[A-Z]{1,6}` token as a ticker candidate, relying only on a stopword list.
Every finance concept in the prompt (CAPEX, FCF, GPU, CUDA, MOAT, EV, ...) and
every ALL-CAPS prose word (MULTI, GROSS, BUILD, FACTS, ASPS, ...) became a
ticker and triggered a bogus price fetch.

### Fixes (`financial_llm_engine.py`)
1. **Case-sensitive extraction** — candidates come from the ORIGINAL query
   (`re.findall(r'\b[A-Z]{1,6}\b', query)`), not the uppercased one. Prose
   words are only candidates when the user actually wrote them in ALL CAPS.
2. **`_SEMANTIC_CONCEPTS` blacklist** — 100+ finance/tech/reporting jargon
   words (CAPEX, FCF, DCF, EBITDA, GPU, CUDA, ASIC, WACC, RSI, MACD, LTM→
   NTM/TTM stay, MULTI, GROSS, BUILD, FACTS, ASPS, LOSS, UNIT, CYCLE, ...)
   are **never** securities, dropped unconditionally.
3. **`_AMBIGUOUS_TICKER_CONCEPTS`** — tokens that ARE real securities AND
   concepts (AI, BASE, MOAT, LTM) resolve only when the query references them
   as securities ("MOAT ETF", "AI stock"), never as concepts ("durable moat",
   "AI infrastructure"). `_is_equity_reference` got a strict standalone check
   (`\W*token\W*` fullmatch) so "AI infrastructure" no longer passes the old
   loose "token appears in <40 chars" test.
4. **Universe final gate** — every candidate must be a known ticker-universe
   member or a special symbol (=X/=F/-USD/^/.). Any residual unknown ALL-CAPS
   word is dropped before a price fetch can ever happen.
5. **Ratio-shorthand guard** — a token immediately followed by `/` (EV/EBITDA,
   P/E) is a metric, not a ticker.
6. **Number words** — ONE..BILLION, FIRST..HALF never resolve (even "FIVE" the
   Five-Below ticker requires explicit security context like "buy FIVE").
7. **FX/commodity/crypto/index normalization** — bare 6-char words only become
   `=X` pairs when in the FX whitelist; `CL=F`, `BTC-USD`, `^GSPC` anchors are
   injected from plain-English names; bare contract codes are dropped when the
   `=F` anchor is present.

### Verification
- **100,000 generated prompts** (`llm_stress_test.py 100000`) — **0 failures**
  at ~2,900 prompts/sec:
  - every concept word never leaks in any template × ticker combination
  - every real ticker (NVDA/AAPL/AMD/...) resolves in both cases
  - ambiguous tokens resolve only with security context
  - ratios never yield metric tickers; number words never resolve
- `test_reported_nvda_prompt_produces_only_real_tickers` — the exact reported
  prompt now yields exactly `['NVDA', 'AMD']`.
- The harness ships as `llm_stress_test.py` for repeatable re-runs.

---

## 6. Mega-Query Decomposition — the "35 Parts" Bug

### Reported bug
> The output repeats Market state / Macro / Risk management / IV / position
> sizing / key takeaways over and over ... 35 independent mini-analyses.

### Root causes
`_decompose_mega_query` split the query on every sentence boundary and task
marker with no cap, and `_build_mega_response` emitted a section per fragment
even when consecutive sections had identical intents/labels.

### Fixes (`financial_llm_engine.py`)
1. **Cap**: `MAX_MEGA_PARTS = 6` — queries that would explode into more parts
   fall through to the single comprehensive-analysis path (one coherent
   memo-style answer) instead of 35 repetitive mini-sections.
2. **Label de-duplication**: `_build_mega_response` skips any section whose
   label was already emitted ("Part 1: Macro Outlook / Part 2: Macro Outlook"
   repetition eliminated).
3. **Instrument-gated verb split**: comma-joined multi-asks ("Analyze MSFT,
   hedge AMD, compare TSLA vs NFLX") now decompose — but ONLY when the
   fragments reference at least two DIFFERENT instruments, so a single-company
   deep-dive thesis ("Analyze Apple: revenue model, growth, DCF...") is never
   chopped into parts.

### Verification
- `test_deep_dive_thesis_not_exploded_into_parts` — the NVDA deep-dive returns
  `None` (single comprehensive path)
- `test_multi_part_query_decomposes_within_cap` — 3-part query → 2–6 parts
- `test_comma_joined_multi_ask_decomposes` — multi-instrument split works;
  single-company deep-dive stays whole
- 6/6 decomposition cases pass in the 100k stress run.

---

## 7. Executive-Summary "Unable to fetch" Spam

With entity resolution fixed, invalid tickers no longer reach the live-data
layer at all. For the residual case (valid ticker, provider fetch fails), the
summary path now renders the per-ticker error compactly instead of a long
"Unable to fetch ... Please verify the ticker symbol." line per bogus symbol
(`financial_llm_engine.py` `_build_executive_summary`).

---

## 8. Financial-Model Formula & Value Audit

- All valuation engines were previously hardened with denominator guards
  (`> 0` → `None`) — no division-by-zero/`inf`/`NaN` escapes.
- DCF pitchbook $B-unit fix (above) corrected a display formula.
- Auto-fill revenue/EBITDA derivation fixed (above) so every downstream model
  (DCF/LBO/M&A) receives correct-scale inputs.
- New regression tests lock the DCF math invariants:
  `assert r.enterprise_value > 0` with real assumptions; pitchbook shows the
  computed EV at correct scale.

---

## 9. Review-Driven Corrections

A post-implementation code review caught two additional correctness issues,
both fixed and regression-tested:

1. **Real tickers wrongly blacklisted.** LOW (Lowe's), KEY (KeyCorp) and MAIN
   (Main Street Capital) are genuine universe securities but were in
   `_SEMANTIC_CONCEPTS` / `_STOPWORDS`, so "outlook for LOW" returned nothing.
   They now live in `_AMBIGUOUS_TICKER_CONCEPTS` (with LOW/KEY/MAIN removed
   from the concept blacklist and MAIN from the stopword list): "LOW stock" /
   "KEY stock" / "MAIN stock" resolve; "low volatility" / "key drivers" /
   "main driver" never do.
2. **Revenue last-resort removed.** `_derive_revenue_millions` no longer
   returns the raw per-share figure when shares are unknown (a magnitude
   fabrication); it returns 0 ("n/a") — consistent with the EBITDA honesty fix.

## 10. Test Suite

| Suite | Result |
|---|---|
| `tests/test_llm_and_models_fixes.py` (new, 152 cases) | 152 passed |
| `llm_stress_test.py 100000` (new harness) | 100,000 prompts, 0 failures |
| `tests/` full suite (excluding UI walkthrough) | **327 passed** |
| `tests/test_ui_walkthrough.py` (26 UI flows) | passed (earlier run, no UI change since) |

## Files Changed

- `breaking_trades_generator.py` — zero-price guard, real-levels alerts, trader-profile sizing, defensive setup reads
- `financial_model_generator.py` — defensive `fetch_ticker_fundamentals`, `_safe_info_fetch`, correct revenue derivation, no fabricated EBITDA
- `financial_llm_engine.py` — case-sensitive entity extraction, semantic-concept blacklist, ambiguous-token gate, universe validation, ratio guard, number-word guard, decomposition cap + dedupe + instrument-gated verb split, compact fetch-error summary
- `financial_model_generator_ui.py` — unique widget keys (Precedents/Comps/IPO/Bridge)
- `presentation_generator.py` — `_a()` assumption reader, DCF $B units, real LBO Sources & Uses, real M&A fields, no fabricated fallbacks
- `tests/test_llm_and_models_fixes.py` — new regression suite
- `tests/test_audit_regressions.py` — updated pitchbook assertion to the new `_a()` contract
- `llm_stress_test.py` — 100k-prompt stress harness


---

## 11. Iterative Hardening v9 → v12 — 48,000-Prompt Full-Pipeline Re-Verification

After the initial fix round, the reported NVDA mega-prompt was re-run through a
new four-bucket evaluation harness (`chatbot_eval/`) that measures entity
cleanliness, task fulfillment, ticker relevance and response quality for
12,000 prompts per bucket (small / medium / huge / mega) at 16 workers, with
all answers stored in `chatbot_eval/octavian_chatbot_eval.db`. The engine was
then hardened iteratively until every bucket reached 100%.

### v9 — phrase-match regression closure (48,000 @ 99.96%)
- Added ~30 missing security-context phrases (drawdown/straddle/sector/
  portfolio/think-about) and fixed a mocked-pipeline `None * float` crash by
  making hedging arithmetic defensive and wrapping mega part-builders in
  per-part try/except so one bad part never kills the response.
- Fixed mega task-word boundaries, the "On a different note" connector and an
  over-aggressive identical-profile collapse.

### v10 — single-letter + pair-coordination fixes (48,000 @ 99.99%)
- Word-boundary anchored the single-letter phrase matcher (fixed "e moat"
  matching inside "durable moat").
- Paired-token resolution no longer excludes common-English universe members
  ("SPY or TGT" now extracts both).

### v11 — mega de-dup correctness (48,000 @ 100%)
- Parts naming **different instruments** under the same label are always kept
  ("outlook for VIX" + "outlook for PYPL" = 2 parts); text de-dup applies only
  when instruments match or neither has one.
- "play TGT" / "protect gains in TGT" security-context phrases added.

### v12 — stopword tickers + specific FX pairs (48,000 @ 100%, 100k stress @ 0 failures)
**Reported stress gap:** genuine universe tickers that are also core English
words (ARE = Alexandria Real Estate, ALL = Allstate, AM = Antero Midstream,
CAN, RUN, PLAY, ...) were unconditionally dropped by `_STOPWORDS` — "buy ARE"
/ "ARE stock" returned nothing, while "buy ALL shares" could never resolve.

**Fixes (`financial_llm_engine.py`):**
1. Both `_STOPWORDS` gates now give genuine universe tickers the same escape
   the ambiguous-token gate has: they resolve **only** when (a) written in
   UPPERCASE in the original query and (b) an explicit security-context phrase
   matches — "all stocks are down", "how are stocks doing", "Which technology
   stocks are undervalued?" and lowercase prose never resolve.
2. New `word_boundary=True` mode on `_is_equity_reference` (default unchanged
   for all existing callers): stopword-ticker phrases must end at a word
   boundary so prose plurals ("all stocks", "are prices") can never satisfy
   the "all stock" / "are price" phrases.
3. "trade setup for X" added to the security-context phrase list — this closed
   the final 13/100,000 stress failures ("Give me a trade setup for ARE...").
4. Uppercase requirement doubles as an entity-resolution guard: lowercase
   "gold" stays a commodity (GC=F) and never resolves as the GOLD equity
   ticker; "real estate" prose never resolves The RealReal (REAL).

**FX named-pair quality fix:** a specific-pair ask ("How will USD/BRL react
to the Fed?") previously returned only a generic G10 momentum ranking.
`_build_fx_outlook_response` now detects the pair literally named in the query
and **leads with its live quote**, a directional read, and an evidence-based
Fed-transmission note (real-rate differential / carry / risk appetite);
generic asks still get the momentum table. EM pair labels (USDBRL, USDMXN,
USDTRY, USDZAR, USDINR, ...) added so the lead reads like a currency-desk note.

### Final verification (v12)

| Suite | Result |
|---|---|
| `tests/test_llm_and_models_fixes.py` (266 cases) | 266 passed |
| v9 negative guards (24 probes) | 24/24 |
| `stress_100k.py` extraction/decomposition stress | **100,000 prompts, 0 failures** |
| Eval bucket small (12k) | 12,000 / 12,000 (100%) |
| Eval bucket medium (12k) | 12,000 / 12,000 (100%) |
| Eval bucket huge (12k) | 12,000 / 12,000 (100%) |
| Eval bucket mega (12k) | 12,000 / 12,000 (100%) |
| **Full pipeline total** | **48,000 / 48,000 (100.00%)** |

All answers for every bucket are stored in `chatbot_eval/octavian_chatbot_eval.db`
(keyed by run_id), matching the original request to store results in the same
database and fix issues against real prompt distributions.

### v12 files changed
- `financial_llm_engine.py` — stopword-ticker escape (uppercase + security
  context + word-boundary), "trade setup for X" phrase, FX named-pair lead +
  Fed-transmission note, EM pair labels, cache version bumped to `fa::v12::`
- `tests/test_llm_and_models_fixes.py` — +25 regression cases (v12 suite)
- `chatbot_eval/octavian_chatbot_eval.db` — 48,000 fresh v12 evaluations
