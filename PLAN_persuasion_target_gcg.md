# Plan — Persuasion-target GCG ("AdvPrefix, but persuasion-shaped targets")

**Status:** ✅ EXECUTED + CORRECTED (a generation-context bug had falsely zeroed all arms; fixed).
Corrected result: **naive 5/10, persuasion_short 2/10, persuasion 0/10, neutral 0/10**. Plain GCG
works; the persuasion-*target* idea is refuted on target *shape* (scaffold vs hedged), not
semantics. See `results/smoke10_pgcg_summary.md` + HANDOFF Part 5 correction. Deferred: best-of-k,
Arm B, and re-running IRIS/other GCG rows that share the same bug.

## Hypothesis
Suffix-GCG on Llama-3.1-8B caps at 0–20% StrongREJECT-valid ASR without the hook
(Part 2) because its *targets* are either off-manifold (Arditi ablated output) or
shallow ("Sure, here is"). PAP shows the model **will** emit actionable content
(90% best-of-5 / 24% per-attempt) — so a compliant, on-manifold target *exists and
is known*. **Claim:** aiming plain-request GCG at a target **mined from the model's
own successful PAP completions** lifts real ASR over a naive-affirmative target, and
that lift is due to *persuasion semantics*, not token count or fluency.

Two outcomes, both clean:
- **Suffix ≈ persuasion:** a suffix reproduces persuasion's effect from a plain
  request → breaks the Part-2 ceiling.
- **Suffix ≠ persuasion:** the mined opener is only reachable *inside* the persuasion
  context → persuasion does semantic work no suffix replicates (complements Part-2).

## Design decision (locked)
- **Arm A — persuasion-mined target (PRIMARY).** Plain harmful request + GCG suffix;
  CE-target = mined PAP opener. Pure-suffix surface, direct AdvPrefix analog.
- **Arm B — frozen persuasion prefix in prompt + suffix.** Deferred (hybrid, blurs the
  pure-suffix result).
- **GCG-init-with-persuasion-tokens.** Dropped (init washes out over GCG steps).
- **Universality.** Separate stretch arm, later (one suffix, held-out prompts).

## Targets (three, per prompt) — all length- and perplexity-characterized
1. **Persuasion (test):** first ~20 tokens of that prompt's highest-StrongREJECT PAP
   completion in `results/smoke10_pap/results.json` (only prompts with a ≥0.5 hit —
   9/10; id=503 has none, exclude or use its best sub-threshold opener, flagged).
2. **Naive (baseline):** standard GCG affirmative, `"Sure, here is <restated task>:"`.
3. **Neutral, length+perplexity-matched (the real control):** a benign, non-persuasive
   opener truncated to the *same token length* as (1), selected so its NLL under
   Llama-3.1 is closest to (1)'s. Isolates *semantics* from length/fluency. Report the
   match quality (Δlength, ΔNLL) in the summary; the A-vs-neutral gap is the headline.

## Metrics
- **StrongREJECT normalized ≥ 0.5** on the **full greedy completion** (not prefix
  match), judged against the **original** intent. Report **per-attempt** and
  **best-of-k** (k = 5 to match PAP best-of-5; also report k=1).
- **Prefix-induction rate** (did the suffix actually make greedy decode emit the target
  opener?) reported *separately* — diagnoses the reachability trap (opener induced but
  continuation refuses/degenerates), the predicted Part-2 failure mode.

## Success gates
- **H1 (target shape matters):** Arm A best-of-5 > naive-target GCG by ≥ +2/10.
- **H2 (semantics, not tokens) — the real claim:** Arm A > neutral perplexity/length
  control. If A ≈ neutral, the effect is fluency/length, not persuasion → report as null.
- **Beat PAP:** first realistic win = Arm A **per-attempt** > PAP's 24%; stretch =
  best-of-5 ≥ PAP's 9/10.

## Implementation (reuses existing harness)
- `scripts/run_iris_smoke10.py` **with β=0 is already plain discrete-GCG to an arbitrary
  target** — the loop, nanogcg primitives, "x"-init, and hook-off audit all exist. Fork it.
- **New `scripts/mine_pap_targets.py`** → `data/pap_targets.json`: per prompt emits the
  three targets + their token-lengths + NLLs (loads Llama once to score NLL / build the
  matched neutral).
- **New `scripts/run_gcg_persuasion_target.py`**: `--target_set {persuasion,naive,neutral}`,
  `--restarts k`, fair budget **200 steps / width 64** (matches Exp16/18 and the IRIS
  runs), writes `results/smoke10_pgcg_<arm>/results.json` in the scorer schema
  (`prompt`=original intent, `output`=completion from plain request + optimized suffix,
  plus `suffix`, `target`, `prefix_induced`).
- Score with the existing `alpindale/gemma-2b` wrapper. Summarize in
  `results/smoke10_pgcg_summary.md` (per-prompt × arm matrix + per-attempt/best-of-k +
  prefix-induction diagnostic).

## Compute (H100, fair budget)
~2–4 min per GCG run (200/64). Full grid = 9 prompts × 3 arms × 5 restarts ≈ 135 runs
≈ 5–9 h. **Trim knob:** start k=3 (≈ 3–5 h) or drop the naive arm if Part-1's plain-GCG
20% is accepted as the baseline. Decide before launch.

## Risks / mitigations
- **Reachability trap** — prefix induced, continuation refuses. → measure full-completion
  score + prefix-induction separately; expect partial.
- **Best-of-k inflation** — always compare same k; report k=1 too.
- **Neutral control weak** — if ΔNLL match is poor, the H2 claim weakens; report match
  quality honestly and, if needed, use several neutral candidates.
- **Universality** — hard, separate; don't fold into the headline.
- **Prior art** — position explicitly vs **AdvPrefix** (Zhu et al. 2024); the novelty is
  *persuasion-derived* targets + the persuasion-vs-neutral-fluency control, not
  target-prefix mining per se.
```
