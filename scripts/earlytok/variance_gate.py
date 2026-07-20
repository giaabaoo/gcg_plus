"""
Step 3: THE GATE. Run before any probe.

The within-prompt design only has something to decode if sampling at temp=1.0
actually produces spread in response quality WITHIN a prompt. If nearly all
variance is between prompts, the contrast is underpowered by construction and
the design needs rethinking -- stop and report rather than fit a probe.

Reports:
  - variance decomposition of sr_expected: between-prompt vs within-prompt
  - ICC(1) = between / (between + within). ICC near 1.0 => no within signal.
  - per-prompt spread distribution (SD, max-min)
  - the score distribution itself: if degenerate (all high / all low), the
    median split is meaningless -- also a stop condition (user's note).
  - judge determinism check: rescoring is a single deterministic forward pass,
    so test-retest should be exactly 1.0 and all variance above is generation
    variance. Verified empirically on 50 responses x 3.

Writes results/earlytok/variance_gate.json + prints a verdict.
"""

import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORES = os.path.join(REPO, "results", "earlytok", "scores.json")
OUT    = os.path.join(REPO, "results", "earlytok", "variance_gate.json")

# gate thresholds (judgment calls; raw numbers reported regardless)
ICC_MAX          = 0.90   # above this, within-prompt variance is negligible
MED_WITHIN_SD_MIN = 0.15  # on the 1-5 expected-score scale
DEGENERATE_FRAC   = 0.95  # if >95% of scores fall on one side of the midpoint


def main():
    rows = json.load(open(SCORES))
    # the within-prompt contrast uses the SAMPLED responses only; greedy is a
    # different decode regime and would act as an outlier in every group.
    samp = [r for r in rows if r["mode"] != "greedy"]
    kept = [r for r in samp if not r["refusal_heuristic"]]
    print(f"{len(rows)} responses; {len(samp)} sampled; "
          f"{len(kept)} after dropping heuristic refusals", flush=True)

    by_prompt = {}
    for r in kept:
        by_prompt.setdefault(r["prompt_id"], []).append(r["sr_expected"])
    groups = {p: v for p, v in by_prompt.items() if len(v) >= 2}
    print(f"{len(groups)} prompts with >=2 usable sampled responses", flush=True)

    all_scores = [s for v in groups.values() for s in v]
    grand = st.mean(all_scores)

    # one-way ANOVA style decomposition
    within_ss = sum((s - st.mean(v)) ** 2 for v in groups.values() for s in v)
    between_ss = sum(len(v) * (st.mean(v) - grand) ** 2 for v in groups.values())
    n, k = len(all_scores), len(groups)
    within_ms = within_ss / max(n - k, 1)
    between_ms = between_ss / max(k - 1, 1)
    k0 = st.mean([len(v) for v in groups.values()])
    var_between = max((between_ms - within_ms) / k0, 0.0)
    icc = var_between / (var_between + within_ms) if (var_between + within_ms) > 0 else 0.0

    sds = [st.stdev(v) for v in groups.values() if len(v) >= 2]
    rngs = [max(v) - min(v) for v in groups.values()]

    lo = sum(s < 3.0 for s in all_scores) / len(all_scores)
    degenerate = max(lo, 1 - lo) > DEGENERATE_FRAC

    rep = {
        "n_responses_scored": len(rows),
        "n_sampled": len(samp),
        "n_after_refusal_drop": len(kept),
        "n_prompts_usable": len(groups),
        "score_mean": round(grand, 4),
        "score_sd": round(st.stdev(all_scores), 4),
        "score_frac_below_3": round(lo, 4),
        "var_between_prompt": round(var_between, 5),
        "var_within_prompt": round(within_ms, 5),
        "icc1": round(icc, 4),
        "within_sd_median": round(st.median(sds), 4),
        "within_sd_mean": round(st.mean(sds), 4),
        "within_range_median": round(st.median(rngs), 4),
        "frac_prompts_range_ge_0.3": round(sum(r >= 0.3 for r in rngs) / len(rngs), 4),
        "cap_frac": round(sum(r["resp_len"] >= 512 for r in rows) / len(rows), 4),
    }

    # two distinct failure modes, which warrant different responses:
    #   underpowered - there IS a real axis, we just may not have the spread to
    #                  resolve it. Probe is still worth running, clearly labelled.
    #   degenerate   - the split itself is meaningless (all scores one side), so
    #                  a probe result could not be interpreted at all. Hard stop.
    underpowered = []
    if icc > ICC_MAX:
        underpowered.append(f"ICC(1)={icc:.3f} > {ICC_MAX}: within-prompt variance negligible")
    if st.median(sds) < MED_WITHIN_SD_MIN:
        underpowered.append(f"median within-prompt SD={st.median(sds):.3f} < {MED_WITHIN_SD_MIN}")
    degen = ([f"score distribution degenerate ({lo:.1%} below 3.0): median split meaningless"]
             if degenerate else [])

    rep["gate_pass"] = not (underpowered or degen)
    rep["underpowered"] = underpowered
    rep["degenerate"] = degen

    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print()
    if degen:
        print("GATE FAILED (DEGENERATE) -- hard stop, probe would be uninterpretable:")
        for f in degen:
            print("  -", f)
        sys.exit(3)
    if underpowered:
        print("GATE WARNING (UNDERPOWERED) -- probe will run but must be reported as such:")
        for f in underpowered:
            print("  -", f)
        sys.exit(2)
    print("GATE PASSED: within-prompt spread is sufficient; proceed to probe.")


if __name__ == "__main__":
    main()
