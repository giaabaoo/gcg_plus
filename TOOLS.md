# External tools

This repo depends on three upstream projects. To keep the repo lean and cloneable,
only **refusal_direction** is vendored (its source is committed under `tools/`); the other
two are documented here — clone them yourself if you want to reproduce those experiments.

## Vendored (in this repo)

### `tools/refusal_direction/`
- Upstream: https://github.com/andyrdt/refusal_direction @ `9d852fae1a`
- **Used by live code**: provides the refusal-direction ablation hooks and the precomputed
  direction (`checkpoints/direction.pt`). Required for `scripts/run_arditi_smoke10.py`,
  `run_exp21_gcg_arditi_hook_target.py`, `run_act_match.py`, `run_reachability_diag.py`,
  and the combined-loss attack's cosine term.
- Only the `meta-llama-3-8b-instruct` run is kept under `pipeline/runs/`; the other 4 models'
  runs were removed (unused).

## Not vendored — clone to reproduce these experiments

### `tools/dynamic-jailbreak/` (DTA — Part 1, 50% no-hook baseline)
- Upstream: https://github.com/ZJU-LLM-Safety/Dynamic-Jailbreaking-Attack @ `d81b5b244d`
- Used by `scripts/run_dta_fixed_target.py` and `scripts/run_dta_smoke10.py`.
- **Local modifications** (were applied on top of upstream): `src/attacker_v3.py` edited and
  `src/suffix_optimization.py` added. The original vendored copy (with these edits) is preserved
  in `_trash/tools/dynamic-jailbreak/` in the working tree if you need the exact version.
- To run DTA: clone into `tools/dynamic-jailbreak/` and re-apply those edits.

### `tools/jailbreak-objectives/` (AdvPrefix — provides pre-generated targets only)
- Upstream: https://github.com/facebookresearch/jailbreak-objectives @ `7b8ea767b0`
- No code is imported; it supplied the per-prompt AdvPrefix target strings baked into
  `data/smoke10_advprefix_targets.json`. You only need to clone it to regenerate those targets.

---
The removed copies (both repos + their `.git` dirs + 4 unused refusal_direction model runs)
live in `_trash/tools/` in the working tree until you delete `_trash/`. Nothing was lost.
