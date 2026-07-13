# data/

## Committed (small, needed to run)

| File | What |
|---|---|
| `advbench_50.json` | 50-prompt AdvBench subset — main attack set. |
| `strongreject_313.json` | Full StrongREJECT prompt set (313). |
| `smoke10_advprefix_targets.json` | 10-prompt smoke set with per-prompt AdvPrefix targets. |

## Gitignored (regenerate; used only by Part-1 r_phi line)

Moved out of git to keep the repo small. Not needed for plain GCG or any Part-2 experiment.

- `rphi_train_v2/{train,val}.jsonl`, `rphi_pool_v2/` — r_phi v2 training data / candidate pool.
- `rphi_train/`, `rphi_pool/` — v1 (superseded).
- `harmbench_audit_sample.json`, `benign_variants.json`, `warmstart_init.json` — one-off
  experiment inputs.

Regenerate the r_phi pools from attack `results.json` files with the pool-building logic in
`scripts/train_rphi_v2.py` (see `checkpoints/README.md`). Everything gitignored here lives in
`_trash/data/` in the working tree from the refactor, if you need the exact originals back.
