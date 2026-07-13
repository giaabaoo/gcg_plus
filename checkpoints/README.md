# checkpoints/

| File | In git? | What |
|---|---|---|
| `direction.pt` | ✅ committed (4.6 MB) | Refusal direction for Llama-3.1-8B-Instruct (Arditi et al.). Used by the combined-loss attack (cosine term) and all Part-2 ablation experiments. |
| `rphi/` | ❌ gitignored (~1.5 GB) | r_phi v1 DeBERTa classifier — **superseded by rphi_v2**, kept only for reference. |
| `rphi_v2/` | ❌ gitignored (~1.5 GB) | r_phi v2 DeBERTa classifier. `best.pt` is loaded by `gcg_plus/attack_combined.py` for candidate selection. |

## Regenerating

- **`direction.pt`** — produced by the vendored `tools/refusal_direction/` pipeline for
  `meta-llama/Llama-3.1-8B-Instruct` (only that model's run is used). If missing, re-run that
  pipeline; the combined attack and ablation experiments need it.
- **`rphi_v2/best.pt`** — train with `scripts/train_rphi_v2.py` (job:
  `sbatch scripts/run_experiment.slurm scripts/train_rphi_v2.py`). Training data is
  `data/rphi_train_v2/` (also gitignored — see `data/README.md`).

Only `attack_combined.py` / the combined-loss configs (`gcg_refusal_dir*`,
`gcg_combined*`) need the r_phi weights. Plain GCG and every Part-2 experiment run without them.
