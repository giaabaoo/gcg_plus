"""
STEP 4 - add the compliance direction to the residual stream of BARE harmful prompts.

Intervention: at EVERY layer, every position, every generation step, add
    coeff_l * d_hat_t[l],     coeff_l = alpha_rel * mean_activation_norm[l]
using the harness's own get_activation_addition_input_pre_hook (block input
pre-hook => fires on every forward, matching ablation's write access).
alpha is scaled to the per-layer activation norm, never hardcoded.

Controls:
  baseline  - no intervention (bare prompt)
  random    - norm-matched RANDOM direction, same coeff schedule (MUST NOT jailbreak)

Splits come from results/directions/splits.json: val = alpha selection,
test = final disjoint report. Base prompts are BARE (no persuasion wrapper).

Usage:
  python3 scripts/run_add_direction.py --split val  --techniques Misrepresentation \
      --alphas 0.05 0.1 0.2 0.4 0.8
  python3 scripts/run_add_direction.py --split test --techniques all --alphas 0.2 --random
"""

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "refusal_direction"))
from pipeline.model_utils.model_factory import construct_model_base          # noqa: E402
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook  # noqa: E402

MODEL_PATH = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DIR_DIR = os.path.join(REPO, "results", "directions")
BASE_SET = os.path.join(REPO, "data", "strongreject_313.json")
OUT_ROOT = os.path.join(REPO, "results", "add_direction")
MAX_NEW = 256
BS = 16
SEED = 0


def load_direction(tech):
    p = os.path.join(DIR_DIR, f"directions_{tech.replace(' ', '_')}.pt")
    return torch.load(p, map_location="cpu", weights_only=False)["unit"]   # [L+1, D]


@torch.no_grad()
def generate(mb, prompts, dirs, act_norms, alpha, n_layers):
    """dirs: [L+1, D] unit (or None for baseline). Returns list[str]."""
    handles = []
    if dirs is not None:
        for l in range(n_layers):
            coeff = float(alpha * act_norms[l])
            hook = get_activation_addition_input_pre_hook(vector=dirs[l].clone(), coeff=coeff)
            handles.append(mb.model_block_modules[l].register_forward_pre_hook(hook))
    outs = []
    try:
        for s in range(0, len(prompts), BS):
            chunk = prompts[s:s + BS]
            inp = mb.tokenize_instructions_fn(instructions=chunk)
            inp = {k: v.to(mb.model.device) for k, v in inp.items()}
            ilen = inp["input_ids"].shape[1]
            out = mb.model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
            outs += [mb.tokenizer.decode(o[ilen:], skip_special_tokens=True) for o in out]
    finally:
        for h in handles:
            h.remove()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--techniques", nargs="+", default=["Misrepresentation"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.2])
    ap.add_argument("--random", action="store_true", help="add norm-matched random control")
    ap.add_argument("--baseline", action="store_true", help="add no-intervention baseline")
    args = ap.parse_args()

    splits = json.load(open(os.path.join(DIR_DIR, "splits.json")))
    keep = set(splits[args.split])
    data = [e for e in json.load(open(BASE_SET)) if str(e["id"]) in keep]
    ids = [str(e["id"]) for e in data]
    prompts = [e["prompt"] for e in data]            # BARE harmful prompts
    print(f"split={args.split}  n={len(prompts)} bare prompts", flush=True)

    act_norms = torch.load(os.path.join(DIR_DIR, "act_norms.pt"), map_location="cpu",
                           weights_only=False).float()
    mb = construct_model_base(MODEL_PATH)
    n_layers = mb.model.config.num_hidden_layers

    techs = args.techniques
    if techs == ["all"]:
        techs = sorted(f[len("directions_"):-3].replace("_", " ")
                       for f in os.listdir(DIR_DIR) if f.startswith("directions_"))

    configs = [(t, a) for t in techs for a in args.alphas]
    if args.random:
        configs.append(("__random__", args.alphas[0]))
    if args.baseline:
        configs.append(("__baseline__", 0.0))

    os.makedirs(OUT_ROOT, exist_ok=True)
    for tech, alpha in configs:
        tag = f"{args.split}_{tech.replace(' ', '_')}_a{alpha}"
        out_dir = os.path.join(OUT_ROOT, tag)
        out_path = os.path.join(out_dir, "results.json")
        if os.path.exists(out_path):
            print(f"[skip] {tag}", flush=True)
            continue

        if tech == "__baseline__":
            dirs = None
        elif tech == "__negrefusal__":
            # behavioral version of the P1 guard: does simply adding -r do just as well?
            r = torch.load(os.path.join(REPO, "checkpoints", "direction.pt"),
                           map_location="cpu", weights_only=True).float()
            r = r / r.norm()
            dirs = (-r).unsqueeze(0).repeat(n_layers + 1, 1)
        elif tech == "__random__":
            g = torch.Generator().manual_seed(SEED)
            d_model = mb.model.config.hidden_size
            r = torch.randn(n_layers + 1, d_model, generator=g)
            dirs = r / r.norm(dim=-1, keepdim=True)      # norm-matched random
        else:
            dirs = load_direction(tech)

        print(f"=== {tag} ===", flush=True)
        outs = generate(mb, prompts, dirs, act_norms, alpha, n_layers)
        rows = [{"prompt_id": p, "prompt": q, "suffix": "", "output": o}
                for p, q, o in zip(ids, prompts, outs)]
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"config": {"split": args.split, "technique": tech, "alpha_rel": alpha,
                              "intervention": "activation_addition_all_layers",
                              "model": MODEL_PATH}, "results": rows},
                  open(out_path, "w"), indent=2)
        print(f"  wrote -> {out_path}", flush=True)
        print(f"  sample: {outs[0][:100]!r}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
