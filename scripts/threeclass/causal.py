"""
3-CLASS STEP 5 - causal test: does the HARMFULNESS axis do anything that -r doesn't?

A probe being accurate proves nothing about causality (it can read a downstream
correlate, or a confound). Arditi's bar is intervention, so that is the bar here:
add each direction to the residual stream of HELD-OUT BARE harmful prompts and
measure StrongREJECT AASR.

Arms:
  harmfulness         +d_harm      (harmful_comply vs benign_comply)
  harmfulness_perp_r  +d_harm with its r-component projected out, renormalised
                      <- THE DECISIVE ONE. If the off-refusal-axis part of the
                      harmfulness direction steers, there is something beyond r.
                      If it collapses to baseline, the axis is r wearing a hat.
  refusal_axis        -d_refuse    (our own refusal contrast, negated to jailbreak)
  __negrefusal__      -r           (Arditi; Part 6 reference = 92.2% at a=0.075)
  __random__          norm-matched random control (must not jailbreak)
  __baseline__        no intervention

alpha is swept on `val` and the winner reported on the disjoint `test` split -
Part 6 tuned alpha for one arm only and noted it as a caveat; here each arm gets
its own sweep.

Usage:
  python3 scripts/threeclass/causal.py --split val  --alphas 0.025 0.05 0.075 0.1
  python3 scripts/threeclass/causal.py --split test --alphas 0.075
"""

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "refusal_direction"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from pipeline.model_utils.model_factory import construct_model_base           # noqa: E402
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook  # noqa: E402
from run_add_direction import generate, MODEL_PATH, BASE_SET                  # noqa: E402

DIR_DIR = os.path.join(REPO, "results", "directions")
TC = os.path.join(REPO, "results", "threeclass")
OUT_ROOT = os.path.join(TC, "causal")
SEED = 0


def project_out(d, r):
    """Remove r's component from each layer of d, renormalise to unit."""
    r = r / r.norm()
    out = d - (d @ r).unsqueeze(-1) * r.unsqueeze(0)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def build_arms(n_layers, d_model, dirs, r):
    arms = {}
    if "harmfulness" in dirs:
        h = dirs["harmfulness"].float()
        arms["harmfulness"] = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        arms["harmfulness_perp_r"] = project_out(h, r)
    if "refusal" in dirs:
        f = dirs["refusal"].float()
        # +d points toward REFUSING; negate to jailbreak
        arms["refusal_axis"] = -(f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    arms["__negrefusal__"] = (-r / r.norm()).unsqueeze(0).repeat(n_layers + 1, 1)
    g = torch.Generator().manual_seed(SEED)
    rnd = torch.randn(n_layers + 1, d_model, generator=g)
    arms["__random__"] = rnd / rnd.norm(dim=-1, keepdim=True)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.075])
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    splits = json.load(open(os.path.join(DIR_DIR, "splits.json")))
    keep = set(splits[args.split])
    data = [e for e in json.load(open(BASE_SET)) if str(e["id"]) in keep]
    ids = [str(e["id"]) for e in data]
    prompts = [e["prompt"] for e in data]          # BARE harmful prompts, no wrapper
    print(f"split={args.split}  n={len(prompts)} bare harmful prompts", flush=True)

    dirs = torch.load(os.path.join(TC, "directions.pt"), map_location="cpu", weights_only=False)
    r = torch.load(os.path.join(REPO, "checkpoints", "direction.pt"),
                   map_location="cpu", weights_only=True).float()
    act_norms = torch.load(os.path.join(DIR_DIR, "act_norms.pt"), map_location="cpu",
                           weights_only=False).float()

    mb = construct_model_base(MODEL_PATH)
    n_layers = mb.model.config.num_hidden_layers
    arms = build_arms(n_layers, mb.model.config.hidden_size, dirs, r)

    want = args.arms if args.arms != ["all"] else list(arms)
    configs = [(a, al) for a in want for al in args.alphas]
    if args.baseline:
        configs.append(("__baseline__", 0.0))

    os.makedirs(OUT_ROOT, exist_ok=True)
    for arm, alpha in configs:
        tag = f"{args.split}_{arm}_a{alpha}"
        out_dir = os.path.join(OUT_ROOT, tag)
        out_path = os.path.join(out_dir, "results.json")
        if os.path.exists(out_path):
            print(f"[skip] {tag}", flush=True)
            continue
        d = None if arm == "__baseline__" else arms[arm]
        print(f"=== {tag} ===", flush=True)
        outs = generate(mb, prompts, d, act_norms, alpha, n_layers)
        rows = [{"prompt_id": p, "prompt": q, "suffix": "", "output": o}
                for p, q, o in zip(ids, prompts, outs)]
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"config": {"split": args.split, "arm": arm, "alpha_rel": alpha,
                              "intervention": "activation_addition_all_layers",
                              "model": MODEL_PATH}, "results": rows},
                  open(out_path, "w"), indent=2)
        print(f"  wrote -> {out_path}\n  sample: {outs[0][:110]!r}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
