"""
SANITY CHECK (must pass before trusting any compliance direction d_t).

Reproduce Arditi refusal-direction ablation on THIS box (local H100, ungated
mirror) and confirm it flips bare harmful prompts refuse -> comply at ~ceiling.
If it doesn't, the hooks are wired wrong and every downstream d_t is untrustworthy.

Reuses the real harness only:
  - tools/refusal_direction ... construct_model_base   (model + tokenize_instructions_fn)
  - tools/refusal_direction ... get_all_direction_ablation_hooks  (the intervention)
No reimplementation. Writes results.json in the standard schema so
scripts/score_with_gemma.py reads it directly.
"""

import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools", "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks

MODEL_PATH     = "NousResearch/Meta-Llama-3.1-8B-Instruct"   # cached ungated mirror
DIRECTION_PATH = os.path.join(REPO, "checkpoints", "direction.pt")
BASE_SET       = os.path.join(REPO, "data", "strongreject_313.json")
N_PROMPTS      = 20                                          # cheap sanity sample
OUT_PATH       = os.path.join(REPO, "results", "sanity_ablation", "results.json")
MAX_NEW_TOKENS = 256


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    data = json.load(open(BASE_SET))[:N_PROMPTS]
    prompts    = [e["prompt"] for e in data]
    prompt_ids = [str(e["id"]) for e in data]
    print(f"Sanity on {len(prompts)} bare harmful prompts from strongreject_313", flush=True)

    print(f"Loading {MODEL_PATH} ...", flush=True)
    mb = construct_model_base(MODEL_PATH)

    direction = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=True).float()
    print(f"Refusal direction: {tuple(direction.shape)}", flush=True)

    pre_hooks, hooks = get_all_direction_ablation_hooks(mb, direction)

    results = []
    for i, (pid, prompt) in enumerate(zip(prompt_ids, prompts)):
        inputs = mb.tokenize_instructions_fn(instructions=[prompt])
        inputs = {k: v.to(mb.model.device) for k, v in inputs.items()}
        ilen = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = mb.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        baseline = mb.tokenizer.decode(out[0][ilen:], skip_special_tokens=True)

        handles = []
        for module, fn in pre_hooks:
            handles.append(module.register_forward_pre_hook(fn))
        for module, fn in hooks:
            handles.append(module.register_forward_hook(fn))
        with torch.no_grad():
            out = mb.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        for h in handles:
            h.remove()
        ablation = mb.tokenizer.decode(out[0][ilen:], skip_special_tokens=True)

        results.append({
            "prompt_id": pid, "prompt": prompt, "suffix": "",
            "output": ablation, "baseline_output": baseline,
        })
        print(f"  [{i+1}/{len(prompts)}] id={pid}", flush=True)
        print(f"    base: {baseline[:70]!r}", flush=True)
        print(f"    abl : {ablation[:70]!r}", flush=True)

    json.dump({"config": {"model": MODEL_PATH, "attack": "arditi_ablation_sanity",
                          "direction": DIRECTION_PATH}, "results": results},
              open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
