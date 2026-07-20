"""
Step 1: generate ablated responses + capture residual-stream activations.

Design (see discussion in HANDOFF "Early-token decodability"):
  Greedy decode gives ONE response per prompt, which makes response quality a
  prompt-level property -- a probe would be reading "does this prompt afford
  concrete steps", not "has the model committed". So we sample k=5 per prompt at
  temp=1.0 to get WITHIN-prompt variance, and keep one greedy pass for
  continuity with earlier parts of the repo.

For every generated response we save, under the SAME Arditi ablation used at
generation time:
  - residual stream at every decoder-layer output (32 layers, d=4096)
  - pooled two ways: mean over the first 10 response tokens, mean over the full
    response
  - response token length

Activations are captured by a teacher-forced forward pass over
[prompt_ids + response_ids] rather than during generation: identical activations,
far simpler code, and it keeps generation batched.

Resumable: one shard file per prompt under results/earlytok/shards/.
"""

import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks

MODEL_PATH     = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DIRECTION_PATH = os.path.join(REPO, "checkpoints", "direction.pt")
PROMPT_SET     = os.path.join(REPO, "data", "strongreject_313.json")
SHARD_DIR      = os.path.join(REPO, "results", "earlytok", "shards")

K_SAMPLES  = 5      # sampled responses per prompt (the within-prompt contrast)
TEMP       = 1.0
TOP_P      = 1.0    # no nucleus truncation: we want the model's own spread
MAX_NEW    = 512    # matches the judge's MAX_RESPONSE_TOKENS; at 256 every
                    # response hit the cap and the length baseline was censored
EARLY_N    = 10     # "first 10 response tokens"
BASE_SEED  = 1234


def pool_activations(mb, capture, ilen, resp_len):
    """capture[layer] is (1, T, d) for the teacher-forced pass. Response tokens
    occupy positions [ilen, ilen+resp_len)."""
    early_end = min(ilen + EARLY_N, ilen + resp_len)
    early, full = [], []
    for layer in range(len(capture)):
        h = capture[layer][0].float()               # (T, d)
        early.append(h[ilen:early_end].mean(dim=0))
        full.append(h[ilen:ilen + resp_len].mean(dim=0))
    return torch.stack(early).half(), torch.stack(full).half()   # (L, d) each


def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
    data = json.load(open(PROMPT_SET))
    limit = int(os.environ.get("EARLYTOK_LIMIT", "0"))
    if limit:
        data = data[:limit]
    print(f"{len(data)} prompts, k={K_SAMPLES} sampled (temp={TEMP}) + 1 greedy", flush=True)

    print(f"Loading {MODEL_PATH} ...", flush=True)
    mb = construct_model_base(MODEL_PATH)
    model, tok = mb.model, mb.tokenizer
    n_layers = model.config.num_hidden_layers

    direction = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=True).float()
    pre_hooks, hooks = get_all_direction_ablation_hooks(mb, direction)
    print(f"ablation hooks: {len(pre_hooks)} pre + {len(hooks)} fwd; layers={n_layers}", flush=True)

    # capture hooks on decoder-layer outputs (post-ablation residual stream)
    capture = {}

    def make_capture(layer):
        def fn(module, inp, out):
            capture[layer] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    cap_specs = [(mb.model_block_modules[i], make_capture(i)) for i in range(n_layers)]

    def add_all(extra_capture):
        handles = []
        for m, f in pre_hooks:
            handles.append(m.register_forward_pre_hook(f))
        for m, f in hooks:
            handles.append(m.register_forward_hook(f))
        if extra_capture:
            for m, f in cap_specs:
                handles.append(m.register_forward_hook(f))
        return handles

    for i, entry in enumerate(data):
        pid = str(entry["id"])
        shard = os.path.join(SHARD_DIR, f"{pid}.pt")
        if os.path.exists(shard):
            continue

        inputs = mb.tokenize_instructions_fn(instructions=[entry["prompt"]])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ilen = inputs["input_ids"].shape[1]

        # ---- generate (ablation on, no capture) ----
        gens, seeds, modes = [], [], []
        handles = add_all(extra_capture=False)
        with torch.no_grad():
            g_seed = BASE_SEED + i * 100
            torch.manual_seed(g_seed)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gens.append(out[0][ilen:]); seeds.append(g_seed); modes.append("greedy")

            # all k samples in ONE batched call (~5x faster than a loop)
            s_seed = BASE_SEED + i * 100 + 1
            torch.manual_seed(s_seed)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=TEMP, top_p=TOP_P,
                                 num_return_sequences=K_SAMPLES,
                                 pad_token_id=tok.eos_token_id)
            for j in range(K_SAMPLES):
                gens.append(out[j][ilen:]); seeds.append(s_seed); modes.append(f"sample{j}")
        for h in handles:
            h.remove()

        # ---- teacher-forced pass per response, capture activations ----
        rows = []
        for resp_ids, seed, mode in zip(gens, seeds, modes):
            # strip trailing EOS/pad so pooling covers real content only
            keep = resp_ids
            eos_mask = (keep == tok.eos_token_id)
            if bool(eos_mask.any()):
                keep = keep[: int(eos_mask.nonzero()[0])]
            if keep.numel() == 0:
                keep = resp_ids[:1]
            resp_len = int(keep.numel())

            full_ids = torch.cat([inputs["input_ids"][0], keep]).unsqueeze(0)
            capture.clear()
            handles = add_all(extra_capture=True)
            with torch.no_grad():
                model(input_ids=full_ids)
            for h in handles:
                h.remove()

            early, full = pool_activations(mb, capture, ilen, resp_len)
            rows.append({
                "text": tok.decode(keep, skip_special_tokens=True),
                "resp_len": resp_len,
                "seed": seed,
                "mode": mode,
                "early": early,
                "full": full,
                "short": resp_len < EARLY_N,
            })
        capture.clear()

        torch.save({
            "prompt_id": pid, "prompt": entry["prompt"], "category": entry["category"],
            "n_layers": n_layers, "early_n": EARLY_N, "rows": rows,
        }, shard)

        if i % 10 == 0 or i == len(data) - 1:
            lens = [r["resp_len"] for r in rows]
            print(f"[{i+1}/{len(data)}] id={pid} lens={lens} "
                  f"greedy={rows[0]['text'][:60]!r}", flush=True)

    print(f"\nDone -> {SHARD_DIR}", flush=True)


if __name__ == "__main__":
    main()
