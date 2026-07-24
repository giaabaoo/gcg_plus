"""
3-CLASS STEP 1 - generate the missing BENIGN-COMPLY class.

Part 6 already gives two of the three classes (results/compliance/labeled_*.json):
    harmful-comply  = PAP-wrapped harmful request -> SR_norm >= 0.5
    refuse          = PAP-wrapped harmful request -> explicit refusal

Missing third class: BENIGN-COMPLY = the model answers helpfully, no refusal, no
harmful content. Without it, "harmful-comply vs refuse" differs on TWO things at
once (refusing-or-not AND harmful-or-not) and can only recover the refusal axis.

The whole point is that the benign class must be SURFACE-MATCHED to the harmful
class, or a probe separates them by reading "is a persuasion wrapper present".
So we run the IDENTICAL persuader (same few-shot templates, same prefill, same 6
techniques) over ALPACA harmless instructions. The wrapper distribution is held
constant; only the underlying request content changes.

k=1 variant per (prompt, technique): 313 x 6 = 1878 PAPs, which matches the
harmful-comply pool (~1810) without Part 6's 5x cost.

Output -> results/compliance3/raw_benign_<tech>.json   (resumable per technique)
"""

import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import gen_compliance_data as G  # noqa: E402  (persuader machinery: templates, prefill, batching)

ALPACA = os.path.join(REPO, "tools", "refusal_direction", "dataset", "processed", "alpaca.json")
OUT_DIR = os.path.join(REPO, "results", "compliance3")
BASE_SET = os.path.join(REPO, "data", "strongreject_313.json")
HARMLESS_SET = os.path.join(REPO, "data", "harmless_313.json")

K = 1
SEED = 0


def build_harmless_set():
    """Sample N alpaca instructions matched in count to strongreject_313. Stable ids h<i>."""
    if os.path.exists(HARMLESS_SET):
        return json.load(open(HARMLESS_SET))
    n = len(json.load(open(BASE_SET)))
    pool = [e["instruction"].strip() for e in json.load(open(ALPACA))
            if e.get("instruction") and len(e["instruction"].split()) >= 4]
    # de-dup, then sample deterministically
    seen, uniq = set(), []
    for p in pool:
        if p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    rng = random.Random(SEED)
    picked = rng.sample(uniq, n)
    data = [{"id": f"h{i}", "prompt": p, "category": "harmless", "source": "alpaca"}
            for i, p in enumerate(picked)]
    json.dump(data, open(HARMLESS_SET, "w"), indent=2)
    print(f"wrote {len(data)} harmless prompts -> {HARMLESS_SET}", flush=True)
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(G.SEED)

    templates, extract_content = G.load_fewshot_assets()
    techniques = G.FEWSHOT + G.CONTROL
    data = build_harmless_set()
    limit = int(os.environ.get("LIMIT", "0"))          # smoke-test knob; 0 = full set
    if limit:
        data = data[:limit]
        print(f"[LIMIT={limit}] smoke-test subset", flush=True)
    ids = [e["id"] for e in data]
    prompts = [e["prompt"] for e in data]
    print(f"{len(prompts)} harmless prompts x {len(techniques)} techniques x k={K}", flush=True)

    print(f"Loading {G.MODEL_PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(G.MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        G.MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()

    for tech in techniques:
        out_path = os.path.join(OUT_DIR, f"raw_benign_{tech.replace(' ', '_')}.json")
        if os.path.exists(out_path):
            print(f"[skip] {tech} (exists)", flush=True)
            continue

        print(f"\n=== {tech} : persuade (harmless) ===", flush=True)
        user_texts = [G.build_prompt(tech, templates, q) for q in prompts]
        comps = G.batched_generate(model, tok, user_texts, "cuda", G.PERSUADE_MAX_NEW,
                                   do_sample=True, num_return=K, prefill=G.CORE_TAG,
                                   batch_size=G.PERSUADE_BS)
        rows = []
        for pid, q, cs in zip(ids, prompts, comps):
            for vi, raw in enumerate(cs):
                m = extract_content(G.MUTATED_TAG, raw)
                if m and len(m.split()) >= G.MIN_PAP_WORDS:
                    rows.append({"prompt_id": pid, "prompt": q, "technique": tech,
                                 "variant": vi, "pap": m.strip()})
        print(f"  parsed {len(rows)}/{len(prompts) * K} valid PAPs -> target decode", flush=True)
        if rows:
            print(f"  sample PAP: {rows[0]['pap'][:160]!r}", flush=True)

        outs = G.batched_generate(model, tok, [r["pap"] for r in rows], "cuda",
                                  G.TARGET_MAX_NEW, do_sample=False, batch_size=G.TARGET_BS)
        for r, o in zip(rows, outs):
            r["output"] = o[0]

        json.dump({"config": {"technique": tech, "persuader": G.MODEL_PATH,
                              "target": G.MODEL_PATH, "k": K, "n_valid_pap": len(rows),
                              "base_set": "alpaca_harmless_313", "target_decoding": "greedy",
                              "persuade_sampling": {"temperature": 1.0, "top_p": 1.0}},
                   "results": rows}, open(out_path, "w"), indent=2)
        print(f"  wrote {len(rows)} rows -> {out_path}", flush=True)

    print("\nAll benign techniques done.", flush=True)


if __name__ == "__main__":
    main()
