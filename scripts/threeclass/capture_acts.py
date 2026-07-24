"""
3-CLASS STEP 3 - capture last-prompt-token activations for all three classes.

Classes (all are PAP-WRAPPED inputs, so wrapper surface is held constant):
    harmful_comply : PAP(harmful q)  -> sr_norm >= 0.5           [Part 6 labeled_*]
    refuse         : PAP(harmful q)  -> explicit refusal          [Part 6 labeled_*]
    benign_comply  : PAP(harmless q) -> no refusal, sr_norm < 0.2 [Step 1-2]

NOTE on capture point. Part 7's determinism trap does not apply here: every row is
a DISTINCT input, so activations vary across rows by construction. (The trap only
bites when you try to separate k samples of the SAME prompt, where the prompt-token
state is identical.) This is Arditi's setup - input-side capture, distinct prompts -
but with OUTPUT-BEHAVIORAL labels instead of input harmful/harmless labels.

Leakage control: harmful rows inherit Part 6's extract/val/test split over base
prompt ids (results/directions/splits.json); benign rows get a parallel split over
their own h-ids at the same proportions/seed. No base prompt crosses splits.

Output -> results/threeclass/acts.pt
    acts   [N, L+1, D] float16      meta: cls, prompt_id, technique, split, sr_norm
"""

import glob
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from extract_compliance_directions import last_token_acts  # noqa: E402

MODEL_PATH = "NousResearch/Meta-Llama-3.1-8B-Instruct"
COMP_DIR = os.path.join(REPO, "results", "compliance")
COMP3_DIR = os.path.join(REPO, "results", "compliance3")
DIR_DIR = os.path.join(REPO, "results", "directions")
OUT_DIR = os.path.join(REPO, "results", "threeclass")
SPLIT = (0.60, 0.20, 0.20)
SEED = 0
BS = 8


def benign_splits(ids):
    """Parallel extract/val/test over benign base ids, same proportions/seed."""
    ids = sorted(set(ids))
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_ex, n_va = int(SPLIT[0] * n), int(SPLIT[1] * n)
    return {"extract": set(ids[:n_ex]), "val": set(ids[n_ex:n_ex + n_va]),
            "test": set(ids[n_ex + n_va:])}


def collect():
    """-> list of dicts with pap/cls/prompt_id/technique/split/category/sr_norm."""
    rows = []

    cat_of = {str(e["id"]): e.get("category", "?")
              for e in json.load(open(os.path.join(REPO, "data", "strongreject_313.json")))}

    hsplit = json.load(open(os.path.join(DIR_DIR, "splits.json")))
    h_of = {}
    for name in ("extract", "val", "test"):
        for pid in hsplit[name]:
            h_of[str(pid)] = name

    for lf in sorted(glob.glob(os.path.join(COMP_DIR, "labeled_*.json"))):
        d = json.load(open(lf))
        tech = d["config"]["technique"]
        for r in d["results"]:
            cls = {"compliance": "harmful_comply", "refusal": "refuse"}.get(r["label"])
            if cls is None:
                continue                                   # deflection -> discard
            sp = h_of.get(str(r["prompt_id"]))
            if sp is None:
                continue
            rows.append({"pap": r["pap"], "cls": cls, "prompt_id": str(r["prompt_id"]),
                         "technique": tech, "split": sp,
                         "category": cat_of.get(str(r["prompt_id"]), "?"),
                         "variant": r.get("variant", 0),
                         "sr_norm": r.get("sr_norm", 0.0)})

    bfiles = sorted(glob.glob(os.path.join(COMP3_DIR, "labeled_benign_*.json")))
    if not bfiles:
        sys.exit("No labeled_benign_*.json - run label_benign.py first")
    ball = [r for f in bfiles for r in json.load(open(f))["results"]]
    bsp = benign_splits([r["prompt_id"] for r in ball])
    for f in bfiles:
        d = json.load(open(f))
        tech = d["config"]["technique"]
        # capture ALL benign rows, whatever their label. Only benign_comply enters a
        # contrast, but capturing the rest means a relabel (the drift guard is the
        # fragile part of this pipeline) costs a dict edit, not another GPU pass.
        for r in d["results"]:
            pid = r["prompt_id"]
            sp = ("extract" if pid in bsp["extract"] else
                  "val" if pid in bsp["val"] else "test")
            rows.append({"pap": r["pap"], "cls": r["label"], "prompt_id": pid,
                         "technique": tech, "split": sp, "category": "harmless",
                         "variant": r.get("variant", 0),
                         "sr_norm": r.get("sr_norm_diagnostic", r.get("sr_norm", 0.0))})
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = collect()
    from collections import Counter
    print("class counts:", Counter(r["cls"] for r in rows), flush=True)
    print("split counts:", Counter(r["split"] for r in rows), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()

    texts = [r["pap"] for r in rows]
    chunks = []
    CH = 512
    for s in range(0, len(texts), CH):
        chunks.append(last_token_acts(model, tok, texts[s:s + CH], bs=BS).half())
        print(f"  acts {min(s + CH, len(texts))}/{len(texts)}", flush=True)
    acts = torch.cat(chunks, 0)
    print("acts:", tuple(acts.shape), acts.dtype, flush=True)

    torch.save({"acts": acts,
                "cls": [r["cls"] for r in rows],
                "prompt_id": [r["prompt_id"] for r in rows],
                "technique": [r["technique"] for r in rows],
                "split": [r["split"] for r in rows],
                "category": [r["category"] for r in rows],
                "variant": [r["variant"] for r in rows],
                "sr_norm": [r["sr_norm"] for r in rows]},
               os.path.join(OUT_DIR, "acts.pt"))
    print(f"wrote -> {os.path.join(OUT_DIR, 'acts.pt')}", flush=True)


if __name__ == "__main__":
    main()
