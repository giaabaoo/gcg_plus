"""
STEP 3 - extract per-technique compliance directions.

Primary contrast is WITHIN-technique and input-matched (P2 fix):
    d_t[layer] = mean(A_t activations) - mean(B_t activations)
where A_t = verified harmful-compliance PAPs, B_t = refused PAPs, SAME technique
and same persuasion wrapper, deflections discarded. Activations are taken at the
LAST PROMPT TOKEN (the point generation begins), every layer.

Leakage control: the PAPs derive from strongreject_313 base prompts, so the base
prompts are partitioned extract/val/test. Directions are built ONLY from the
extract split; val is for causal layer/alpha selection, test for the final report.

Outputs -> results/directions/
    directions_<tech>.pt   [n_layers+1, 4096] per-layer unit directions (+ raw)
    act_norms.pt           per-layer mean activation norm (for alpha scaling)
    splits.json            extract/val/test base-prompt ids
    summary.md             counts, cosine-vs-refusal guard, cross-technique matrix

Falsification guard (P1): cosine vs the existing Arditi refusal direction, to rule
out the trivial case d_t ~= -r. Reported as a guard only; the headline is behavioral.
Cross-technique matrix uses MATCHED n (subsampling) so geometric divergence is not
just estimation noise from unequal pool sizes.
"""

import glob
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "NousResearch/Meta-Llama-3.1-8B-Instruct"
COMP_DIR = os.path.join(REPO, "results", "compliance")
OUT_DIR = os.path.join(REPO, "results", "directions")
REFUSAL_PATH = os.path.join(REPO, "checkpoints", "direction.pt")
BASE_SET = os.path.join(REPO, "data", "strongreject_313.json")

SPLIT = (0.60, 0.20, 0.20)   # extract / val / test  (on BASE prompts)
SEED = 0
BS = 8
MIN_POOL = 30                # gate on BOTH pools, within extract split


def make_splits():
    ids = [str(e["id"]) for e in json.load(open(BASE_SET))]
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_ex = int(SPLIT[0] * n)
    n_va = int(SPLIT[1] * n)
    return {"extract": ids[:n_ex], "val": ids[n_ex:n_ex + n_va], "test": ids[n_ex + n_va:]}


@torch.inference_mode()
def last_token_acts(model, tok, texts, device="cuda", bs=BS):
    """[n, n_layers+1, d] hidden states at the final prompt token (left-padded => index -1)."""
    pad_id = tok.pad_token_id or tok.eos_token_id
    out = []
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        seqs = [tok.apply_chat_template([{"role": "user", "content": t}],
                                        add_generation_prompt=True, return_tensors="pt")[0]
                for t in chunk]
        maxlen = max(x.shape[0] for x in seqs)
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, maxlen - x.shape[0]:] = x
            attn[i, maxlen - x.shape[0]:] = 1
        o = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                  output_hidden_states=True)
        hs = torch.stack(o.hidden_states, dim=0)       # [L+1, B, S, D]
        out.append(hs[:, :, -1, :].permute(1, 0, 2).float().cpu())
        del o, hs
    return torch.cat(out, dim=0)


def diff_in_means(A, B):
    """A,B: [n, L+1, D] -> raw diff [L+1, D]."""
    return A.mean(0) - B.mean(0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    splits = make_splits()
    json.dump(splits, open(os.path.join(OUT_DIR, "splits.json"), "w"), indent=2)
    extract_ids = set(splits["extract"])
    print(f"splits: extract={len(splits['extract'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()

    pools, dirs, rows = {}, {}, []
    act_norms = None
    for lf in sorted(glob.glob(os.path.join(COMP_DIR, "labeled_*.json"))):
        d = json.load(open(lf))
        tech = d["config"]["technique"]
        A = [r["pap"] for r in d["results"]
             if r["label"] == "compliance" and r["prompt_id"] in extract_ids]
        B = [r["pap"] for r in d["results"]
             if r["label"] == "refusal" and r["prompt_id"] in extract_ids]
        gate = len(A) >= MIN_POOL and len(B) >= MIN_POOL
        print(f"[{tech}] extract-split A={len(A)} B={len(B)} "
              f"{'PASS' if gate else 'UNDERPOWERED -> skipped'}", flush=True)
        rows.append((tech, len(A), len(B), gate))
        if not gate:
            continue
        actA = last_token_acts(model, tok, A)
        actB = last_token_acts(model, tok, B)
        pools[tech] = (actA, actB)
        raw = diff_in_means(actA, actB)
        dirs[tech] = raw
        if act_norms is None:
            act_norms = torch.cat([actA, actB], 0).norm(dim=-1).mean(0)   # [L+1]
        torch.save({"raw": raw, "unit": raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8),
                    "n_A": len(A), "n_B": len(B), "technique": tech},
                   os.path.join(OUT_DIR, f"directions_{tech.replace(' ', '_')}.pt"))
    torch.save(act_norms, os.path.join(OUT_DIR, "act_norms.pt"))

    # ---- P1 falsification guard: cosine vs Arditi refusal direction ----
    refusal = torch.load(REFUSAL_PATH, map_location="cpu", weights_only=True).float()
    refusal = refusal / refusal.norm()
    guard = {}
    for tech, raw in dirs.items():
        unit = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (unit @ refusal)                     # [L+1]
        guard[tech] = {"per_layer_max_abs": float(cos.abs().max()),
                       "argmax_layer": int(cos.abs().argmax()),
                       "cos_at_argmax": float(cos[cos.abs().argmax()])}

    # ---- cross-technique cosine matrix at MATCHED n ----
    techs = sorted(dirs)
    n_match = min(min(pools[t][0].shape[0], pools[t][1].shape[0]) for t in techs) if techs else 0
    rng = random.Random(SEED)
    matched = {}
    for t in techs:
        A, B = pools[t]
        ia = rng.sample(range(A.shape[0]), n_match)
        ib = rng.sample(range(B.shape[0]), n_match)
        m = diff_in_means(A[ia], B[ib])
        matched[t] = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    lines = ["# Step 3 - compliance directions\n",
             f"splits (base prompts): extract={len(splits['extract'])} "
             f"val={len(splits['val'])} test={len(splits['test'])}, seed={SEED}\n",
             "## Pools (extract split only)\n",
             "| technique | A_t | B_t | gate |", "|---|---|---|---|"]
    for tech, na, nb, g in rows:
        lines.append(f"| {tech} | {na} | {nb} | {'PASS' if g else 'UNDERPOWERED'} |")

    lines += ["\n## Falsification guard: cos(d_t, refusal direction)\n",
              "Trivial-case check only (d_t ~= -r would mean Arditi re-derived).",
              "Headline remains behavioral.\n",
              "| technique | max abs cos over layers | layer | signed cos |", "|---|---|---|---|"]
    for t in techs:
        g = guard[t]
        lines.append(f"| {t} | {g['per_layer_max_abs']:.3f} | {g['argmax_layer']} | "
                     f"{g['cos_at_argmax']:+.3f} |")

    lines += [f"\n## Cross-technique cosine (matched n={n_match} per pool)\n",
              "Layer of max |cos| vs refusal used as a reference layer per column below;",
              "matrix reported at the mid-stack layer 14 (typical best-effect region).\n"]
    L = 14
    lines.append("| | " + " | ".join(techs) + " |")
    lines.append("|---" * (len(techs) + 1) + "|")
    for a in techs:
        cells = [f"{float(matched[a][L] @ matched[b][L]):+.3f}" for b in techs]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    open(os.path.join(OUT_DIR, "summary.md"), "w").write(md)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
