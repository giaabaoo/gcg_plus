"""
Exp20: Discrete GCG with fixed multi-layer cosine loss.

Loss signal (for gradient computation):
    loss = sum over layers [0,4,8,12,16,20,24,28]
               of sum over suffix positions [before_len : before_len + suffix_len]
               of cosine(h_l_s, r_l)

where:
    h_l_s = hidden state at layer l at suffix position s
    r_l   = normalized refusal direction at layer l (from mean_diffs[0, l], normalized)

No CE term, no hook at inference.
Uses discrete token search (GCG) via nanogcg's sample_ids_from_grad and filter_ids.

Key fixes (bugs avoided vs prior experiments):
  1. Layer indexing: hidden[layer_id + 1], not hidden[li + 1] — hidden[0] is embedding
  2. Cosine loss summed over suffix positions [before_len : before_len + suffix_len],
     NOT at -(target_len+1)
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanogcg.gcg import sample_ids_from_grad, filter_ids

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
MEANDIFF_PATH = "tools/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/generate_directions/mean_diffs.pt"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
OUT_DIR       = "results/smoke10_exp20"

SUFFIX_LEN    = 20
STEPS         = 200
SEARCH_WIDTH  = 64
LAYER_STRIDE  = 4   # layers 0,4,8,...,28
POS_IDX       = 0   # mean_diffs[0] = position index for pos -5
SEED          = 42
MAX_NEW       = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(device):
    print(f"Loading {MODEL_PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def load_directions(device, n_layers):
    """Return (dirs, layer_ids) where dirs is [L, 4096] float32 on device."""
    mean_diffs = torch.load(MEANDIFF_PATH, map_location="cpu", weights_only=False)
    dirs_raw   = mean_diffs[POS_IDX]                               # [32, 4096]
    dirs_raw   = dirs_raw / (dirs_raw.norm(dim=-1, keepdim=True) + 1e-8)
    layer_ids  = list(range(0, n_layers, LAYER_STRIDE))            # [0,4,8,...,28]
    return dirs_raw[layer_ids].to(device, torch.float32), layer_ids


def gcg_attack(tok, model, dirs, layer_ids, prompt, device):
    """Run discrete GCG for one prompt. Returns dict with suffix + output."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    n_suffix_layers = len(layer_ids)

    emb = model.get_input_embeddings()

    # ---- build static parts of the sequence --------------------------------
    messages = [{"role": "user", "content": prompt + "{optim_str}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{optim_str}")

    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device, torch.int64)
    after_ids  = tok([after_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device, torch.int64)

    before_len = before_ids.shape[1]
    suffix_len = SUFFIX_LEN

    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()

    # ---- initial suffix: repeated "x" tokens --------------------------------
    optim_ids = tok(
        " ".join(["x"] * suffix_len), add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device, torch.int64)                         # [1, suffix_len]

    best_loss      = float("inf")
    best_ids       = optim_ids.clone()
    loss_history   = []

    for step in tqdm(range(STEPS), leave=False):
        # ---- forward pass to get gradient -----------------------------------
        onehot = torch.zeros(
            *optim_ids.shape, emb.weight.shape[0],
            device=device, dtype=model.dtype
        ).scatter_(-1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)

        suffix_emb = onehot @ emb.weight                           # [1, S, D]

        inp = torch.cat([before_emb, suffix_emb, after_emb], dim=1)
        out = model(inputs_embeds=inp, output_hidden_states=True)
        hidden = out.hidden_states                                 # tuple of len n_layers+1

        # Multi-layer cosine loss over suffix positions
        cos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for li, layer_id in enumerate(layer_ids):
            r       = dirs[li]                                     # [4096]
            # FIX 1: hidden[layer_id + 1] (hidden[0] is embedding output)
            h_layer = hidden[layer_id + 1][0, before_len:before_len + suffix_len, :].float()
            # [suffix_len, D]
            # FIX 2: sum cosine over all suffix positions
            cos_per_pos = F.cosine_similarity(
                h_layer,
                r.unsqueeze(0).expand(suffix_len, -1)
            )                                                       # [suffix_len]
            cos_loss = cos_loss + cos_per_pos.sum()

        cos_loss.backward()

        grad = onehot.grad.detach()                                 # [1, S, V]

        # ---- discrete token search (GCG) ------------------------------------
        with torch.no_grad():
            sampled = sample_ids_from_grad(
                optim_ids.squeeze(0),
                grad.squeeze(0),
                SEARCH_WIDTH,
                topk=256,
                n_replace=1,
                not_allowed_ids=None,
            )                                                       # [W, S]
            sampled = filter_ids(sampled, tok)                     # [W', S]
            n_cands = sampled.shape[0]

            cand_losses = []
            for ci in range(n_cands):
                emb_ci = emb(sampled[ci].unsqueeze(0))             # [1, S, D]
                inp_ci = torch.cat([before_emb, emb_ci, after_emb], dim=1)
                out_ci = model(inputs_embeds=inp_ci, output_hidden_states=True)
                hid_ci = out_ci.hidden_states

                cl = torch.tensor(0.0, device=device, dtype=torch.float32)
                for li, layer_id in enumerate(layer_ids):
                    r   = dirs[li]
                    h_l = hid_ci[layer_id + 1][0, before_len:before_len + suffix_len, :].float()
                    cl  = cl + F.cosine_similarity(
                        h_l, r.unsqueeze(0).expand(suffix_len, -1)
                    ).sum()
                cand_losses.append(cl.item())

            best_ci   = int(torch.tensor(cand_losses).argmin())
            optim_ids = sampled[best_ci].unsqueeze(0)
            step_loss = cand_losses[best_ci]
            loss_history.append(step_loss)

            if step_loss < best_loss:
                best_loss = step_loss
                best_ids  = optim_ids.clone()

        print(f"  step={step} loss={step_loss:.4f} best={best_loss:.4f}", flush=True)

    # ---- decode best suffix and generate response ---------------------------
    best_suffix_str = tok.batch_decode(best_ids, skip_special_tokens=False)[0]

    text    = prompt + " " + best_suffix_str
    msgs    = [{"role": "user", "content": text}]
    chat    = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_gen = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out_ids = model.generate(
            ids_gen, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    output = tok.decode(out_ids[0][ids_gen.shape[1]:], skip_special_tokens=True)

    return {
        "suffix":    best_suffix_str,
        "output":    output,
        "best_loss": best_loss,
        "losses":    loss_history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    tok, model = load_model(device)
    n_layers   = model.config.num_hidden_layers
    dirs, layer_ids = load_directions(device, n_layers)
    print(f"Using {len(layer_ids)} layers: {layer_ids}", flush=True)

    with open(DATA_PATH) as f:
        data = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")

    # Resume support
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        existing = {r["prompt_id"]: r for r in saved.get("results", [])}

    results = list(existing.values())

    for i, item in enumerate(data):
        pid    = item.get("id", item.get("prompt_id", i))
        prompt = item["prompt"]

        if pid in existing:
            print(f"[{i+1}/10] ID={pid}: skipping (done)", flush=True)
            continue

        print(f"\n[{i+1}/10] ID={pid}: {prompt[:60]}...", flush=True)

        r = gcg_attack(tok, model, dirs, layer_ids, prompt, device)
        r["prompt_id"] = pid
        r["prompt"]    = prompt
        results.append(r)

        with open(out_path, "w") as f:
            json.dump({
                "config": {
                    "exp":         "exp20",
                    "layers":      layer_ids,
                    "steps":       STEPS,
                    "search_width": SEARCH_WIDTH,
                    "suffix_len":  SUFFIX_LEN,
                    "seed":        SEED,
                    "loss":        "multilayer_cosine_only",
                },
                "results": results,
            }, f, indent=2)

    print(f"\nDone. Results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
