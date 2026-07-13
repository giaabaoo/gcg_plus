"""
Exp16: Multi-layer refusal direction cosine loss with continuous GD (soft suffix).
Variants:
  A) loss = sum_l cos(h_l, r_l)            (cosine only, every 4th layer)
  B) loss = sum_l cos(h_l, r_l) + CE       (cosine + CE)

Soft suffix: [suffix_len x vocab] logits, optimized with AdamW.
Soft embedding = softmax(logits / tau) @ embed_weight, tau=0.1.
"""

import argparse, json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
MEANDIFF_PATH = "tools/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/generate_directions/mean_diffs.pt"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
TARGET        = "Sure, here is"
SUFFIX_LEN    = 20
STEPS         = 256
LR            = 5e-2
TAU           = 0.1          # softmax temperature for soft embeddings
LAYER_STRIDE  = 4            # use every 4th layer: 0,4,8,...,28  (8 layers)
CE_LAMBDA     = 1.0
SEED          = 42
POS_IDX       = 0            # position index in mean_diffs (0 = pos -5)
MAX_NEW       = 200

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

def load_directions(device, model):
    mean_diffs = torch.load(MEANDIFF_PATH, map_location="cpu", weights_only=False)
    # shape: [5, 32, 4096]; use pos index POS_IDX
    dirs = mean_diffs[POS_IDX]  # [32, 4096]
    # normalise per layer
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers, LAYER_STRIDE))
    return dirs[layers].to(device, torch.float32), layers   # [L, 4096]

def make_inputs(tok, model, prompt, suffix_ids, target_ids, device):
    emb = model.get_input_embeddings()
    messages = [{"role": "user", "content": prompt + " "}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    # split on the trailing space we added as suffix placeholder
    before_str = chat
    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    target_str = " " + target_ids  # reuse variable name — actually target string passed
    return before_ids, emb(before_ids).detach()

def greedy_decode(tok, model, prompt, suffix_str, device):
    text = prompt + " " + suffix_str
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def attack_one(tok, model, dirs, prompt, variant, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    emb = model.get_input_embeddings()
    V = emb.weight.shape[0]

    # build static embeddings
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")

    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    target_str = " " + TARGET
    tgt_ids    = tok([target_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    tgt_len    = tgt_ids.shape[1]

    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()
    target_emb = emb(tgt_ids).detach()

    # initialise suffix logits as uniform
    suffix_logits = torch.zeros(1, SUFFIX_LEN, V, device=device, dtype=torch.float32)
    suffix_logits.requires_grad_(True)
    optimizer = torch.optim.AdamW([suffix_logits], lr=LR)

    best_loss = float("inf")
    best_suffix_str = ""

    for step in tqdm(range(STEPS), leave=False):
        optimizer.zero_grad()

        # soft suffix embedding
        probs      = F.softmax(suffix_logits / TAU, dim=-1)           # [1, S, V]
        suffix_emb = probs @ emb.weight.to(torch.float32)             # [1, S, D]

        inp = torch.cat([before_emb.float(), suffix_emb,
                         after_emb.float(), target_emb.float()], dim=1)
        out = model(inputs_embeds=inp.to(model.dtype), output_hidden_states=True)

        hidden = out.hidden_states  # tuple of [1, seq, D], len = n_layers+1

        # cosine loss: sum over selected layers, at position just before target
        pos = -(tgt_len + 1)
        cos_loss = torch.tensor(0.0, device=device)
        for li, r in enumerate(dirs):
            h = hidden[li + 1][0, pos, :].float()  # +1 because hidden[0]=embed
            cos_loss = cos_loss + F.cosine_similarity(h.unsqueeze(0), r.unsqueeze(0)).squeeze()

        if variant == "A":
            loss = cos_loss
        else:  # variant B: cosine + CE
            ce_loss = F.cross_entropy(
                out.logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0]
            )
            loss = cos_loss + CE_LAMBDA * ce_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            # decode best suffix from argmax of logits
            best_ids = suffix_logits.detach().argmax(dim=-1)  # [1, S]
            best_suffix_str = tok.batch_decode(best_ids)[0]

        if step % 50 == 0:
            print(f"    step={step} loss={loss.item():.4f} cos={cos_loss.item():.4f}", flush=True)

    output = greedy_decode(tok, model, prompt, best_suffix_str, device)
    return {"suffix": best_suffix_str, "output": output, "best_loss": best_loss}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "both"], default="both")
    parser.add_argument("--out_dir", default="results/smoke10_multilayer")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tok, model = load_model(device)
    dirs, layers = load_directions(device, model)
    print(f"Using {len(layers)} layers: {layers}", flush=True)

    with open(DATA_PATH) as f:
        prompts = json.load(f)

    variants = ["A", "B"] if args.variant == "both" else [args.variant]

    for variant in variants:
        label = f"variant_{variant}"
        out_path = os.path.join(args.out_dir, f"{label}.json")
        os.makedirs(args.out_dir, exist_ok=True)

        # resume if partial
        existing = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                saved = json.load(f)
            existing = {r["prompt_id"]: r for r in saved.get("results", [])}

        results = []
        for i, item in enumerate(prompts):
            pid = item.get("prompt_id", i)
            prompt = item["prompt"] if isinstance(item, dict) else item

            if pid in existing:
                print(f"[{i+1}/{len(prompts)}] ID={pid}: skipping (done)")
                results.append(existing[pid])
                continue

            print(f"[{i+1}/{len(prompts)}] ID={pid}: {prompt[:60]}...", flush=True)
            r = attack_one(tok, model, dirs, prompt, variant, device)
            r["prompt_id"] = pid
            r["prompt"]    = prompt
            results.append(r)

            with open(out_path, "w") as f:
                json.dump({"config": {"variant": variant, "layers": layers,
                                      "steps": STEPS, "suffix_len": SUFFIX_LEN,
                                      "lr": LR, "tau": TAU},
                           "results": results}, f, indent=2)

        print(f"Done variant {variant} -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
