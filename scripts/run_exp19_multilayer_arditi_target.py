"""
Exp19: Multi-layer cosine loss (fixed) + Arditi output as CE target, continuous GD.

Fixes vs Exp16:
  1. Correct layer indexing: hidden[layers[li] + 1] not hidden[li + 1]
  2. Cosine loss summed over ALL suffix positions, not just pos before target
  3. CE target = first 50 tokens of Arditi harmful output (confirmed asr=1)

Variants:
  A) loss = sum_{l in layers} sum_{s in suffix_pos} cos(h_l_s, r_l)
  B) loss = cosine_A + CE(arditi_output[:50 tokens])
"""

import argparse, json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
MEANDIFF_PATH = "tools/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/generate_directions/mean_diffs.pt"
ARDITI_PATH   = "results/smoke10_arditi/results.json"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
SUFFIX_LEN    = 20
TARGET_TOKENS = 50
STEPS         = 256
LR            = 5e-2
TAU           = 0.1
CE_LAMBDA     = 1.0
LAYER_STRIDE  = 4   # layers 0,4,8,12,16,20,24,28
POS_IDX       = 0   # mean_diffs position index (pos -5)
SEED          = 42
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

def load_directions(device, n_layers):
    mean_diffs = torch.load(MEANDIFF_PATH, map_location="cpu", weights_only=False)
    dirs_raw = mean_diffs[POS_IDX]                                    # [32, 4096]
    dirs_raw = dirs_raw / (dirs_raw.norm(dim=-1, keepdim=True) + 1e-8)
    layer_ids = list(range(0, n_layers, LAYER_STRIDE))
    return dirs_raw[layer_ids].to(device, torch.float32), layer_ids   # [L, 4096]

def attack_one(tok, model, dirs, layer_ids, prompt, arditi_text, variant, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    emb = model.get_input_embeddings()
    V   = emb.weight.shape[0]

    # static embeddings: before / after suffix placeholder
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")

    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device)
    before_len = before_ids.shape[1]

    # CE target: first TARGET_TOKENS of Arditi output
    tgt_ids = tok(arditi_text, add_special_tokens=False,
                  return_tensors="pt")["input_ids"][:, :TARGET_TOKENS].to(device)
    tgt_len = tgt_ids.shape[1]

    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()
    target_emb = emb(tgt_ids).detach()

    # suffix positions in the full sequence (after before_emb)
    suffix_start = before_len
    suffix_end   = before_len + SUFFIX_LEN  # exclusive

    suffix_logits = torch.zeros(1, SUFFIX_LEN, V,
                                device=device, dtype=torch.float32,
                                requires_grad=True)
    optimizer = torch.optim.AdamW([suffix_logits], lr=LR)

    best_loss, best_suffix_str = float("inf"), ""

    for step in tqdm(range(STEPS), leave=False):
        optimizer.zero_grad()

        probs      = F.softmax(suffix_logits / TAU, dim=-1)
        suffix_emb = (probs @ emb.weight.float()).to(model.dtype)

        inp = torch.cat([before_emb, suffix_emb, after_emb, target_emb], dim=1)
        out = model(inputs_embeds=inp, output_hidden_states=True)

        hidden = out.hidden_states  # len = n_layers+1, each [1, seq, D]

        # FIX 1: correct layer index — hidden[layer_id + 1] for each layer
        # FIX 2: sum cosine over all suffix positions
        cos_loss = torch.tensor(0.0, device=device)
        for li, r in enumerate(dirs):
            layer_id = layer_ids[li]
            h_layer  = hidden[layer_id + 1][0, suffix_start:suffix_end, :].float()  # [S, D]
            # cosine sim between each suffix position and refusal direction
            cos_per_pos = F.cosine_similarity(h_layer, r.unsqueeze(0).expand(SUFFIX_LEN, -1))
            cos_loss = cos_loss + cos_per_pos.sum()

        if variant == "A":
            loss = cos_loss
        else:
            ce_loss = F.cross_entropy(
                out.logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0]
            )
            loss = cos_loss + CE_LAMBDA * ce_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_ids  = suffix_logits.detach().argmax(dim=-1)
            best_suffix_str = tok.batch_decode(best_ids)[0]

        if step % 64 == 0:
            ce_val = ce_loss.item() if variant == "B" else 0.0
            print(f"    step={step} loss={loss.item():.3f} cos={cos_loss.item():.3f}"
                  + (f" ce={ce_val:.3f}" if variant == "B" else ""), flush=True)

    # final greedy decode
    text = prompt + " " + best_suffix_str
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
    return {"suffix": best_suffix_str, "output": output, "best_loss": best_loss}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "both"], default="both")
    parser.add_argument("--out_dir", default="results/smoke10_exp19")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    tok, model = load_model(device)
    n_layers = model.config.num_hidden_layers
    dirs, layer_ids = load_directions(device, n_layers)
    print(f"Using {len(layer_ids)} layers: {layer_ids}", flush=True)

    with open(ARDITI_PATH) as f:
        arditi = json.load(f)
    arditi_map = {r["prompt_id"]: r["output"] for r in arditi["results"]}

    with open(DATA_PATH) as f:
        data = json.load(f)

    variants = ["A", "B"] if args.variant == "both" else [args.variant]
    os.makedirs(args.out_dir, exist_ok=True)

    for variant in variants:
        out_path = os.path.join(args.out_dir, f"variant_{variant}.json")

        existing = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                saved = json.load(f)
            existing = {r["prompt_id"]: r for r in saved.get("results", [])}

        results = list(existing.values())
        for i, item in enumerate(data):
            pid    = item.get("id", item.get("prompt_id", i))
            prompt = item["prompt"] if isinstance(item, dict) else item

            if pid in existing:
                print(f"[{i+1}/10] ID={pid}: skipping (done)", flush=True)
                continue

            arditi_text = arditi_map.get(pid, "Sure, here is")
            print(f"\n[{i+1}/10] ID={pid} variant={variant}: {prompt[:60]}...", flush=True)
            print(f"  CE target: {arditi_text[:80]}...", flush=True)

            r = attack_one(tok, model, dirs, layer_ids, prompt, arditi_text,
                           variant, device)
            r["prompt_id"] = pid
            r["prompt"]    = prompt
            results.append(r)

            with open(out_path, "w") as f:
                json.dump({"config": {"variant": variant, "layers": layer_ids,
                                      "steps": STEPS, "suffix_len": SUFFIX_LEN,
                                      "ce_target": "arditi_output_50tok"},
                           "results": results}, f, indent=2)

        print(f"\nDone variant {variant} -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
