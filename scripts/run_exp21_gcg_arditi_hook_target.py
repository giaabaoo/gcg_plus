"""
Exp21: Discrete GCG under Arditi hook, CE target = first 20 tokens of Arditi output.

Per-prompt attack:
- Arditi ablation hook active during every GCG forward pass (projects out refusal dir)
- CE target = first 20 tokens of the already-generated Arditi harmful response
- Discrete GCG token search (same as standard GCG)
- At inference: hook active + argmax suffix -> greedy decode
"""

import json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanogcg.gcg import sample_ids_from_grad, filter_ids

MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
DIRECTION_PT  = "checkpoints/direction.pt"
ARDITI_PATH   = "results/smoke10_arditi/results.json"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
OUT_DIR       = "results/smoke10_exp21"
SUFFIX_LEN    = 20
TARGET_TOKENS = 20
STEPS         = 200
SEARCH_WIDTH  = 64
SEED          = 42
MAX_NEW       = 200

# ── Arditi hook ───────────────────────────────────────────────────────────────
def make_hooks(model, r):
    def pre_hook(module, input):
        x = input[0] if isinstance(input, tuple) else input
        rv = r.to(x.dtype)
        x = x - (x @ rv).unsqueeze(-1) * rv
        return (x,) + (input[1:] if isinstance(input, tuple) else ())

    def post_hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        rv = r.to(x.dtype)
        x = x - (x @ rv).unsqueeze(-1) * rv
        return (x,) + output[1:] if isinstance(output, tuple) else x

    handles = []
    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.self_attn.register_forward_hook(post_hook))
        handles.append(layer.mlp.register_forward_hook(post_hook))
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()

# ── attack ────────────────────────────────────────────────────────────────────
def attack_one(tok, model, r, prompt, arditi_text, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    emb = model.get_input_embeddings()

    # build static embeddings
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")

    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device)
    tgt_ids    = tok(arditi_text, add_special_tokens=False,
                     return_tensors="pt")["input_ids"][:, :TARGET_TOKENS].to(device)
    tgt_len    = tgt_ids.shape[1]

    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()
    target_emb = emb(tgt_ids).detach()

    # init suffix
    optim_ids = tok(
        " ".join(["x"] * SUFFIX_LEN), add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)

    best_loss_ever = float("inf")
    best_ids_ever  = optim_ids.clone()

    for step in tqdm(range(STEPS), leave=False):
        onehot = torch.zeros(*optim_ids.shape, emb.weight.shape[0],
                             device=device, dtype=model.dtype).scatter_(
                             -1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)

        suffix_emb = onehot @ emb.weight

        inp = torch.cat([before_emb, suffix_emb, after_emb, target_emb], dim=1)

        # forward with hook active
        handles = make_hooks(model, r)
        out = model(inputs_embeds=inp, output_hidden_states=False)
        remove_hooks(handles)

        ce = F.cross_entropy(out.logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0])
        ce.backward()

        grad = onehot.grad.detach()
        with torch.no_grad():
            sampled = sample_ids_from_grad(
                optim_ids.squeeze(0), grad.squeeze(0),
                SEARCH_WIDTH, topk=256, n_replace=1, not_allowed_ids=None
            )
            sampled = filter_ids(sampled, tok)
            n = sampled.shape[0]

            # eval candidates with hook
            losses = []
            handles = make_hooks(model, r)
            for i in range(n):
                inp_i = torch.cat([before_emb,
                                   emb(sampled[i].unsqueeze(0)),
                                   after_emb, target_emb], dim=1)
                o_i = model(inputs_embeds=inp_i)
                ce_i = F.cross_entropy(
                    o_i.logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0]
                ).item()
                losses.append(ce_i)
            remove_hooks(handles)

            best_idx = int(torch.tensor(losses).argmin())
            optim_ids = sampled[best_idx].unsqueeze(0)

            if losses[best_idx] < best_loss_ever:
                best_loss_ever = losses[best_idx]
                best_ids_ever  = optim_ids.clone()

        if step % 50 == 0:
            print(f"  step={step} ce={losses[best_idx]:.4f} best={best_loss_ever:.4f}",
                  flush=True)

    best_suffix = tok.batch_decode(best_ids_ever)[0]

    # generate with hook active
    text = prompt + " " + best_suffix
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    handles = make_hooks(model, r)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    remove_hooks(handles)
    output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)

    return {"suffix": best_suffix, "output": output, "best_ce": best_loss_ever}

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    print(f"Loading {MODEL_PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    r = torch.load(DIRECTION_PT, map_location="cpu", weights_only=False)
    if isinstance(r, dict):
        r = next(iter(r.values()))
    r = (r.float() / (r.norm() + 1e-8)).to(device)
    print("Direction shape:", r.shape, flush=True)

    with open(ARDITI_PATH) as f:
        arditi = json.load(f)
    arditi_map = {x["prompt_id"]: x["output"] for x in arditi["results"]}

    with open(DATA_PATH) as f:
        data = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        existing = {r_["prompt_id"]: r_ for r_ in saved.get("results", [])}

    results = list(existing.values())
    for i, item in enumerate(data):
        pid    = item.get("id", item.get("prompt_id", i))
        prompt = item["prompt"] if isinstance(item, dict) else item

        if pid in existing:
            print(f"[{i+1}/10] ID={pid}: skipping", flush=True)
            continue

        arditi_text = arditi_map.get(pid, "Sure, here is")
        print(f"\n[{i+1}/10] ID={pid}: {prompt[:60]}...", flush=True)
        print(f"  target: {arditi_text[:80]}...", flush=True)

        res = attack_one(tok, model, r, prompt, arditi_text, device)
        res["prompt_id"] = pid
        res["prompt"]    = prompt
        results.append(res)

        with open(out_path, "w") as f:
            json.dump({"config": {"steps": STEPS, "search_width": SEARCH_WIDTH,
                                  "suffix_len": SUFFIX_LEN, "target_tokens": TARGET_TOKENS,
                                  "method": "gcg_arditi_hook_arditi_target"},
                       "results": results}, f, indent=2)

    print(f"\nDone -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
