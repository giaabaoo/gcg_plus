"""
Activation matching: optimize suffix via continuous GD to match Arditi ablated
activations at suffix positions across all 32 layers simultaneously.
No hook at inference time.
"""

import json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH   = "meta-llama/Llama-3.1-8B-Instruct"
ARDITI_PATH  = "results/smoke10_arditi/results.json"
DATA_PATH    = "data/smoke10_advprefix_targets.json"
DIRECTION_PT = "checkpoints/direction.pt"
OUT_DIR      = "results/act_match"
SUFFIX_LEN   = 20
STEPS        = 256
LR           = 5e-2
TAU          = 0.1
SEED         = 42
MAX_NEW      = 200

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

def attack_one(tok, model, r, prompt, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    emb = model.get_input_embeddings()
    V   = emb.weight.shape[0]

    # build prompt split around suffix placeholder
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")

    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device)
    before_len = before_ids.shape[1]
    suffix_start = before_len
    suffix_end   = before_len + SUFFIX_LEN

    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()

    # ── Step 1: run WITH hook, record ablated activations at suffix positions ──
    # use zero suffix (embedding of pad token) as placeholder — we only need
    # the ablated activations, which are dominated by the prompt anyway
    pad_emb = emb(torch.full((1, SUFFIX_LEN), tok.pad_token_id,
                              device=device, dtype=torch.long)).detach()
    inp_ref = torch.cat([before_emb, pad_emb, after_emb], dim=1)

    handles = make_hooks(model, r)
    with torch.no_grad():
        out_ref = model(inputs_embeds=inp_ref, output_hidden_states=True)
    remove_hooks(handles)

    # h_star: list of [SUFFIX_LEN, D] tensors, one per layer (layers 1..32)
    h_star = [out_ref.hidden_states[l][0, suffix_start:suffix_end, :].detach()
              for l in range(1, len(out_ref.hidden_states))]
    n_layers = len(h_star)

    # per-layer MSE before optimization (baseline)
    with torch.no_grad():
        out_base = model(inputs_embeds=inp_ref, output_hidden_states=True)
    pre_mse = [F.mse_loss(out_base.hidden_states[l][0, suffix_start:suffix_end, :].float(),
                           h_star[l-1].float()).item()
               for l in range(1, n_layers+1)]

    # ── Step 2: optimize suffix to match h_star ──
    suffix_logits = torch.zeros(1, SUFFIX_LEN, V, device=device,
                                dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([suffix_logits], lr=LR)

    best_loss, best_suffix_str = float("inf"), ""

    for step in tqdm(range(STEPS), leave=False):
        optimizer.zero_grad()

        probs      = F.softmax(suffix_logits / TAU, dim=-1)
        suffix_emb = (probs @ emb.weight.float()).to(model.dtype)

        inp = torch.cat([before_emb, suffix_emb, after_emb], dim=1)
        out = model(inputs_embeds=inp, output_hidden_states=True)

        act_loss = torch.tensor(0.0, device=device)
        for l in range(1, n_layers + 1):
            h_l = out.hidden_states[l][0, suffix_start:suffix_end, :].float()
            act_loss = act_loss + F.mse_loss(h_l, h_star[l-1].float())

        act_loss.backward()
        optimizer.step()

        if act_loss.item() < best_loss:
            best_loss = act_loss.item()
            best_ids  = suffix_logits.detach().argmax(dim=-1)
            best_suffix_str = tok.batch_decode(best_ids)[0]

        if step % 64 == 0:
            print(f"  step={step} act_loss={act_loss.item():.4f}", flush=True)

    # per-layer MSE after optimization
    best_suffix_ids = suffix_logits.detach().argmax(dim=-1)
    best_suffix_emb = emb(best_suffix_ids).detach()
    inp_final = torch.cat([before_emb, best_suffix_emb, after_emb], dim=1)
    with torch.no_grad():
        out_final = model(inputs_embeds=inp_final, output_hidden_states=True)
    post_mse = [F.mse_loss(out_final.hidden_states[l][0, suffix_start:suffix_end, :].float(),
                            h_star[l-1].float()).item()
                for l in range(1, n_layers+1)]

    # ── Step 3: greedy decode without hook ──
    text = prompt + " " + best_suffix_str
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)

    return {
        "suffix": best_suffix_str,
        "output": output,
        "best_act_loss": best_loss,
        "pre_mse_mean":  sum(pre_mse)  / n_layers,
        "post_mse_mean": sum(post_mse) / n_layers,
        "pre_mse_per_layer":  pre_mse,
        "post_mse_per_layer": post_mse,
    }

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    r = torch.load(DIRECTION_PT, map_location="cpu", weights_only=False)
    if isinstance(r, dict):
        r = next(iter(r.values()))
    r = (r.float() / (r.norm() + 1e-8)).to(device)

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

        print(f"\n[{i+1}/10] ID={pid}: {prompt[:60]}...", flush=True)
        res = attack_one(tok, model, r, prompt, device)
        res["prompt_id"] = pid
        res["prompt"]    = prompt

        print(f"  pre_mse={res['pre_mse_mean']:.4f} post_mse={res['post_mse_mean']:.4f}", flush=True)
        print(f"  output: {res['output'][:100]}", flush=True)

        results.append(res)
        with open(out_path, "w") as f:
            json.dump({"config": {"steps": STEPS, "suffix_len": SUFFIX_LEN,
                                  "method": "act_match_all_layers"},
                       "results": results}, f, indent=2)

    print(f"\nDone -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
