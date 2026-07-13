"""
Reachability-gap diagnostic for the backward-inversion idea.

For one prompt, at each layer l we ask:
  Can any input reach the ablated target activation h_l* at the suffix positions?

Procedure (l = 31 down to 1):
  1. Target at layer l = ablated activation h_l* (Arditi hook), at suffix positions.
  2. INVERT: optimize a free vector h in R^D (per suffix pos) to minimize
       ||f_{>=l}(h at layer l-1) ... ||  -- approximated locally as
       ||layer_l(h) - h_l*||^2
     starting from the real unablated activation h_{l-1}.
     -> gives inversion residual (how well one transformer layer can be inverted).
  3. REACHABILITY: how far is the inverted target h_{l-1}* from the manifold of
     activations a *real suffix* can produce? Approximate the manifold by the
     unablated activation h_{l-1} and report ||h_{l-1}* - h_{l-1}|| (drift needed).

Outputs per layer:
  - inversion_residual: ||layer_l(h_{l-1}*) - h_l*||   (can the layer be inverted?)
  - reachability_drift: ||h_{l-1}* - h_{l-1}||         (how far off the real manifold?)

If inversion_residual is small but reachability_drift is large and GROWS as we go
backward, that is the reachability gap: targets are satisfiable in activation space
but drift ever further from anything a real input can produce.
"""

import json, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH   = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH    = "data/smoke10_advprefix_targets.json"
DIRECTION_PT = "checkpoints/direction.pt"
OUT_PATH     = "results/reachability_diag.json"
SUFFIX_LEN   = 20
INV_STEPS    = 200
INV_LR       = 1e-2
SEED         = 42

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

    # use first prompt as the diagnostic case
    item   = data[0]
    pid    = item.get("id", item.get("prompt_id", 0))
    prompt = item["prompt"] if isinstance(item, dict) else item
    print(f"Diagnostic prompt ID={pid}: {prompt[:60]}...", flush=True)

    emb = model.get_input_embeddings()
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")
    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device)
    before_len = before_ids.shape[1]
    ss, se = before_len, before_len + SUFFIX_LEN

    pad_emb    = emb(torch.full((1, SUFFIX_LEN), tok.pad_token_id,
                     device=device, dtype=torch.long)).detach()
    before_emb = emb(before_ids).detach()
    after_emb  = emb(after_ids).detach()
    inp = torch.cat([before_emb, pad_emb, after_emb], dim=1)

    # unablated activations
    with torch.no_grad():
        out_orig = model(inputs_embeds=inp, output_hidden_states=True)
    h_orig = [hs[0, ss:se, :].detach().float() for hs in out_orig.hidden_states]  # [L+1][S,D]

    # ablated activations (targets)
    handles = make_hooks(model, r)
    with torch.no_grad():
        out_abl = model(inputs_embeds=inp, output_hidden_states=True)
    remove_hooks(handles)
    h_star = [hs[0, ss:se, :].detach().float() for hs in out_abl.hidden_states]

    layers = model.model.layers
    n_layers = len(layers)

    # position ids + rotary embeddings for the suffix block (fp32 forward)
    pos_ids = torch.arange(ss, se, device=device).unsqueeze(0)
    rotary = model.model.rotary_emb

    def layer_forward_fp32(layer, h):
        """Run one decoder layer in fp32 on activation h [1,S,D]."""
        cos, sin = rotary(h, pos_ids)
        out = layer(h, position_ids=pos_ids,
                    position_embeddings=(cos.float(), sin.float()))
        return out[0] if isinstance(out, tuple) else out

    # cast the layer stack to fp32 for stable inversion
    model.model.rotary_emb = rotary  # unchanged
    diag = []
    for l in range(n_layers, 0, -1):
        layer = layers[l-1].float()                     # fp32 copy of weights
        target = h_star[l].float()                      # [S, D] ablated output of layer l
        h = h_orig[l-1].clone().float().unsqueeze(0).requires_grad_(True)  # [1,S,D]

        opt = torch.optim.AdamW([h], lr=INV_LR)
        for _ in range(INV_STEPS):
            opt.zero_grad()
            y = layer_forward_fp32(layer, h)
            loss = F.mse_loss(y[0].float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([h], 1.0)
            opt.step()

        with torch.no_grad():
            y = layer_forward_fp32(layer, h)
            inv_res  = (y[0].float() - target).norm(dim=-1).mean().item()
            drift    = (h[0].float() - h_orig[l-1]).norm(dim=-1).mean().item()
            orig_nrm = h_orig[l-1].norm(dim=-1).mean().item()
        layers[l-1].half()                              # restore fp16 to save memory

        diag.append({"layer": l, "inversion_residual": inv_res,
                     "reachability_drift": drift,
                     "orig_act_norm": orig_nrm,
                     "drift_ratio": drift / (orig_nrm + 1e-8)})
        if l % 4 == 0 or l == n_layers:
            print(f"  layer {l:2d}: inv_res={inv_res:.3f} drift={drift:.3f} "
                  f"drift/norm={drift/(orig_nrm+1e-8):.3f}", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"prompt_id": pid, "prompt": prompt,
                   "inv_steps": INV_STEPS, "diag": diag}, f, indent=2)
    print(f"\nDone -> {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
