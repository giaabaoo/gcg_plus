"""
Exp17: Universal suffix search under Arditi refusal-direction ablation hook.

One suffix optimized jointly over all prompts (average CE loss across batch).
Arditi hook active during every forward pass — projects out refusal direction
from all residual stream activations at every layer.

At eval, apply hook + suffix to each prompt, generate, score with Gemma.
"""

import json, os, sys, contextlib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanogcg.gcg import sample_ids_from_grad, filter_ids

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
DIRECTION_PT  = "checkpoints/direction.pt"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
TARGET        = "Sure, here is"
SUFFIX_LEN    = 20
STEPS         = 500
SEARCH_WIDTH  = 64
SEED          = 42
MAX_NEW       = 200
OUT_DIR       = "results/smoke10_universal_arditi"

# ── Arditi hook helpers ───────────────────────────────────────────────────────
def make_ablation_hooks(model, direction):
    """Returns (pre_hooks, post_hooks) that project out `direction` everywhere."""
    r = direction / (direction.norm() + 1e-8)

    def pre_hook(module, input):
        x = input[0] if isinstance(input, tuple) else input
        rv = r.to(x.dtype)
        x = x - (x @ rv).unsqueeze(-1) * rv
        return (x,) + (input[1:] if isinstance(input, tuple) else ())

    def post_hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        rv = r.to(x.dtype)
        x = x - (x @ rv).unsqueeze(-1) * rv
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    layers = model.model.layers
    pre_handles, post_handles = [], []
    for layer in layers:
        pre_handles.append(layer.register_forward_pre_hook(pre_hook))
        post_handles.append(layer.self_attn.register_forward_hook(post_hook))
        post_handles.append(layer.mlp.register_forward_hook(post_hook))
    return pre_handles + post_handles

@contextlib.contextmanager
def ablation_context(model, direction):
    handles = make_ablation_hooks(model, direction)
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ── build per-prompt static tensors ──────────────────────────────────────────
def build_prompt_tensors(tok, model, prompts, device):
    """Returns list of (before_emb, after_emb, target_ids) per prompt."""
    emb = model.get_input_embeddings()
    target_str = " " + TARGET
    target_ids = tok([target_str], add_special_tokens=False,
                     return_tensors="pt")["input_ids"].to(device)

    tensors = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt + "{S}"}]
        chat = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        if tok.bos_token and chat.startswith(tok.bos_token):
            chat = chat[len(tok.bos_token):]
        before_str, after_str = chat.split("{S}")
        before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
        after_ids  = tok([after_str], add_special_tokens=False,
                         return_tensors="pt")["input_ids"].to(device)
        tensors.append((
            emb(before_ids).detach(),
            emb(after_ids).detach(),
        ))
    return tensors, target_ids

# ── forward: average CE over all prompts ─────────────────────────────────────
def batch_ce_loss(model, prompt_tensors, target_ids, suffix_emb, direction):
    emb_suffix = suffix_emb  # [1, S, D]
    emb_t      = model.get_input_embeddings()(target_ids).detach()
    tgt_len    = target_ids.shape[1]
    total_loss = torch.tensor(0.0, device=suffix_emb.device)

    with ablation_context(model, direction):
        for before_emb, after_emb in prompt_tensors:
            inp = torch.cat([before_emb, emb_suffix, after_emb, emb_t], dim=1)
            out = model(inputs_embeds=inp.to(model.dtype))
            ce  = F.cross_entropy(
                out.logits[0, -(tgt_len + 1):-1].float(), target_ids[0]
            )
            total_loss = total_loss + ce

    return total_loss / len(prompt_tensors)

# ── evaluate candidates (no_grad) ────────────────────────────────────────────
def eval_candidates(model, prompt_tensors, target_ids, sampled, emb, direction):
    """Batch all candidates together per prompt, average CE across prompts."""
    n       = sampled.shape[0]
    emb_t   = emb(target_ids).detach()          # [1, T, D]
    tgt_len = target_ids.shape[1]
    emb_s   = emb(sampled).detach()              # [N, S, D]
    losses  = torch.zeros(n, device=sampled.device)

    with torch.no_grad(), ablation_context(model, direction):
        for before_emb, after_emb in prompt_tensors:
            # expand static parts to batch dim
            b = before_emb.expand(n, -1, -1)     # [N, Lb, D]
            a = after_emb.expand(n, -1, -1)      # [N, La, D]
            t = emb_t.expand(n, -1, -1)          # [N, T, D]
            inp = torch.cat([b, emb_s, a, t], dim=1)  # [N, *, D]
            out = model(inputs_embeds=inp.to(model.dtype))
            # CE per item in batch
            logits = out.logits[:, -(tgt_len + 1):-1, :].float()  # [N, T, V]
            tgt    = target_ids[0].unsqueeze(0).expand(n, -1)      # [N, T]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="none"
            ).reshape(n, tgt_len).mean(dim=1)    # [N]
            losses += ce

    return (losses / len(prompt_tensors)).tolist()

# ── greedy decode under hook ──────────────────────────────────────────────────
def generate_with_hook(tok, model, prompt, suffix_str, direction, device):
    text = prompt + " " + suffix_str
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with ablation_context(model, direction):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    print(f"Loading {MODEL_PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    direction = torch.load(DIRECTION_PT, map_location="cpu", weights_only=False)
    if isinstance(direction, dict):
        direction = next(iter(direction.values()))
    direction = direction.float().to(device)
    direction = direction / (direction.norm() + 1e-8)
    print("Direction loaded, shape:", direction.shape, flush=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    prompts   = [d["prompt"] if isinstance(d, dict) else d for d in data]
    prompt_ids = [d.get("prompt_id", i) if isinstance(d, dict) else i
                  for i, d in enumerate(data)]
    print(f"Prompts: {len(prompts)}", flush=True)

    emb = model.get_input_embeddings()
    prompt_tensors, target_ids = build_prompt_tensors(tok, model, prompts, device)
    tgt_len = target_ids.shape[1]

    # init suffix
    optim_ids = tok(
        " ".join(["x"] * SUFFIX_LEN), add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)

    best_loss_ever = float("inf")
    best_ids_ever  = optim_ids.clone()

    for step in tqdm(range(STEPS)):
        # grad on one-hot to get gradient
        onehot = torch.zeros(*optim_ids.shape, emb.weight.shape[0],
                             device=device, dtype=model.dtype).scatter_(
                             -1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)

        suffix_emb = (onehot.float() @ emb.weight.float()).to(model.dtype)  # [1, S, D]
        loss = batch_ce_loss(model, prompt_tensors, target_ids, suffix_emb, direction)
        loss.backward()

        grad = onehot.grad.detach()

        with torch.no_grad():
            sampled = sample_ids_from_grad(
                optim_ids.squeeze(0), grad.squeeze(0),
                SEARCH_WIDTH, topk=256, n_replace=1, not_allowed_ids=None
            )
            sampled = filter_ids(sampled, tok)

            cand_losses = eval_candidates(model, prompt_tensors, target_ids,
                                          sampled, emb, direction)
            best_idx  = int(torch.tensor(cand_losses).argmin())
            best_loss = cand_losses[best_idx]

            optim_ids = sampled[best_idx].unsqueeze(0)

            if best_loss < best_loss_ever:
                best_loss_ever = best_loss
                best_ids_ever  = optim_ids.clone()

        if step % 50 == 0:
            print(f"  step={step} loss={best_loss:.4f} best_ever={best_loss_ever:.4f}",
                  flush=True)

    best_suffix = tok.batch_decode(best_ids_ever)[0]
    print(f"\nBest suffix: {best_suffix}", flush=True)
    print(f"Best CE loss: {best_loss_ever:.4f}", flush=True)

    # evaluate on all prompts
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i, (prompt, pid) in enumerate(zip(prompts, prompt_ids)):
        print(f"[{i+1}/{len(prompts)}] generating ...", flush=True)
        output = generate_with_hook(tok, model, prompt, best_suffix, direction, device)
        results.append({"prompt_id": pid, "prompt": prompt,
                        "suffix": best_suffix, "output": output})

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"steps": STEPS, "search_width": SEARCH_WIDTH,
                              "suffix_len": SUFFIX_LEN, "target": TARGET,
                              "seed": SEED, "method": "universal_arditi_gcg"},
                   "results": results}, f, indent=2)
    print(f"Saved -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
