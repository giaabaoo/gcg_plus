"""
Full-budget direct GCG: CE toward Arditi harmful target, no hook.
Larger budget to find the real ceiling of a suffix-only attack.

vs previous gcg_arditi_target: more steps, wider search, longer suffix, multi-restart.
"""

import json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanogcg.gcg import sample_ids_from_grad, filter_ids

MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
ARDITI_PATH   = "results/smoke10_arditi/results.json"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
OUT_DIR       = "results/gcg_full_budget"
SUFFIX_LEN    = 40      # was 20
TARGET_TOKENS = 20
STEPS         = 500     # was 200
SEARCH_WIDTH  = 256     # was 64
TOPK          = 256
N_RESTARTS    = 1       # single init (no restart)
MAX_NEW       = 200
CAND_BATCH    = 64      # candidates per forward batch

def build_static(tok, model, emb, prompt, arditi_text, device):
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
    return (emb(before_ids).detach(), emb(after_ids).detach(),
            emb(tgt_ids).detach(), tgt_ids)

def eval_candidates(model, emb, before_emb, after_emb, target_emb, tgt_ids, sampled):
    """Batched CE over candidate suffixes."""
    tgt_len = tgt_ids.shape[1]
    n = sampled.shape[0]
    losses = torch.empty(n, device=sampled.device)
    for s in range(0, n, CAND_BATCH):
        chunk = sampled[s:s+CAND_BATCH]
        b = chunk.shape[0]
        suffix_emb = emb(chunk)                                   # [b, S, D]
        inp = torch.cat([before_emb.expand(b, -1, -1), suffix_emb,
                         after_emb.expand(b, -1, -1),
                         target_emb.expand(b, -1, -1)], dim=1)
        out = model(inputs_embeds=inp)
        logits = out.logits[:, -(tgt_len+1):-1, :].float()       # [b, tgt_len, V]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt_ids[0].repeat(b), reduction="none").view(b, tgt_len).mean(1)
        losses[s:s+b] = ce
    return losses

def run_restart(tok, model, emb, before_emb, after_emb, target_emb, tgt_ids,
                seed, device):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tgt_len = tgt_ids.shape[1]
    V = emb.weight.shape[0]

    optim_ids = torch.randint(0, V, (1, SUFFIX_LEN), device=device)
    best_loss, best_ids = float("inf"), optim_ids.clone()

    for step in tqdm(range(STEPS), leave=False):
        onehot = torch.zeros(*optim_ids.shape, V, device=device,
                             dtype=model.dtype).scatter_(-1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)
        inp = torch.cat([before_emb, onehot @ emb.weight, after_emb, target_emb], dim=1)
        out = model(inputs_embeds=inp)
        ce  = F.cross_entropy(out.logits[0, -(tgt_len+1):-1].float(), tgt_ids[0])
        ce.backward()

        with torch.no_grad():
            sampled = sample_ids_from_grad(optim_ids.squeeze(0), onehot.grad.detach().squeeze(0),
                                           SEARCH_WIDTH, topk=TOPK, n_replace=1,
                                           not_allowed_ids=None)
            sampled = filter_ids(sampled, tok)
            losses  = eval_candidates(model, emb, before_emb, after_emb,
                                      target_emb, tgt_ids, sampled)
            bi = int(losses.argmin())
            optim_ids = sampled[bi].unsqueeze(0)
            if losses[bi].item() < best_loss:
                best_loss = losses[bi].item()
                best_ids  = optim_ids.clone()

        if step % 100 == 0:
            print(f"    step={step} ce={losses[bi].item():.4f} best={best_loss:.4f}", flush=True)

    return best_loss, best_ids

def attack_one(tok, model, prompt, arditi_text, device):
    emb = model.get_input_embeddings()
    before_emb, after_emb, target_emb, tgt_ids = build_static(
        tok, model, emb, prompt, arditi_text, device)

    global_best_loss, global_best_ids = float("inf"), None
    for r in range(N_RESTARTS):
        print(f"  restart {r+1}/{N_RESTARTS}", flush=True)
        loss, ids = run_restart(tok, model, emb, before_emb, after_emb,
                                target_emb, tgt_ids, seed=42 + r, device=device)
        if loss < global_best_loss:
            global_best_loss, global_best_ids = loss, ids

    suffix = tok.batch_decode(global_best_ids)[0]
    text = prompt + " " + suffix
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
    return {"suffix": suffix, "output": output, "best_ce": global_best_loss}

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

    with open(ARDITI_PATH) as f:
        arditi_map = {r["prompt_id"]: r["output"] for r in json.load(f)["results"]}
    with open(DATA_PATH) as f:
        data = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = {r["prompt_id"]: r for r in json.load(f).get("results", [])}

    results = list(existing.values())
    for i, item in enumerate(data):
        pid    = item.get("id", item.get("prompt_id", i))
        prompt = item["prompt"] if isinstance(item, dict) else item
        if pid in existing:
            print(f"[{i+1}/10] ID={pid}: skipping", flush=True)
            continue

        arditi_text = arditi_map.get(pid, "Sure, here is")
        print(f"\n[{i+1}/10] ID={pid}: {prompt[:60]}...", flush=True)
        res = attack_one(tok, model, prompt, arditi_text, device)
        res["prompt_id"], res["prompt"] = pid, prompt
        print(f"  best_ce={res['best_ce']:.4f} | {res['output'][:90]}", flush=True)

        results.append(res)
        with open(out_path, "w") as f:
            json.dump({"config": {"steps": STEPS, "search_width": SEARCH_WIDTH,
                                  "suffix_len": SUFFIX_LEN, "n_restarts": N_RESTARTS,
                                  "target_tokens": TARGET_TOKENS,
                                  "method": "gcg_full_budget_no_hook"},
                       "results": results}, f, indent=2)

    print(f"\nDone -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
