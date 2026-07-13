"""
GCG + Arditi output as CE target (20 tokens), no hook anywhere.
Tests whether a specific harmful target (vs "Sure, here is") improves plain GCG.
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
OUT_DIR       = "results/gcg_arditi_target"
SUFFIX_LEN    = 20
TARGET_TOKENS = 20
STEPS         = 200
SEARCH_WIDTH  = 64
SEED          = 42
MAX_NEW       = 200

def attack_one(tok, model, prompt, arditi_text, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    emb = model.get_input_embeddings()

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

    optim_ids = tok(" ".join(["x"] * SUFFIX_LEN), add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)

    best_loss, best_ids_ever = float("inf"), optim_ids.clone()

    for step in tqdm(range(STEPS), leave=False):
        onehot = torch.zeros(*optim_ids.shape, emb.weight.shape[0],
                             device=device, dtype=model.dtype).scatter_(
                             -1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)

        inp = torch.cat([before_emb, onehot @ emb.weight, after_emb, target_emb], dim=1)
        out = model(inputs_embeds=inp)
        ce  = F.cross_entropy(out.logits[0, -(tgt_len+1):-1].float(), tgt_ids[0])
        ce.backward()

        grad = onehot.grad.detach()
        with torch.no_grad():
            sampled = sample_ids_from_grad(optim_ids.squeeze(0), grad.squeeze(0),
                                           SEARCH_WIDTH, topk=256, n_replace=1,
                                           not_allowed_ids=None)
            sampled = filter_ids(sampled, tok)

            losses = []
            for i in range(sampled.shape[0]):
                inp_i = torch.cat([before_emb, emb(sampled[i].unsqueeze(0)),
                                   after_emb, target_emb], dim=1)
                o_i = model(inputs_embeds=inp_i)
                losses.append(F.cross_entropy(
                    o_i.logits[0, -(tgt_len+1):-1].float(), tgt_ids[0]).item())

            best_idx  = int(torch.tensor(losses).argmin())
            optim_ids = sampled[best_idx].unsqueeze(0)
            if losses[best_idx] < best_loss:
                best_loss     = losses[best_idx]
                best_ids_ever = optim_ids.clone()

        if step % 50 == 0:
            print(f"  step={step} ce={losses[best_idx]:.4f} best={best_loss:.4f}", flush=True)

    suffix = tok.batch_decode(best_ids_ever)[0]
    text = prompt + " " + suffix
    messages = [{"role": "user", "content": text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
    return {"suffix": suffix, "output": output, "best_ce": best_loss}

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
            saved = json.load(f)
        existing = {r["prompt_id"]: r for r in saved.get("results", [])}

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
        res["prompt_id"] = pid
        res["prompt"]    = prompt
        results.append(res)

        with open(out_path, "w") as f:
            json.dump({"config": {"steps": STEPS, "search_width": SEARCH_WIDTH,
                                  "suffix_len": SUFFIX_LEN, "target_tokens": TARGET_TOKENS,
                                  "method": "gcg_arditi_target_no_hook"},
                       "results": results}, f, indent=2)

    print(f"\nDone -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
