"""
Exp18: Continuous GD suffix optimization with Arditi ablation outputs as CE targets.

For each prompt:
- CE target = first 50 tokens of the Arditi ablation response (harmful)
- Optimizer: AdamW on soft suffix logits [suffix_len x vocab]
- No hook, no activation intervention — plain forward pass
- Final suffix = argmax of logits, greedy decode

Hypothesis: a more natural/specific target (actual harmful response) is easier
to optimize toward than "Sure, here is", giving better AASR.
"""

import json, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH    = "meta-llama/Llama-3.1-8B-Instruct"
ARDITI_PATH   = "results/smoke10_arditi/results.json"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
OUT_DIR       = "results/smoke10_arditi_target_gd"
SUFFIX_LEN    = 20
TARGET_TOKENS = 50
STEPS         = 256
LR            = 5e-2
TAU           = 0.1
SEED          = 42
MAX_NEW       = 200

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

    # load Arditi outputs indexed by prompt_id
    with open(ARDITI_PATH) as f:
        arditi = json.load(f)
    arditi_map = {r["prompt_id"]: r["output"] for r in arditi["results"]}

    with open(DATA_PATH) as f:
        data = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")

    # resume
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        existing = {r["prompt_id"]: r for r in saved.get("results", [])}

    emb = model.get_input_embeddings()
    V   = emb.weight.shape[0]
    results = list(existing.values())

    for i, item in enumerate(data):
        pid    = item.get("id", item.get("prompt_id", i)) if isinstance(item, dict) else i
        prompt = item["prompt"] if isinstance(item, dict) else item

        if pid in existing:
            print(f"[{i+1}/10] ID={pid}: skipping (done)", flush=True)
            continue

        print(f"\n[{i+1}/10] ID={pid}: {prompt[:70]}...", flush=True)

        # build CE target from Arditi output — truncate to TARGET_TOKENS
        arditi_text = arditi_map.get(pid, "Sure, here is")
        target_ids_full = tok(arditi_text, add_special_tokens=False,
                              return_tensors="pt")["input_ids"]
        target_ids = target_ids_full[:, :TARGET_TOKENS].to(device)
        tgt_len    = target_ids.shape[1]
        print(f"  target: {tgt_len} tokens | {arditi_text[:60]}...", flush=True)

        # build static embeddings
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        messages = [{"role": "user", "content": prompt + "{S}"}]
        chat = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        if tok.bos_token and chat.startswith(tok.bos_token):
            chat = chat[len(tok.bos_token):]
        before_str, after_str = chat.split("{S}")

        before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
        after_ids  = tok([after_str], add_special_tokens=False,
                         return_tensors="pt")["input_ids"].to(device)

        before_emb = emb(before_ids).detach()
        after_emb  = emb(after_ids).detach()
        target_emb = emb(target_ids).detach()

        # init suffix logits
        suffix_logits = torch.zeros(1, SUFFIX_LEN, V,
                                    device=device, dtype=torch.float32,
                                    requires_grad=True)
        optimizer = torch.optim.AdamW([suffix_logits], lr=LR)

        best_loss = float("inf")
        best_suffix_str = ""

        for step in tqdm(range(STEPS), leave=False):
            optimizer.zero_grad()
            probs      = F.softmax(suffix_logits / TAU, dim=-1)
            suffix_emb = (probs @ emb.weight.float()).to(model.dtype)

            inp = torch.cat([before_emb.float(),
                             suffix_emb,
                             after_emb.float(),
                             target_emb.float()], dim=1).to(model.dtype)
            out = model(inputs_embeds=inp)
            ce  = F.cross_entropy(
                out.logits[0, -(tgt_len + 1):-1].float(), target_ids[0]
            )
            ce.backward()
            optimizer.step()

            if ce.item() < best_loss:
                best_loss = ce.item()
                best_ids  = suffix_logits.detach().argmax(dim=-1)
                best_suffix_str = tok.batch_decode(best_ids)[0]

            if step % 64 == 0:
                print(f"  step={step} ce={ce.item():.4f}", flush=True)

        # final generation with argmax suffix, no hook
        text = prompt + " " + best_suffix_str
        messages = [{"role": "user", "content": text}]
        chat = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok([chat], return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out_ids = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        output = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)

        print(f"  best_loss={best_loss:.4f} | suffix: {best_suffix_str[:60]}", flush=True)
        print(f"  output: {output[:100]}", flush=True)

        results.append({
            "prompt_id": pid, "id": pid, "prompt": prompt,
            "suffix": best_suffix_str, "output": output,
            "best_ce": best_loss,
            "target_preview": arditi_text[:100],
        })

        with open(out_path, "w") as f:
            json.dump({"config": {"steps": STEPS, "suffix_len": SUFFIX_LEN,
                                  "target_tokens": TARGET_TOKENS, "lr": LR,
                                  "tau": TAU, "method": "arditi_target_gd"},
                       "results": results}, f, indent=2)

    print(f"\nDone. {len(results)} results -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
