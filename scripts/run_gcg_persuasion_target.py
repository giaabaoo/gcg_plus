"""
Step 2 of the persuasion-target-GCG experiment (see PLAN_persuasion_target_gcg.md).

Plain discrete-GCG (token-swap, greedy coordinate; nanogcg primitives) on the PLAIN
harmful request + a 20-token adversarial suffix, optimizing pure CE toward a chosen
target opener. Identical opt setting to the table's other suffix rows (IRIS / plain
GCG): suffix_len 20, 200 steps, width 64, topk 256, greedy gen, NO activation hook,
one run per prompt (k=1). The ONLY thing that varies across arms is the CE target:

  --arm persuasion : the PAP-mined, model-emitted compliant opener   (test)
  --arm naive      : the AdvPrefix affirmative (existing table baseline)
  --arm neutral    : length + NLL-matched benign filler              (semantics control)

Targets come from data/pap_targets.json (built by scripts/mine_pap_targets.py).
Output schema matches score_with_gemma.py: prompt (=ORIGINAL intent), output, suffix,
target, plus prefix_induced diagnostics (did the suffix actually make greedy decode
emit the target opener? -> separates the "prefix induced but continuation refuses" trap).
"""

import json, os, sys, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanogcg.gcg import sample_ids_from_grad, filter_ids

MODEL_PATH = "NousResearch/Meta-Llama-3.1-8B-Instruct"
TARGETS_PATH = "data/pap_targets.json"

SUFFIX_LEN = 20
TARGET_TOKENS = 20
STEPS = 200
SEARCH_WIDTH = 64
TOPK = 256
CAND_BATCH = 512   # CE-only (no hidden states) -> lighter than the IRIS runs
MAX_NEW = 512
SEED = 42


def build_static(tok, emb, prompt, target_text, device):
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")
    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids = tok([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    tgt_ids = tok(target_text, add_special_tokens=False,
                  return_tensors="pt")["input_ids"][:, :TARGET_TOKENS].to(device)
    return (emb(before_ids).detach(), emb(after_ids).detach(),
            emb(tgt_ids).detach(), tgt_ids)


def run_one(tok, model, emb, prompt, target_text, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    before_emb, after_emb, target_emb, tgt_ids = build_static(tok, emb, prompt, target_text, device)
    tgt_len = tgt_ids.shape[1]
    V = emb.weight.shape[0]

    optim_ids = tok(" ".join(["x"] * SUFFIX_LEN), add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)
    best = {"L": float("inf"), "ids": optim_ids.clone()}

    for step in tqdm(range(STEPS), leave=False):
        onehot = torch.zeros(*optim_ids.shape, V, device=device,
                             dtype=model.dtype).scatter_(-1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)
        inp = torch.cat([before_emb, onehot @ emb.weight, after_emb, target_emb], dim=1)
        logits = model(inputs_embeds=inp).logits
        ce = F.cross_entropy(logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0])
        ce.backward()

        with torch.no_grad():
            sampled = sample_ids_from_grad(optim_ids.squeeze(0), onehot.grad.detach().squeeze(0),
                                           SEARCH_WIDTH, topk=TOPK, n_replace=1, not_allowed_ids=None)
            sampled = filter_ids(sampled, tok)
            n = sampled.shape[0]
            ce_all = torch.empty(n, device=device)
            for s in range(0, n, CAND_BATCH):
                chunk = sampled[s:s + CAND_BATCH]; b = chunk.shape[0]
                inp = torch.cat([before_emb.expand(b, -1, -1), emb(chunk),
                                 after_emb.expand(b, -1, -1), target_emb.expand(b, -1, -1)], dim=1)
                lg = model(inputs_embeds=inp).logits[:, -(tgt_len + 1):-1, :].float()
                ce_all[s:s + b] = F.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]), tgt_ids[0].repeat(b),
                    reduction="none").view(b, tgt_len).mean(1)
            bi = int(ce_all.argmin())
            optim_ids = sampled[bi].unsqueeze(0)
            if ce_all[bi].item() < best["L"]:
                best = {"L": ce_all[bi].item(), "ids": optim_ids.clone()}
        if step % 50 == 0:
            print(f"    step={step} ce={ce_all[bi].item():.4f} best={best['L']:.4f}", flush=True)

    n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    assert n_hooks == 0, f"hook present at gen: {n_hooks}"
    suffix = tok.batch_decode(best["ids"])[0]
    msgs = [{"role": "user", "content": prompt + " " + suffix}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        oids = model.generate(gids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    gen_ids = oids[0][gids.shape[1]:]
    output = tok.decode(gen_ids, skip_special_tokens=True)

    # prefix-induction diagnostic: how much of the target opener did greedy decode reproduce?
    gen_head = gen_ids[:tgt_len].tolist()
    tgt_head = tgt_ids[0].tolist()
    match = sum(1 for a, b in zip(gen_head, tgt_head) if a == b)
    return {
        "suffix": suffix, "output": output, "best_ce": best["L"], "gen_hook_count": n_hooks,
        "target": target_text, "prefix_match_frac": round(match / tgt_len, 3),
        "prefix_induced": match >= int(0.6 * tgt_len),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["persuasion", "naive", "neutral"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or f"results/smoke10_pgcg_{args.arm}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()

    targets = json.load(open(TARGETS_PATH))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    existing = {}
    if os.path.exists(out_path):
        existing = {r["prompt_id"]: r for r in json.load(open(out_path)).get("results", [])}
    results = list(existing.values())

    for i, o in enumerate(targets):
        pid = o["id"]
        if pid in existing:
            print(f"[{i+1}/{len(targets)}] id={pid}: skip (done)", flush=True); continue
        tgt = o[args.arm]["text"]
        print(f"\n[{i+1}/{len(targets)}] id={pid} arm={args.arm}: {o['prompt'][:45]}...\n  TGT: {tgt[:70]}", flush=True)
        r = run_one(tok, model, emb, o["prompt"], tgt, device)
        r["prompt_id"], r["prompt"] = pid, o["prompt"]
        print(f"  best_ce={r['best_ce']:.4f} prefix_match={r['prefix_match_frac']} "
              f"induced={r['prefix_induced']}\n  OUT: {r['output'][:100]}", flush=True)
        results.append(r)
        json.dump({"config": {"method": f"pgcg_{args.arm}", "steps": STEPS,
                              "search_width": SEARCH_WIDTH, "topk": TOPK, "suffix_len": SUFFIX_LEN,
                              "target_tokens": TARGET_TOKENS, "model": MODEL_PATH},
                   "results": results}, open(out_path, "w"), indent=2)
    print(f"\nDone -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
