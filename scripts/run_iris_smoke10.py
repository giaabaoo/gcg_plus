"""
Faithful IRIS replication (Huang et al., NAACL 2025, "Stronger Universal and
Transferable Attacks by Suppressing Refusals"), scored under Gemma-StrongREJECT.

Fills the empty cell: does IRIS's headline gain survive a strong judge, or was it
LlamaGuard inflation? (IRIS's own eval used LlamaGuard-2 / HarmBench CLS.)

IRIS objective (Eq. 8):
    L_IRIS(x) = -(1-beta) * log p(y | q||x)  +  beta * sum_{h in H} (r_hat^T h)^2

Faithful choices (verbatim from the paper):
  * Optimizer: DISCRETE token-swap GCG (Algorithm 1) -- greedy coordinate, n_replace=1.
  * Term 1 target y: the Arditi activation-edited harmful output (reuse smoke10_arditi),
    truncated to the first TARGET_TOKENS tokens (matches prior discrete-GCG exps = 20).
  * Term 2 penalty set H: "embeddings of the LAST INPUT TOKEN on every layer" ->
    a SINGLE position (last input token, i.e. the token before generation) x all 32 layers.
    (POSITIONS="suffix" reproduces the proposal-literal all-suffix-position reading, which
    is essentially the already-0% exp16 multilayer-cosine variant -- kept for bracketing.)
  * r_hat: a SINGLE refusal vector (Eq. 8 has no layer index). Paper design-choice #1 picks
    the single best layer = layer 10 for Llama-3. We use mean_diffs[0, 10] (pos -5, layer 10),
    RAW diff-in-means (NOT unit-normalized), and log ||r_hat|| for a scale sanity-check.
  * beta = 0.75 (their best) and beta = 1.0 (term-2 only, no CE).

Generation: greedy decode with the final argmax discrete suffix, NO activation hook at
gen time -- the suffix must work on its own.

Model: NousResearch/Meta-Llama-3.1-8B-Instruct (ungated mirror, identical weights to the
gated meta-llama checkpoint; the gated one is not cached and no HF_TOKEN is set here).
"""

import json, os, sys, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanogcg.gcg import sample_ids_from_grad, filter_ids

MODEL_PATH    = "NousResearch/Meta-Llama-3.1-8B-Instruct"
ARDITI_PATH   = "results/smoke10_arditi/results.json"
DATA_PATH     = "data/smoke10_advprefix_targets.json"
MEANDIFF_PATH = "tools/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/generate_directions/mean_diffs.pt"

# --- discrete GCG config: match prior discrete-GCG exps (exp20 / gcg_arditi_target) ---
SUFFIX_LEN    = 20
TARGET_TOKENS = 20
STEPS         = 200
SEARCH_WIDTH  = 64
TOPK          = 256
CAND_BATCH    = 256    # batched candidate eval (hidden_states make this heavier than CE-only)
MAX_NEW       = 512

# --- IRIS refusal vector ---
RVEC_LAYER    = 10     # paper: layer 10 best for Llama-3
RVEC_POS_IDX  = 0      # mean_diffs[0] = post-instruction position -5 (repo convention)
SEED          = 42


def load_rhat(device):
    md = torch.load(MEANDIFF_PATH, map_location="cpu", weights_only=False)  # [5,32,4096] f64
    r = md[RVEC_POS_IDX, RVEC_LAYER].to(device=device, dtype=torch.float32)  # RAW, not normalized
    return r


def build_static(tok, emb, prompt, arditi_text, device):
    messages = [{"role": "user", "content": prompt + "{S}"}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    before_str, after_str = chat.split("{S}")
    before_ids = tok([before_str], return_tensors="pt")["input_ids"].to(device)
    after_ids  = tok([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    tgt_ids    = tok(arditi_text, add_special_tokens=False,
                     return_tensors="pt")["input_ids"][:, :TARGET_TOKENS].to(device)
    return (emb(before_ids).detach(), emb(after_ids).detach(),
            emb(tgt_ids).detach(), tgt_ids, before_ids.shape[1])


def term2_from_hidden(hidden_states, r_hat, tgt_len, before_len, positions):
    """sum_{l, pos in H} (r_hat . h)^2 over all layers. Returns [B] (batched) or scalar-ish."""
    # hidden_states: tuple len 33, each [B, T, D]. h[0]=embeddings; layer l output = hidden_states[l+1].
    acc = None
    for l in range(len(hidden_states) - 1):
        h = hidden_states[l + 1]
        if positions == "last":
            hv = h[:, -(tgt_len + 1), :].float()               # [B, D] last input token
            proj = hv @ r_hat                                   # [B]
            sq = proj * proj                                   # [B]
        else:  # "suffix": all suffix positions
            hv = h[:, before_len:before_len + SUFFIX_LEN, :].float()   # [B, S, D]
            proj = hv @ r_hat                                   # [B, S]
            sq = (proj * proj).sum(dim=1)                       # [B]
        acc = sq if acc is None else acc + sq
    return acc                                                  # [B]


def run_one(tok, model, emb, r_hat, prompt, arditi_text, beta, positions, device,
            steps=STEPS, search_width=SEARCH_WIDTH, topk=TOPK):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    before_emb, after_emb, target_emb, tgt_ids, before_len = build_static(
        tok, emb, prompt, arditi_text, device)
    tgt_len = tgt_ids.shape[1]
    V = emb.weight.shape[0]

    # clean init (repeated "x") -- round-trips through filter_ids, matches prior exp20
    optim_ids = tok(" ".join(["x"] * SUFFIX_LEN), add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)
    best = {"L": float("inf")}

    def full_forward_grad(onehot):
        inp = torch.cat([before_emb, onehot @ emb.weight, after_emb, target_emb], dim=1)
        out = model(inputs_embeds=inp, output_hidden_states=True)
        ce = F.cross_entropy(out.logits[0, -(tgt_len + 1):-1].float(), tgt_ids[0])
        t2 = term2_from_hidden(out.hidden_states, r_hat, tgt_len, before_len, positions)[0]
        return ce, t2

    for step in tqdm(range(steps), leave=False):
        onehot = torch.zeros(*optim_ids.shape, V, device=device,
                             dtype=model.dtype).scatter_(-1, optim_ids.unsqueeze(-1), 1)
        onehot.requires_grad_(True)
        ce, t2 = full_forward_grad(onehot)
        L = (1.0 - beta) * ce + beta * t2
        L.backward()

        with torch.no_grad():
            sampled = sample_ids_from_grad(optim_ids.squeeze(0), onehot.grad.detach().squeeze(0),
                                           search_width, topk=topk, n_replace=1, not_allowed_ids=None)
            sampled = filter_ids(sampled, tok)
            n = sampled.shape[0]
            L_all = torch.empty(n, device=device)
            ce_all = torch.empty(n, device=device)
            t2_all = torch.empty(n, device=device)
            for s in range(0, n, CAND_BATCH):
                chunk = sampled[s:s + CAND_BATCH]; b = chunk.shape[0]
                inp = torch.cat([before_emb.expand(b, -1, -1), emb(chunk),
                                 after_emb.expand(b, -1, -1), target_emb.expand(b, -1, -1)], dim=1)
                out = model(inputs_embeds=inp, output_hidden_states=True)
                logits = out.logits[:, -(tgt_len + 1):-1, :].float()
                ce_c = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       tgt_ids[0].repeat(b), reduction="none").view(b, tgt_len).mean(1)
                t2_c = term2_from_hidden(out.hidden_states, r_hat, tgt_len, before_len, positions)
                ce_all[s:s + b] = ce_c; t2_all[s:s + b] = t2_c
                L_all[s:s + b] = (1.0 - beta) * ce_c + beta * t2_c
            bi = int(L_all.argmin())
            optim_ids = sampled[bi].unsqueeze(0)
            if L_all[bi].item() < best["L"]:
                best = {"L": L_all[bi].item(), "ce": ce_all[bi].item(), "t2": t2_all[bi].item(),
                        "ids": optim_ids.clone()}
        if step % 50 == 0:
            print(f"    step={step} L={L_all[bi].item():.4f} ce={ce_all[bi].item():.4f} "
                  f"t2={t2_all[bi].item():.2f} best_L={best['L']:.4f}", flush=True)

    # ---- generate with final discrete suffix, NO hook ----
    # audit: assert no forward/pre hooks are registered on ANY module at gen time
    n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                  for m in model.modules())
    assert n_hooks == 0, f"activation hook(s) present at gen time: {n_hooks}"
    suffix = tok.batch_decode(best["ids"])[0]
    msgs = [{"role": "user", "content": prompt + " " + suffix}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gids = tok([chat], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        oids = model.generate(gids, max_new_tokens=MAX_NEW, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    output = tok.decode(oids[0][gids.shape[1]:], skip_special_tokens=True)
    return {
        "suffix": suffix, "output": output, "gen_hook_count": n_hooks,
        "best_L": best["L"], "best_ce": best["ce"], "best_t2": best["t2"],
        "term1_weighted": (1.0 - beta) * best["ce"], "term2_weighted": beta * best["t2"],
    }


def run_config(tok, model, emb, r_hat_raw, data, arditi_map, cfg, device):
    beta, positions, out_dir = cfg["beta"], cfg["positions"], cfg["out_dir"]
    steps, sw, tk = cfg["steps"], cfg["search_width"], cfg["topk"]
    normalize = cfg["normalize"]
    r_hat = (r_hat_raw / r_hat_raw.norm()) if normalize else r_hat_raw
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = {r["prompt_id"]: r for r in json.load(f).get("results", [])}
    results = list(existing.values())
    for i, item in enumerate(data):
        pid = item.get("id", item.get("prompt_id", i))
        prompt = item["prompt"]
        if pid in existing:
            print(f"[{i+1}/10] ID={pid}: skip (done)", flush=True); continue
        arditi_text = arditi_map.get(pid, "Sure, here is")
        print(f"\n[{i+1}/10] ID={pid} beta={beta} pos={positions} norm={normalize} "
              f"steps={steps} w={sw}: {prompt[:45]}...", flush=True)
        r = run_one(tok, model, emb, r_hat, prompt, arditi_text, beta, positions, device,
                    steps=steps, search_width=sw, topk=tk)
        r["prompt_id"], r["prompt"] = pid, prompt
        print(f"  best_L={r['best_L']:.4f} ce={r['best_ce']:.4f} t2={r['best_t2']:.2f} "
              f"| term1={r['term1_weighted']:.4f} term2={r['term2_weighted']:.2f}", flush=True)
        print(f"  OUT: {r['output'][:100]}", flush=True)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump({"config": {"method": "iris", "beta": beta, "positions": positions,
                                  "normalize_rhat": normalize, "rvec_layer": RVEC_LAYER,
                                  "rvec_pos_idx": RVEC_POS_IDX, "rvec_norm": float(r_hat.norm()),
                                  "steps": steps, "search_width": sw, "topk": tk,
                                  "suffix_len": SUFFIX_LEN, "target_tokens": TARGET_TOKENS,
                                  "model": MODEL_PATH}, "results": results}, f, indent=2)
    print(f"Done -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="b075,b100",
                    help="comma list of: b075,b100,b075_suffix")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                                 device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()

    r_hat = load_rhat(device)
    print(f"r_hat: layer={RVEC_LAYER} pos_idx={RVEC_POS_IDX} ||r_hat||={r_hat.norm():.4f} "
          f"(normalized variant would be unit; logged for scale sanity-check)", flush=True)

    with open(ARDITI_PATH) as f:
        arditi_map = {r["prompt_id"]: r["output"] for r in json.load(f)["results"]}
    with open(DATA_PATH) as f:
        data = json.load(f)

    def C(beta, positions, out_dir, normalize=False, steps=STEPS, search_width=SEARCH_WIDTH, topk=TOPK):
        return dict(beta=beta, positions=positions, out_dir=out_dir, normalize=normalize,
                    steps=steps, search_width=search_width, topk=topk)
    CONFIGS = {
        # raw r_hat, repo budget (apples-to-apples with existing table rows)
        "b075":        C(0.75, "last",   "results/smoke10_iris_b075"),
        "b100":        C(1.0,  "last",   "results/smoke10_iris_b100"),
        "b075_suffix": C(0.75, "suffix", "results/smoke10_iris_b075_suffix"),
        # normalized r_hat (Arditi unit-vector convention)
        "b075_norm":   C(0.75, "last",   "results/smoke10_iris_b075_norm", normalize=True),
        "b100_norm":   C(1.0,  "last",   "results/smoke10_iris_b100_norm", normalize=True),
        # STRICT paper budget: normalized r_hat + "same settings as GCG": 500 steps, width 512, topk 256
        "b075_paper":  C(0.75, "last",   "results/smoke10_iris_b075_paper", normalize=True,
                         steps=500, search_width=512, topk=256),
        "b100_paper":  C(1.0,  "last",   "results/smoke10_iris_b100_paper", normalize=True,
                         steps=500, search_width=512, topk=256),
    }
    for key in [k.strip() for k in args.runs.split(",") if k.strip()]:
        cfg = CONFIGS[key]
        print(f"\n===== RUN {key}: {cfg} =====", flush=True)
        run_config(tok, model, emb, r_hat, data, arditi_map, cfg, device)


if __name__ == "__main__":
    main()
