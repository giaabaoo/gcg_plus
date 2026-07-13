"""
Sample K completions from Llama-3.1-8B base at high temperature for all smoke prompts.
Takes the best (highest naive_asr) sample per prompt, saves results.json compatible output.
"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcg_plus.scoring import naive_asr

DATA_PATH  = './data/strongreject_313.json'
PROMPT_IDS = [21, 16, 124, 193, 171, 86, 128, 61, 25, 205]
OUT_DIR    = './results/smoke_rphi_v2/base_hightemp'
K          = 10
TEMP       = 1.2
MAX_TOKENS = 300
MODEL      = 'meta-llama/Llama-3.1-8B'

os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)
id_to_item = {item['id']: item for item in data}

print(f'Loading {MODEL} ...', flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()

results = []
for pid in PROMPT_IDS:
    item   = id_to_item[pid]
    prompt = item['prompt']
    print(f'\nPID={pid}: {prompt[:80]}...', flush=True)

    ids       = tok(prompt, return_tensors='pt').to(model.device)
    input_len = ids['input_ids'].shape[1]

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMP,
            num_return_sequences=K,
            pad_token_id=tok.eos_token_id,
        )

    samples = [tok.decode(out[i][input_len:], skip_special_tokens=True) for i in range(K)]
    scores  = [naive_asr(s) for s in samples]

    # pick sample with highest naive_asr; ties broken by first
    best_idx = max(range(K), key=lambda i: scores[i])
    best_out = samples[best_idx]
    best_naive = scores[best_idx]

    print(f'  naive_asr best={best_naive} ({sum(scores)}/{K} samples pass)')
    print(f'  best output: {best_out[:200]}')

    results.append({
        'prompt_id': pid,
        'prompt':    prompt,
        'category':  item.get('category', ''),
        'suffix':    '',
        'best_loss': 0.0,
        'output':    best_out,
        'all_samples': samples,
        'all_naive_asr': scores,
        'naive_asr': best_naive,
    })

payload = {
    'config': {'model': MODEL, 'dataset': 'smoke_rphi_v2', 'attack': 'base_hightemp',
               'num_steps': 0, 'search_width': 0, 'suffix_len': 0,
               'temperature': TEMP, 'num_samples': K},
    'results': results,
}
out_path = os.path.join(OUT_DIR, 'results.json')
with open(out_path, 'w') as f:
    json.dump(payload, f, indent=2)
print(f'\nSaved {len(results)} results -> {out_path}')
