"""
Run GCG with AdvPrefix paper's pre-generated per-prompt targets on the smoke10 prompts.

Each prompt gets its own target string from the advprefix CSV instead of the
generic "Sure, here is" — these targets were selected by the paper to have high
PASR on Llama-3.1-8B-Instruct.

Usage:
    python scripts/run_advprefix_gcg.py --out results/advbench_50/smoke10_advprefix_gcg/results.json
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import nanogcg
from transformers import AutoModelForCausalLM, AutoTokenizer
from gcg_plus.scoring import naive_asr

MODEL_PATH     = 'meta-llama/Llama-3.1-8B-Instruct'
TARGETS_PATH   = 'data/smoke10_advprefix_targets.json'
SEARCH_WIDTH   = 512
SUFFIX_LEN     = 20
SEED           = 42
MAX_NEW_TOKENS = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/advbench_50/smoke10_advprefix_gcg/results.json')
    ap.add_argument('--num_steps', type=int, default=1000)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    print(f'Loading {MODEL_PATH} ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto',
        attn_implementation='eager')
    model.eval()
    device = next(model.parameters()).device
    print(f'Model on {device}')

    all_targets = json.load(open(TARGETS_PATH))
    targets = [t for i, t in enumerate(all_targets) if i % args.num_shards == args.shard]
    print(f'Shard {args.shard}/{args.num_shards}: running GCG on {len(targets)}/{len(all_targets)} prompts, {args.num_steps} steps ...')

    results = []
    for i, entry in enumerate(targets):
        pid    = entry['id']
        prompt = entry['prompt']
        target = entry['target']

        print(f'\n[{i+1}/{len(targets)}] ID={pid}')
        print(f'  prompt: {prompt[:80]}')
        print(f'  target: {repr(target[:60])}')

        messages = [{'role': 'user', 'content': prompt}]
        gcg_config = nanogcg.GCGConfig(
            num_steps=args.num_steps,
            search_width=SEARCH_WIDTH,
            optim_str_init=' '.join(['x'] * SUFFIX_LEN),
            verbosity='WARNING',
            seed=SEED,
        )
        result = nanogcg.run(model, tokenizer, messages, target, config=gcg_config)
        suffix = result.best_string

        # Generate output
        prompt_with_suffix = prompt + ' ' + suffix
        msgs = [{'role': 'user', 'content': prompt_with_suffix}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(formatted, return_tensors='pt').to(device)
        input_len = ids['input_ids'].shape[1]
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        asr = naive_asr(output)

        print(f'  best_loss={result.best_loss:.4f} naive_asr={asr}')
        print(f'  output: {output[:120]}')

        results.append({
            'prompt_id': pid,
            'prompt':    prompt,
            'target':    target,
            'suffix':    suffix,
            'best_loss': float(result.best_loss),
            'output':    output,
            'naive_asr': asr,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        'config': {
            'model': MODEL_PATH,
            'dataset': 'advbench_50_smoke10',
            'attack': 'gcg_advprefix_targets',
            'num_steps': args.num_steps,
            'search_width': SEARCH_WIDTH,
            'suffix_len': SUFFIX_LEN,
            'seed': SEED,
        },
        'results': results,
    }
    json.dump(payload, open(args.out, 'w'), indent=2)

    passed = sum(1 for r in results if r['naive_asr'] == 1)
    print(f'\nDone. Naive ASR: {passed}/{len(results)} = {passed/len(results)*100:.1f}%')
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
