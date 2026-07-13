"""
Run DTA (Dynamic Jailbreaking Attack) on smoke10 prompts.

Uses GPTFuzz as internal judge (local, no API key needed).
Outputs results.json compatible with our Gemma scorer for apples-to-apples
comparison with exp13/14/advprefix.

Usage:
    python scripts/run_dta_smoke10.py --out results/smoke10_dta/results.json
"""

import argparse
import json
import os
import sys

import torch

DTA_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'tools', 'dynamic-jailbreak', 'src')
sys.path.insert(0, DTA_SRC)

from attacker_v3 import DynamicTemperatureAttacker

SMOKE10_PATH  = 'data/smoke10_advprefix_targets.json'
MODEL_PATH    = 'meta-llama/Llama-3.1-8B-Instruct'
GPTFUZZ_PATH  = 'hubert233/GPTFuzz'
NUM_ITERS     = 200   # outer loop (paper uses 2000; 200 is enough for a smoke run)
NUM_INNER     = 10    # inner loop steps per outer iter
SAMPLE_COUNT  = 30    # reference samples per outer iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/smoke10_dta/results.json')
    ap.add_argument('--local-device', default='cuda:0')
    ap.add_argument('--ref-device',   default='cuda:1')
    ap.add_argument('--judge-device', default='cuda:1')
    ap.add_argument('--num-iters',    type=int, default=NUM_ITERS)
    ap.add_argument('--num-inner',    type=int, default=NUM_INNER)
    ap.add_argument('--sample-count', type=int, default=SAMPLE_COUNT)
    args = ap.parse_args()

    with open(SMOKE10_PATH) as f:
        smoke10 = json.load(f)
    prompts = [entry['prompt'] for entry in smoke10]
    prompt_ids = [entry['id'] for entry in smoke10]
    print(f'Loaded {len(prompts)} prompts from {SMOKE10_PATH}')

    print(f'Loading attacker: local={args.local_device} ref={args.ref_device} judge={args.judge_device}')
    attacker = DynamicTemperatureAttacker(
        local_client_name='llama3',
        local_llm_model_name_or_path=MODEL_PATH,
        local_llm_device=args.local_device,
        ref_local_llm_model_name_or_path=MODEL_PATH,
        ref_local_llm_device=args.ref_device,
        judge_llm_model_name_or_path=GPTFUZZ_PATH,
        judge_llm_device=args.judge_device,
        dtype=torch.bfloat16,
        reference_model_infer_temperature=2.0,
        num_ref_infer_samples=args.sample_count,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_jsonl = args.out.replace('.json', '_raw.jsonl')

    print(f'Running DTA: {args.num_iters} outer x {args.num_inner} inner iters ...')
    raw_results = attacker.attack(
        target_set=prompts,
        num_iters=args.num_iters,
        num_inner_iters=args.num_inner,
        learning_rate=1.5,
        response_length=256,
        forward_response_length=20,
        suffix_max_length=20,
        suffix_topk=10,
        mask_rejection_words=True,
        adaptive_sample=True,
        adaptive_sample_max=100,
        adaptive_sample_threshold=0.5,
        use_quality_scoring=False,   # no API judge
        early_stop_threshold=0.6,
        save_path=tmp_jsonl,
        start_index=0,
        end_index=len(prompts),
    )

    # Convert to our results.json schema for Gemma scoring
    results = []
    for entry, pid, raw in zip(smoke10, prompt_ids, raw_results):
        results.append({
            'prompt_id': pid,
            'prompt':    entry['prompt'],
            'suffix':    raw['suffix'],
            'output':    raw['response'],
            'best_unsafe_score': raw['best_unsafe_score'],
        })

    payload = {
        'config': {
            'model':        MODEL_PATH,
            'dataset':      'smoke10',
            'attack':       'dta',
            'judge':        GPTFUZZ_PATH,
            'num_iters':    args.num_iters,
            'num_inner':    args.num_inner,
            'sample_count': args.sample_count,
        },
        'results': results,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'\nDone. Saved -> {args.out}')
    print(f'Raw JSONL  -> {tmp_jsonl}')


if __name__ == '__main__':
    main()
