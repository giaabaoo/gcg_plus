"""
Audit coderchen01/HarmfulGeneration-HarmBench for StrongREJECT positives.

Steps:
1. Download dataset from HuggingFace
2. Apply naive ASR (keyword blacklist) to get a quick count
3. Sample N responses for manual StrongREJECT scoring
4. Save sampled outputs to data/harmbench_audit_sample.json for scoring

Usage:
    python scripts/audit_harmbench_positives.py
    python scripts/audit_harmbench_positives.py --sample_size 50 --seed 42
"""

import argparse
import json
import random

from datasets import load_dataset

from gcg_plus.scoring import naive_asr, REFUSAL_KEYWORDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of responses to sample for manual StrongREJECT scoring')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='data/harmbench_audit_sample.json')
    args = parser.parse_args()

    print('Loading coderchen01/HarmfulGeneration-HarmBench ...')
    ds = load_dataset('coderchen01/HarmfulGeneration-HarmBench', split='test')
    total = len(ds)
    print(f'Total rows: {total}')

    # Naive ASR pass
    naive_pass = [r for r in ds if naive_asr(r['answer']) == 1]
    naive_fail = total - len(naive_pass)
    print(f'\nNaive ASR (keyword blacklist):')
    print(f'  Pass (naive_asr=1): {len(naive_pass)} ({len(naive_pass)/total*100:.1f}%)')
    print(f'  Fail (refusal):     {naive_fail} ({naive_fail/total*100:.1f}%)')

    # Per-category breakdown
    print(f'\nNaive ASR by semantic_category:')
    cats = {}
    for r in ds:
        cat = r['semantic_category']
        cats.setdefault(cat, {'total': 0, 'pass': 0})
        cats[cat]['total'] += 1
        if naive_asr(r['answer']) == 1:
            cats[cat]['pass'] += 1
    for cat, counts in sorted(cats.items()):
        rate = counts['pass'] / counts['total'] * 100
        print(f'  {cat:<35} {counts["pass"]:>5}/{counts["total"]} ({rate:.1f}%)')

    # Sample from naive_pass for manual StrongREJECT scoring
    random.seed(args.seed)
    sample = random.sample(naive_pass, min(args.sample_size, len(naive_pass)))

    out_items = [
        {
            'id':               i,
            'behavior':         r['behavior'],
            'behavior_id':      r['behavior_id'],
            'semantic_category': r['semantic_category'],
            'test_case':        r['test_case'],
            'answer':           r['answer'],
            'naive_asr':        1,
        }
        for i, r in enumerate(sample)
    ]

    with open(args.out, 'w') as f:
        json.dump(out_items, f, indent=2)

    print(f'\nSampled {len(out_items)} naive-pass responses -> {args.out}')
    print('Next: feed data/harmbench_audit_sample.json to Claude with SCORING_INSTRUCTIONS.md')
    print('      to estimate StrongREJECT positive rate in this dataset.')


if __name__ == '__main__':
    main()
