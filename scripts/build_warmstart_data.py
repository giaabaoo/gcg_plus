"""
Build warm-start dataset from Exp4 (benign variant) results.

Reads exp4 results.json, extracts per-prompt suffix, and writes a JSON file
where each entry has the original harmful prompt + the benign-variant suffix
as optim_str_init for GCG warm-start.

Usage:
    python scripts/build_warmstart_data.py \
        --exp4    results/smoke_rphi_v2/exp4_transfer/results.json \
        --benign  data/benign_variants.json \
        --out     data/warmstart_init.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp4',   required=True)
    ap.add_argument('--benign', required=True)
    ap.add_argument('--out',    required=True)
    args = ap.parse_args()

    exp4 = {r['prompt_id']: r for r in json.load(open(args.exp4))['results']}
    benign = {r['id']: r for r in json.load(open(args.benign))}

    out = []
    for pid, r in sorted(exp4.items()):
        b = benign.get(pid, {})
        out.append({
            'id':             pid,
            'prompt':         r['prompt'],          # original harmful prompt
            'benign_prompt':  r.get('benign_prompt', b.get('benign', '')),
            'suffix_init':    r['suffix'],           # benign-variant suffix → warm-start
            'category':       r.get('category', b.get('category', '')),
        })

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {len(out)} entries -> {args.out}')


if __name__ == '__main__':
    main()
