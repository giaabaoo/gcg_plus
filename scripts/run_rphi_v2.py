"""
Run trained r_phi v2 on results.json files, scoring prompt+response pairs.

Usage:
    python scripts/run_rphi_v2.py \
        --results results/strongreject/gcg_shard0of3/results.json \
                  results/strongreject/gcg_shard1of3/results.json \
                  results/strongreject/gcg_shard2of3/results.json \
        --ckpt   checkpoints/rphi_v2/best.pt \
        --out    results/strongreject/scores/rphi_v2_scores.json
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rphi_v2 import MODEL_NAME, RPhi, collate


def load_entries(paths):
    entries = []
    for p in paths:
        d = json.load(open(p))
        raw = d['results'] if isinstance(d, dict) and 'results' in d else d
        for r in raw:
            entries.append({
                'prompt_id': r.get('prompt_id', r.get('id')),
                'prompt':    r.get('prompt',    r.get('test_case')),
                'response':  r.get('output',    r.get('answer')),
            })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',    nargs='+', required=True)
    ap.add_argument('--ckpt',       default='checkpoints/rphi_v2/best.pt')
    ap.add_argument('--out',        required=True)
    ap.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int,   default=16)
    ap.add_argument('--max_len',    type=int,   default=512)
    ap.add_argument('--threshold',  type=float, default=0.5)
    args = ap.parse_args()

    print(f'Device: {args.device}', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RPhi().to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded {args.ckpt}; val metrics: {ckpt.get("val", "n/a")}', flush=True)

    entries = load_entries(args.results)
    print(f'Scoring {len(entries)} entries from {len(args.results)} file(s)', flush=True)

    def encode(e):
        prompt, response = e.get('prompt') or None, e['response'] or ''
        if prompt:
            enc = tok(prompt, response,
                      truncation=True, max_length=args.max_len,
                      padding=False, return_tensors='pt')
        else:
            enc = tok(response,
                      truncation=True, max_length=args.max_len,
                      padding=False, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'][0],
            'attention_mask': enc['attention_mask'][0],
            'label':          torch.tensor(0.0),
        }

    items = [encode(e) for e in entries]
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    out_rows, n_pos = [], 0
    entry_iter = iter(entries)
    with torch.inference_mode():
        for b in loader:
            logits = model(b['input_ids'].to(args.device), b['attention_mask'].to(args.device))
            probs = torch.sigmoid(logits).cpu().tolist()
            for p in probs:
                e = next(entry_iter)
                label = int(p >= args.threshold)
                n_pos += label
                out_rows.append({
                    'prompt_id': e['prompt_id'],
                    'prob':      round(p, 4),
                    'label':     label,
                })

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out_rows, open(args.out, 'w'), indent=2)
    n = len(out_rows)
    print(f'Wrote {n} scores -> {args.out}')
    print(f'r_phi_v2 positive rate (>= {args.threshold}): {n_pos}/{n} = {n_pos/n*100:.2f}%')


if __name__ == '__main__':
    main()
