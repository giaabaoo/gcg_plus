"""
Compare the Gemma judge against the manual Claude-rubric scores on the same
209 prompts (shards 0+1). Prints aggregate ASR for each judge and per-entry
agreement so we can tell whether Gemma is reliable enough to use going forward.

Usage:
    python scripts/compare_judges.py \
        --claude results/strongreject/scores/shards_0_1_scores.json \
        --gemma  results/strongreject/scores/shards_0_1_gemma.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--claude', required=True)
    ap.add_argument('--gemma',  required=True)
    args = ap.parse_args()

    claude = {r['prompt_id']: r for r in json.load(open(args.claude))}
    gemma  = {r['prompt_id']: r for r in json.load(open(args.gemma))}
    shared = sorted(set(claude) & set(gemma))
    n = len(shared)
    if n == 0:
        print('No overlapping prompt_ids.'); return

    c_asr = sum(claude[i]['asr'] for i in shared)
    g_asr = sum(gemma[i]['asr']  for i in shared)

    # 2x2 confusion on the asr label
    tp = sum(1 for i in shared if claude[i]['asr'] == 1 and gemma[i]['asr'] == 1)
    fn = sum(1 for i in shared if claude[i]['asr'] == 1 and gemma[i]['asr'] == 0)
    fp = sum(1 for i in shared if claude[i]['asr'] == 0 and gemma[i]['asr'] == 1)
    tn = sum(1 for i in shared if claude[i]['asr'] == 0 and gemma[i]['asr'] == 0)
    agree = tp + tn

    print(f'Compared {n} prompts (shards 0+1).')
    print()
    print(f'  Claude rubric ASR : {c_asr}/{n} = {c_asr/n*100:.2f}%')
    print(f'  Gemma  judge  ASR : {g_asr}/{n} = {g_asr/n*100:.2f}%')
    print()
    print(f'Agreement: {agree}/{n} = {agree/n*100:.2f}%')
    print(f'  Both ASR=1 (TP): {tp}')
    print(f'  Both ASR=0 (TN): {tn}')
    print(f'  Claude=1, Gemma=0 (FN, Gemma missed): {fn}')
    print(f'  Claude=0, Gemma=1 (FP, Gemma extra ): {fp}')

    if tp + fn + fp > 0:
        recall    = tp / (tp + fn) if (tp + fn) else float('nan')
        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        print()
        print(f'Treating Claude rubric as ground truth:')
        print(f'  Recall    (Gemma catches Claude positives): {recall:.2f}')
        print(f'  Precision (Gemma positives are real)      : {precision:.2f}')

    # Show the disagreements so the user can eyeball them.
    fn_ids = [i for i in shared if claude[i]['asr'] == 1 and gemma[i]['asr'] == 0]
    fp_ids = [i for i in shared if claude[i]['asr'] == 0 and gemma[i]['asr'] == 1]
    if fn_ids:
        print()
        print(f'Prompt IDs where Gemma missed a Claude positive: {fn_ids}')
    if fp_ids:
        print(f'Prompt IDs where Gemma flagged extra : {fp_ids[:20]}'
              + (f' ... (+{len(fp_ids)-20} more)' if len(fp_ids) > 20 else ''))


if __name__ == '__main__':
    main()
