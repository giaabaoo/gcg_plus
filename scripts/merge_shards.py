"""
Merge shard results into a single results.json.

Usage:
    python scripts/merge_shards.py --out_dir results/strongreject --num_shards 3
    # reads  results/strongreject/gcg_shard0of3/results.json
    #        results/strongreject/gcg_shard1of3/results.json
    #        results/strongreject/gcg_shard2of3/results.json
    # writes results/strongreject/gcg_merged/results.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--num_shards', type=int, required=True)
    parser.add_argument('--exp_prefix', default='gcg_shard')
    parser.add_argument('--merged_name', default='gcg_merged')
    args = parser.parse_args()

    all_results = []
    config = None
    for i in range(args.num_shards):
        path = os.path.join(args.out_dir, f'{args.exp_prefix}{i}of{args.num_shards}', 'results.json')
        with open(path) as f:
            data = json.load(f)
        if config is None:
            config = data['config']
        all_results.extend(data['results'])

    all_results.sort(key=lambda r: r['prompt_id'])

    merged_dir = os.path.join(args.out_dir, args.merged_name)
    os.makedirs(merged_dir, exist_ok=True)
    out_path = os.path.join(merged_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({'config': config, 'results': all_results}, f, indent=2)

    print(f'Merged {len(all_results)} results from {args.num_shards} shards -> {out_path}')


if __name__ == '__main__':
    main()
