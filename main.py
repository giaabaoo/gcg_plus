"""
GCG-Plus: Evaluating GCG attack quality with StrongREJECT rubric.

Run:
    python main.py --config configs/base.yml configs/models/llama3_8b.yml \
                            configs/datasets/advbench_50.yml configs/attacks/gcg.yml
"""

import json
import os

from gcg_plus import setup_config, LlamaModel, GCGAttack
from gcg_plus.attack_combined import GCGCombinedAttack
from gcg_plus.dataset import HarmfulPromptsDataset
from gcg_plus.scoring import naive_asr

DATASET_CLASSES = {
    'harmful_prompts': HarmfulPromptsDataset,
}

MODEL_CLASSES = {
    'llama': LlamaModel,
}

ATTACK_CLASSES = {
    'gcg':              GCGAttack,
    'gcg_combined':     GCGCombinedAttack,
    'gcg_refusal_dir':  GCGCombinedAttack,  # Exp15: cosine + CE + CE floor + r_phi select
}


def main():
    config = setup_config()

    dataset_cls = DATASET_CLASSES[config.dataset.dataset_class]
    model_cls   = MODEL_CLASSES[config.model.model_class]
    attack_cls  = ATTACK_CLASSES[getattr(config.attack, 'attack_class', 'gcg')]

    dataset = dataset_cls(config)
    model   = model_cls(config)
    attack  = attack_cls(config)

    out_path = os.path.join(config.output_dir, 'results.json')
    os.makedirs(config.output_dir, exist_ok=True)

    done_ids = set()
    results  = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        results  = saved.get('results', [])
        done_ids = {r['prompt_id'] for r in results}
        print(f'Resuming — {len(done_ids)} prompts already done.')

    total = len(dataset)
    for i, item in enumerate(dataset):
        prompt_id = item['id']
        if prompt_id in done_ids:
            print(f'[{i+1}/{total}] ID={prompt_id}: skipping (done)')
            continue

        prompt = item['prompt']
        print(f'\n[{i+1}/{total}] ID={prompt_id}: {prompt[:80]}...')

        result = attack.run(prompt, model)
        result['prompt_id'] = prompt_id
        result['category']  = item.get('category', '')
        result['naive_asr'] = naive_asr(result['output'])
        results.append(result)

        print(f"  suffix:    {result['suffix']}")
        print(f"  best_loss: {result['best_loss']:.4f}")
        print(f"  naive_asr: {result['naive_asr']}")
        print(f"  output:    {result['output'][:200]}")

        _save(config, results, out_path)

    print(f'\nDone. {len(results)} results -> {out_path}')


def _save(config, results, out_path):
    payload = {
        'config': {
            'model':        config.model.hf_path,
            'dataset':      config.dataset.name,
            'attack':       getattr(config.attack, 'attack_class', 'gcg'),
            'num_steps':    config.attack.num_steps,
            'search_width': config.attack.search_width,
            'suffix_len':   config.attack.suffix_len,
            'target':       dict.get(config.attack, 'target'),
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


if __name__ == '__main__':
    main()
