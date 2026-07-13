"""
Run Arditi et al. refusal direction ablation on smoke10 prompts.
Uses their pre-extracted direction.pt for Llama-3-8B-Instruct.
No suffix optimization — pure inference-time intervention.
"""

import json
import os
import sys
import torch

REFUSAL_DIR_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tools', 'refusal_direction')
sys.path.insert(0, REFUSAL_DIR_SRC)

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks

SMOKE10_PATH   = 'data/smoke10_advprefix_targets.json'
MODEL_PATH     = 'meta-llama/Llama-3.1-8B-Instruct'
DIRECTION_PATH = 'tools/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/direction.pt'
OUT_PATH       = 'results/smoke10_arditi/results.json'
MAX_NEW_TOKENS = 256


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(SMOKE10_PATH) as f:
        smoke10 = json.load(f)
    prompts    = [e['prompt'] for e in smoke10]
    prompt_ids = [e['id'] for e in smoke10]
    print(f'Loaded {len(prompts)} prompts')

    print(f'Loading model: {MODEL_PATH}')
    model_base = construct_model_base(MODEL_PATH)

    direction = torch.load(DIRECTION_PATH, map_location='cpu', weights_only=True).float()
    print(f'Direction shape: {direction.shape}')

    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)

    results = []
    for i, (entry, pid, prompt) in enumerate(zip(smoke10, prompt_ids, prompts)):
        print(f'[{i+1}/{len(prompts)}] {prompt[:60]}...')

        # Baseline: no intervention
        inputs = model_base.tokenize_instructions_fn(instructions=[prompt])
        inputs = {k: v.to(model_base.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model_base.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        baseline_response = model_base.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        # Ablation: refusal direction removed
        handles = []
        for module, hook_fn in fwd_pre_hooks:
            handles.append(module.register_forward_pre_hook(hook_fn))
        for module, hook_fn in fwd_hooks:
            handles.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            out = model_base.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        ablation_response = model_base.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        for h in handles:
            h.remove()

        results.append({
            'prompt_id':         pid,
            'prompt':            prompt,
            'suffix':            '',
            'output':            ablation_response,
            'baseline_output':   baseline_response,
        })
        print(f'  baseline: {baseline_response[:80]}')
        print(f'  ablation: {ablation_response[:80]}')

    payload = {
        'config': {
            'model':     MODEL_PATH,
            'dataset':   'smoke10',
            'attack':    'arditi_ablation',
            'direction': DIRECTION_PATH,
        },
        'results': results,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nDone. Saved -> {OUT_PATH}')


if __name__ == '__main__':
    main()
