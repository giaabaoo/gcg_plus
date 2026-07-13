"""
GCG Warm-Start Attack
=====================
Runs standard GCG on the original harmful prompt, but initializes the suffix
from the benign-variant suffix (passed via item['suffix_init']) instead of
random tokens.
"""

import nanogcg
import torch


class GCGWarmStartAttack:

    def __init__(self, config):
        self.num_steps    = getattr(config.attack, 'num_steps',    250)
        self.search_width = getattr(config.attack, 'search_width', 512)
        self.suffix_len   = getattr(config.attack, 'suffix_len',   20)
        self.target       = getattr(config.attack, 'target',       'Sure, here is')
        self.seed         = dict.get(config.attack, 'seed')

    def run(self, prompt: str, model, original_prompt: str = None,
            suffix_init: str = None) -> dict:
        messages = [{'role': 'user', 'content': prompt}]

        if suffix_init is None:
            suffix_init = ' '.join(['x'] * self.suffix_len)

        gcg_config = nanogcg.GCGConfig(
            num_steps=self.num_steps,
            search_width=self.search_width,
            optim_str_init=suffix_init,
            verbosity='WARNING',
            seed=self.seed,
        )

        result = nanogcg.run(
            model.model,
            model.tokenizer,
            messages,
            self.target,
            config=gcg_config,
        )

        suffix = result.best_string
        prompt_with_suffix = prompt + ' ' + suffix
        formatted = model.format_prompt(prompt_with_suffix)
        output = model.generate(formatted)

        return {
            'prompt':      prompt,
            'suffix_init': suffix_init,
            'suffix':      suffix,
            'best_loss':   float(result.best_loss),
            'output':      output,
        }
