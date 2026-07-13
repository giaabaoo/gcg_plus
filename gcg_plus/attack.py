"""
GCG Attack
==========
Wraps nanoGCG to optimize a suffix appended to each prompt.

Config fields (attack.*):
    num_steps    : GCG optimization steps (default 250)
    search_width : token candidates per step (default 512)
    suffix_len   : initial suffix token length (default 20)
    target       : target string to maximize (default 'Sure, here is')
"""

import nanogcg
import torch


class GCGAttack:

    def __init__(self, config):
        self.num_steps    = getattr(config.attack, 'num_steps',    250)
        self.search_width = getattr(config.attack, 'search_width', 512)
        self.suffix_len   = getattr(config.attack, 'suffix_len',   20)
        self.target       = getattr(config.attack, 'target',       'Sure, here is')
        self.seed         = config.attack['seed'] if 'seed' in config.attack else None

    def run(self, prompt: str, model, original_prompt: str = None, suffix_init: str = None) -> dict:
        """Optimize a GCG suffix for prompt and return the model output."""
        messages = [{'role': 'user', 'content': prompt}]
        optim_str_init = ' '.join(['x'] * self.suffix_len)
        gcg_config = nanogcg.GCGConfig(
            num_steps=self.num_steps,
            search_width=self.search_width,
            optim_str_init=optim_str_init,
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
            'prompt':    prompt,
            'suffix':    suffix,
            'best_loss': float(result.best_loss),
            'output':    output,
        }
