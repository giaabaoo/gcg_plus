"""
GCG Transfer Attack
===================
Optimizes suffix on a benign variant of the prompt, then appends it to
the original harmful prompt for generation and scoring.
"""

import nanogcg
import torch


class GCGTransferAttack:

    def __init__(self, config):
        self.num_steps    = getattr(config.attack, 'num_steps',    250)
        self.search_width = getattr(config.attack, 'search_width', 512)
        self.suffix_len   = getattr(config.attack, 'suffix_len',   20)
        self.target       = getattr(config.attack, 'target',       'Sure, here is')
        self.seed         = dict.get(config.attack, 'seed')

    def run(self, prompt: str, model, original_prompt: str = None, suffix_init: str = None) -> dict:
        """
        prompt          : benign variant (used for GCG optimization)
        original_prompt : original harmful prompt (used for final generation)
                          falls back to prompt if not provided
        """
        if original_prompt is None:
            original_prompt = prompt

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

        # Generate on the ORIGINAL harmful prompt + optimized suffix
        prompt_with_suffix = original_prompt + ' ' + suffix
        formatted = model.format_prompt(prompt_with_suffix)
        output = model.generate(formatted)

        return {
            'prompt':          original_prompt,
            'benign_prompt':   prompt,
            'suffix':          suffix,
            'best_loss':       float(result.best_loss),
            'output':          output,
        }
