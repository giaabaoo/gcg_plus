"""Attack wrapper for Exp9: r_phi-guided candidate selection."""

import os, sys
import torch
from transformers import AutoTokenizer

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from train_rphi_v2 import RPhi, MODEL_NAME as DEFAULT_RPHI_MODEL

from .gcg_rphi import RPhiScorer
from .attack_rphi_select import GCGRphiSelect, GCGRphiSelectConfig


class GCGRphiSelectAttack:

    def __init__(self, config):
        a = config.attack
        self.num_steps       = dict.get(a, 'num_steps',       250)
        self.search_width    = dict.get(a, 'search_width',    512)
        self.suffix_len      = dict.get(a, 'suffix_len',      20)
        self.target          = dict.get(a, 'target',          'Sure, here is')
        self.rphi_top_k      = dict.get(a, 'rphi_top_k',      16)
        self.rphi_ckpt       = dict.get(a, 'rphi_ckpt',       'checkpoints/rphi_v2/best.pt')
        self.rphi_model_name = dict.get(a, 'rphi_model_name', DEFAULT_RPHI_MODEL)
        self.seed            = dict.get(a, 'seed')
        self._scorer         = None

    def _load_scorer(self, device):
        if self._scorer is not None:
            return self._scorer
        tok   = AutoTokenizer.from_pretrained(self.rphi_model_name)
        model = RPhi(self.rphi_model_name).to(device)
        ckpt  = torch.load(self.rphi_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        self._scorer = RPhiScorer(model, tok, device)
        return self._scorer

    def run(self, prompt: str, model, original_prompt: str = None,
            suffix_init: str = None) -> dict:
        device = model.device
        scorer = self._load_scorer(device)

        cfg = GCGRphiSelectConfig(
            num_steps      = self.num_steps,
            search_width   = self.search_width,
            optim_str_init = ' '.join(['x'] * self.suffix_len),
            verbosity      = 'WARNING',
            seed           = self.seed,
            rphi_top_k     = self.rphi_top_k,
        )

        gcg = GCGRphiSelect(model.model, model.tokenizer, cfg,
                            llama_model=model, rphi_scorer=scorer)

        messages = [{'role': 'user', 'content': prompt}]
        result   = gcg.run(messages, target=self.target, prompt_text=prompt)

        suffix             = result.best_string
        prompt_with_suffix = prompt + ' ' + suffix
        formatted          = model.format_prompt(prompt_with_suffix)
        output             = model.generate(formatted)

        return {
            'prompt':      prompt,
            'suffix':      suffix,
            'best_loss':   float(result.best_loss),
            'output':      output,
            'ce_losses':   [float(x) for x in result.losses],
            'rphi_scores': [float(x) for x in result.rphi_scores],
        }
