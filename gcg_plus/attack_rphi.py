"""
GCG+r_phi Attack
================
Wraps the two-phase GCGRphi optimizer + a trained r_phi scorer.

Config fields (attack.*):
    num_steps          : total steps (default 250)
    search_width       : token candidates per step (default 512)
    suffix_len         : initial suffix token length (default 20)
    target             : target string (default 'Sure, here is')
    phase1_steps       : N (default 150)
    phase2_threshold   : best_loss <= T to fire Phase 2 (default 1.0)
    phase2_every_k     : K — sample-and-score cadence in Phase 2 (default 20)
    rphi_lambda        : weighting on REINFORCE term (default 0.3)
    sample_max_tokens  : completion length when sampling for r_phi (default 96)
    sample_temperature : sampling temperature (default 0.7)
    rphi_ckpt          : path to checkpoints/rphi/best.pt
    rphi_model_name    : DeBERTa model name (default microsoft/deberta-v3-base)
"""

import sys
import os

import torch
from transformers import AutoTokenizer

# Make scripts/train_rphi.py importable so we can reuse RPhi + MODEL_NAME.
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from train_rphi import RPhi, MODEL_NAME as DEFAULT_RPHI_MODEL  # noqa: E402

from .gcg_rphi import GCGRphi, GCGRphiConfig, RPhiScorer  # noqa: E402


class GCGRphiAttack:

    def __init__(self, config):
        self.num_steps          = getattr(config.attack, 'num_steps',          250)
        self.search_width       = getattr(config.attack, 'search_width',       512)
        self.suffix_len         = getattr(config.attack, 'suffix_len',         20)
        self.target             = getattr(config.attack, 'target',             'Sure, here is')
        self.phase1_steps       = getattr(config.attack, 'phase1_steps',       150)
        self.phase2_threshold   = getattr(config.attack, 'phase2_threshold',   1.0)
        self.phase2_every_k     = getattr(config.attack, 'phase2_every_k',     20)
        self.rphi_lambda        = getattr(config.attack, 'rphi_lambda',        0.3)
        self.sample_max_tokens  = getattr(config.attack, 'sample_max_tokens',  96)
        self.sample_temperature = getattr(config.attack, 'sample_temperature', 0.7)
        self.rphi_ckpt          = config.attack.rphi_ckpt
        self.rphi_model_name    = getattr(config.attack, 'rphi_model_name',    DEFAULT_RPHI_MODEL)

        # Defer r_phi load until run() — Llama is already on GPU and we need to know the device.
        self._scorer = None

    def _load_scorer(self, device):
        if self._scorer is not None:
            return self._scorer
        print(f'Loading r_phi tokenizer ({self.rphi_model_name}) ...', flush=True)
        tok = AutoTokenizer.from_pretrained(self.rphi_model_name)
        print(f'Loading r_phi weights ({self.rphi_ckpt}) ...', flush=True)
        model = RPhi(self.rphi_model_name).to(device)
        ckpt = torch.load(self.rphi_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        self._scorer = RPhiScorer(model, tok, device)
        return self._scorer

    def run(self, prompt: str, model, original_prompt: str = None, suffix_init: str = None) -> dict:
        device = model.device
        scorer = self._load_scorer(device)

        cfg = GCGRphiConfig(
            num_steps=self.num_steps,
            search_width=self.search_width,
            optim_str_init=' '.join(['x'] * self.suffix_len),
            verbosity='WARNING',
            phase1_steps=self.phase1_steps,
            phase2_threshold=self.phase2_threshold,
            phase2_every_k=self.phase2_every_k,
            rphi_lambda=self.rphi_lambda,
            sample_max_tokens=self.sample_max_tokens,
            sample_temperature=self.sample_temperature,
        )

        gcg = GCGRphi(model.model, model.tokenizer, cfg, llama_model=model, rphi_scorer=scorer)
        messages = [{'role': 'user', 'content': prompt}]
        result = gcg.run(messages, self.target, prompt_text=prompt)

        suffix = result.best_string
        prompt_with_suffix = prompt + ' ' + suffix
        formatted = model.format_prompt(prompt_with_suffix)
        output = model.generate(formatted)

        return {
            'prompt':           prompt,
            'suffix':           suffix,
            'best_loss':        float(result.best_loss),
            'output':           output,
            'phase2_fired':     bool(result.phase2_fired),
            'n_phase2_fires':   int(result.n_phase2_fires),
            'phase1_best_loss': float(result.phase1_best_loss),
            'rewards':          [float(x) for x in result.rewards],
            'advantages':       [float(x) for x in result.advantages],
        }
