"""
Attack wrapper for GCG with iterative beam-search target construction.

Config fields (attack.*):
    attack_class:        gcg_beam_target
    num_steps:           250
    search_width:        512
    suffix_len:          20
    target_update_every: 50      (N: reconstruct y* every N steps)
    beam_width:          5
    beam_max_tokens:     64
    rphi_lambda:         1.0
    rphi_ckpt:           checkpoints/rphi_v2/best.pt
    rphi_model_name:     microsoft/deberta-v3-base
"""

import os
import sys

import torch
from transformers import AutoTokenizer

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from train_rphi_v2 import RPhi, MODEL_NAME as DEFAULT_RPHI_MODEL  # noqa: E402

from .gcg_beam_target import GCGBeamTarget, GCGBeamTargetConfig, RPhiScorerV2
from .scoring import naive_asr


class GCGBeamTargetAttack:

    def __init__(self, config):
        a = config.attack
        self.num_steps           = getattr(a, 'num_steps',           250)
        self.search_width        = getattr(a, 'search_width',        512)
        self.suffix_len          = getattr(a, 'suffix_len',          20)
        self.target_update_every = getattr(a, 'target_update_every', 50)
        self.beam_width          = getattr(a, 'beam_width',          5)
        self.beam_max_tokens     = getattr(a, 'beam_max_tokens',     64)
        self.rphi_lambda         = getattr(a, 'rphi_lambda',         1.0)
        self.rphi_ckpt           = a.rphi_ckpt
        self.rphi_model_name     = getattr(a, 'rphi_model_name',     DEFAULT_RPHI_MODEL)
        self._scorer             = None

    def _load_scorer(self, device):
        if self._scorer is not None:
            return self._scorer
        print(f'Loading r_phi v2 tokenizer ({self.rphi_model_name}) ...', flush=True)
        tok = AutoTokenizer.from_pretrained(self.rphi_model_name)
        print(f'Loading r_phi v2 weights ({self.rphi_ckpt}) ...', flush=True)
        model = RPhi(self.rphi_model_name).to(device)
        ckpt  = torch.load(self.rphi_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        self._scorer = RPhiScorerV2(model, tok, device)
        return self._scorer

    def run(self, prompt: str, model, original_prompt: str = None, suffix_init: str = None) -> dict:
        device = model.device
        scorer = self._load_scorer(device)

        cfg = GCGBeamTargetConfig(
            num_steps           = self.num_steps,
            search_width        = self.search_width,
            optim_str_init      = ' '.join(['x'] * self.suffix_len),
            verbosity           = 'WARNING',
            target_update_every = self.target_update_every,
            beam_width          = self.beam_width,
            beam_max_tokens     = self.beam_max_tokens,
            rphi_lambda         = self.rphi_lambda,
            rphi_ckpt           = self.rphi_ckpt,
            rphi_model_name     = self.rphi_model_name,
        )

        gcg = GCGBeamTarget(
            model.model, model.tokenizer, cfg,
            llama_model=model, rphi=scorer,
        )
        messages = [{'role': 'user', 'content': prompt}]
        # target arg is ignored inside GCGBeamTarget.run(); pass placeholder.
        result = gcg.run(messages, target='', prompt_text=prompt)

        suffix             = result.best_string
        prompt_with_suffix = prompt + ' ' + suffix
        formatted          = model.format_prompt(prompt_with_suffix)
        output             = model.generate(formatted)

        return {
            'prompt':        prompt,
            'suffix':        suffix,
            'best_loss':     float(result.best_loss),
            'output':        output,
            'naive_asr':     naive_asr(output),
            'target_texts':  result.target_texts,
            'target_rphis':  [float(r) for r in result.target_rphis],
            'target_steps':  result.target_steps,
        }
