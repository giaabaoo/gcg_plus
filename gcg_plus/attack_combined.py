"""
Combined Loss GCG Attack (Exp13/17)

Loss: L(s) = cos(h_16(x ⊕ s), r) + lambda * CE(target | x ⊕ s)
Candidate selection: top-K by combined loss → greedy decode → pick highest r_phi score.
best_string = suffix with highest r_phi score seen across all steps.

Config fields (attack.*):
    num_steps      : GCG steps (default 500)
    search_width   : token candidates per step (default 256)
    suffix_len     : initial suffix length (default 20)
    target         : target string (default 'Sure, here is')
    ce_lambda      : weight for CE term (default 1.0)
    ce_floor       : reject candidates with CE < this (incoherent suffix filter, default 0.0 = off)
    rphi_top_k     : top-K candidates to score with r_phi (default 4)
    direction_path : path to refusal direction .pt (default 'checkpoints/direction.pt')
    hook_layer     : hidden state layer for cosine term (default 16)
    rphi_ckpt      : path to r_phi v2 checkpoint (default 'checkpoints/rphi_v2/best.pt')
    seed           : random seed (default None)
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from nanogcg.gcg import GCG, GCGConfig, GCGResult, sample_ids_from_grad, filter_ids
from nanogcg.utils import find_executable_batch_size
from transformers import AutoTokenizer

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from train_rphi_v2 import RPhi, MODEL_NAME as RPHI_MODEL_NAME


@dataclass
class CombinedGCGConfig(GCGConfig):
    direction_path: str   = 'checkpoints/direction.pt'
    hook_layer:     int   = 16
    ce_lambda:      float = 1.0
    rphi_top_k:     int   = 4


@dataclass
class CombinedGCGResult(GCGResult):
    rphi_scores:   List[float] = field(default_factory=list)
    cosine_losses: List[float] = field(default_factory=list)
    ce_losses:     List[float] = field(default_factory=list)


class GCGCombinedAttack:

    def __init__(self, config):
        self.num_steps     = getattr(config.attack, 'num_steps',      500)
        self.search_width  = getattr(config.attack, 'search_width',   256)
        self.suffix_len    = getattr(config.attack, 'suffix_len',     20)
        self.target        = getattr(config.attack, 'target',         'Sure, here is')
        self.ce_lambda     = getattr(config.attack, 'ce_lambda',      1.0)
        self.ce_floor      = getattr(config.attack, 'ce_floor',       0.0)
        self.rphi_top_k    = getattr(config.attack, 'rphi_top_k',     4)
        self.direction_path = getattr(config.attack, 'direction_path', 'checkpoints/direction.pt')
        self.hook_layer    = getattr(config.attack, 'hook_layer',     16)
        self.rphi_ckpt     = getattr(config.attack, 'rphi_ckpt',      'checkpoints/rphi_v2/best.pt')
        self.seed          = config.attack['seed'] if 'seed' in config.attack else None

    def _load_rphi(self, device):
        tok = AutoTokenizer.from_pretrained(RPHI_MODEL_NAME)
        model = RPhi().to(device)
        ckpt = torch.load(self.rphi_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return tok, model

    @torch.inference_mode()
    def _rphi_score(self, rphi_tok, rphi_model, prompt, response, device):
        enc = rphi_tok(prompt, response, truncation=True, max_length=512,
                       padding=False, return_tensors='pt')
        logit = rphi_model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        return torch.sigmoid(logit).item()

    def _greedy_decode(self, llama_model, prompt, suffix_str):
        prompt_with_suffix = prompt + ' ' + suffix_str
        formatted = llama_model.format_prompt(prompt_with_suffix)
        ids = llama_model.tokenizer(formatted, return_tensors='pt').to(llama_model.device)
        input_len = ids['input_ids'].shape[1]
        with torch.no_grad():
            out = llama_model.model.generate(
                **ids, max_new_tokens=200, do_sample=False,
                pad_token_id=llama_model.tokenizer.eos_token_id,
            )
        return llama_model.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def run(self, prompt: str, llama_model, **kwargs) -> dict:
        device = llama_model.device
        model  = llama_model.model
        tok    = llama_model.tokenizer

        rphi_tok, rphi_model = self._load_rphi(device)

        r = torch.load(self.direction_path, map_location='cpu', weights_only=False)
        if isinstance(r, dict):
            r = next(iter(r.values()))
        r = r.float().to(device)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        messages = [{'role': 'user', 'content': prompt + '{optim_str}'}]
        template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tok.bos_token and template.startswith(tok.bos_token):
            template = template[len(tok.bos_token):]
        before_str, after_str = template.split('{optim_str}')

        target_str = ' ' + self.target
        before_ids = tok([before_str], return_tensors='pt')['input_ids'].to(device, torch.int64)
        after_ids  = tok([after_str], add_special_tokens=False, return_tensors='pt')['input_ids'].to(device, torch.int64)
        target_ids = tok([target_str], add_special_tokens=False, return_tensors='pt')['input_ids'].to(device, torch.int64)

        emb = model.get_input_embeddings()
        before_emb = emb(before_ids).detach()
        after_emb  = emb(after_ids).detach()
        target_emb = emb(target_ids).detach()
        target_len = target_ids.shape[1]

        optim_ids = tok(
            ' '.join(['x'] * self.suffix_len), add_special_tokens=False, return_tensors='pt'
        )['input_ids'].to(device, torch.int64)

        cosine_losses, ce_losses = [], []
        # pool of (loss, ids) across all steps for post-hoc r_phi selection
        step_pool = []

        for step in tqdm(range(self.num_steps)):
            onehot = torch.zeros(*optim_ids.shape, emb.weight.shape[0],
                                  device=device, dtype=model.dtype).scatter_(-1, optim_ids.unsqueeze(-1), 1)
            onehot.requires_grad_(True)

            suffix_emb = onehot @ emb.weight
            inp = torch.cat([before_emb, suffix_emb, after_emb, target_emb], dim=1)
            out = model(inputs_embeds=inp, output_hidden_states=True)

            h        = out.hidden_states[self.hook_layer].float()[:, -(target_len + 1), :]
            cos_term = F.cosine_similarity(h, r.unsqueeze(0)).squeeze()
            ce_term  = F.cross_entropy(out.logits[0, -(target_len + 1):-1], target_ids[0])
            (cos_term + self.ce_lambda * ce_term).backward()

            grad = onehot.grad.detach()
            with torch.no_grad():
                sampled = sample_ids_from_grad(optim_ids.squeeze(0), grad.squeeze(0),
                                               self.search_width, topk=256, n_replace=1,
                                               not_allowed_ids=None)
                sampled = filter_ids(sampled, tok)
                n = sampled.shape[0]

                cos_list, ce_list, comb_list = [], [], []
                for i in range(n):
                    inp_i = torch.cat([before_emb,
                                       emb(sampled[i].unsqueeze(0)),
                                       after_emb, target_emb], dim=1)
                    o_i = model(inputs_embeds=inp_i, output_hidden_states=True)
                    h_i = o_i.hidden_states[self.hook_layer][0, -(target_len + 1), :].float()
                    cos_i = F.cosine_similarity(h_i.unsqueeze(0), r.unsqueeze(0)).squeeze().item()
                    ce_i  = F.cross_entropy(o_i.logits[0, -(target_len + 1):-1], target_ids[0]).item()
                    cos_list.append(cos_i); ce_list.append(ce_i)
                    # CE floor: penalise incoherent suffixes (CE too low = model trivially predicts target)
                    ce_effective = ce_i if ce_i >= self.ce_floor else 1e9
                    comb_list.append(cos_i + self.ce_lambda * ce_effective)

                best_idx = int(torch.tensor(comb_list).argmin())
                optim_ids = sampled[best_idx].unsqueeze(0)
                cosine_losses.append(cos_list[best_idx])
                ce_losses.append(ce_list[best_idx])
                step_pool.append((comb_list[best_idx], optim_ids.clone()))

            print(f'  step={step} cos={cosine_losses[-1]:.4f} ce={ce_losses[-1]:.4f} '
                  f'loss={step_pool[-1][0]:.4f}', flush=True)

        # post-hoc r_phi selection: score top-K by loss from pool
        step_pool.sort(key=lambda x: x[0])
        top_k = min(self.rphi_top_k, len(step_pool))
        best_rphi_ever = -1.0
        best_ids_ever  = step_pool[0][1]
        rphi_scores = []
        for loss_val, cand_ids in step_pool[:top_k]:
            cand_str = tok.batch_decode(cand_ids)[0]
            decoded  = self._greedy_decode(llama_model, prompt, cand_str)
            score    = self._rphi_score(rphi_tok, rphi_model, prompt, decoded, device)
            rphi_scores.append(score)
            if score > best_rphi_ever:
                best_rphi_ever = score
                best_ids_ever  = cand_ids
        print(f'  post-hoc rphi scores: {rphi_scores} → best={best_rphi_ever:.4f}', flush=True)

        best_suffix = tok.batch_decode(best_ids_ever)[0]
        prompt_with_suffix = prompt + ' ' + best_suffix
        output = llama_model.generate(llama_model.format_prompt(prompt_with_suffix))

        return {
            'prompt':      prompt,
            'suffix':      best_suffix,
            'best_loss':   min(comb_list) if comb_list else 0.0,
            'output':      output,
            'rphi_scores': rphi_scores,
        }
