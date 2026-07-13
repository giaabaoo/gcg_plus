"""
Train r_phi v2: DeBERTa-base + scalar head, BCE loss.

Input format change: tokenizes "prompt [SEP] response" as a pair so DeBERTa
sees both the forbidden prompt and the model response (sentence-pair encoding).
Records with no 'prompt' field fall back to response-only (backwards compat
with old v1 train set entries folded in).

Inputs: data/rphi_train_v2/{train,val}.jsonl
        fields: {prompt: str (optional), response: str, label: 0|1, ...}
Output: checkpoints/rphi_v2/{best,last}.pt and metrics.json

Usage:
    python scripts/train_rphi_v2.py
    python scripts/train_rphi_v2.py --epochs 5 --batch_size 16 --lr 2e-5
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_NAME = 'microsoft/deberta-v3-base'


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ActionabilityDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt   = r.get('prompt') or None
        response = r['response']
        if prompt:
            # DeBERTa sentence-pair: [CLS] prompt [SEP] response [SEP]
            enc = self.tok(
                prompt, response,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors='pt',
            )
        else:
            enc = self.tok(
                response,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors='pt',
            )
        return {
            'input_ids':      enc['input_ids'][0],
            'attention_mask': enc['attention_mask'][0],
            'label':          torch.tensor(r['label'], dtype=torch.float32),
        }


def collate(batch, pad_id):
    maxlen = max(b['input_ids'].size(0) for b in batch)
    ids   = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    mask  = torch.zeros((len(batch), maxlen), dtype=torch.long)
    label = torch.zeros(len(batch), dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b['input_ids'].size(0)
        ids[i, :L]  = b['input_ids']
        mask[i, :L] = b['attention_mask']
        label[i]    = b['label']
    return {'input_ids': ids, 'attention_mask': mask, 'label': label}


class RPhi(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for b in loader:
        logits = model(b['input_ids'].to(device), b['attention_mask'].to(device))
        all_logits.append(logits.cpu()); all_labels.append(b['label'])
    logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits) >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    order = torch.argsort(logits)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(logits) + 1, dtype=torch.float32)
    pos = labels == 1; neg = labels == 0
    n_pos, n_neg = pos.sum().item(), neg.sum().item()
    if n_pos == 0 or n_neg == 0:
        auroc = float('nan')
    else:
        auroc = (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    return {'loss': bce, 'acc': acc, 'auroc': auroc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path',   default='data/rphi_train_v2/train.jsonl')
    ap.add_argument('--val_path',     default='data/rphi_train_v2/val.jsonl')
    ap.add_argument('--out_dir',      default='checkpoints/rphi_v2')
    ap.add_argument('--epochs',       type=int,   default=3)
    ap.add_argument('--batch_size',   type=int,   default=16)
    ap.add_argument('--lr',           type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_frac',  type=float, default=0.1)
    ap.add_argument('--max_len',      type=int,   default=512)
    ap.add_argument('--seed',         type=int,   default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ActionabilityDataset(args.train_path, tok, args.max_len)
    val_ds   = ActionabilityDataset(args.val_path,   tok, args.max_len)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    collate_fn = lambda batch: collate(batch, tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = RPhi().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_frac), total_steps
    )

    os.makedirs(args.out_dir, exist_ok=True)
    history = []
    best_auroc = -1.0
    for ep in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for step, b in enumerate(train_loader):
            logits = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, b['label'].to(device))
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            running += loss.item() * b['label'].size(0); n += b['label'].size(0)
            if (step + 1) % 20 == 0:
                print(f'  ep{ep+1} step{step+1}/{len(train_loader)}  train_loss={running/n:.4f}', flush=True)
        train_loss = running / n
        val = evaluate(model, val_loader, device)
        print(f'epoch {ep+1}: train_loss={train_loss:.4f}  val_loss={val["loss"]:.4f}  '
              f'val_acc={val["acc"]:.4f}  val_auroc={val["auroc"]:.4f}', flush=True)
        history.append({'epoch': ep + 1, 'train_loss': train_loss, **val})
        if val['auroc'] > best_auroc:
            best_auroc = val['auroc']
            torch.save({'state_dict': model.state_dict(), 'val': val, 'args': vars(args)},
                       os.path.join(args.out_dir, 'best.pt'))
            print(f'  -> saved best (auroc={best_auroc:.4f})')

    torch.save({'state_dict': model.state_dict(), 'args': vars(args)},
               os.path.join(args.out_dir, 'last.pt'))
    json.dump(history, open(os.path.join(args.out_dir, 'metrics.json'), 'w'), indent=2)
    print(f'Done. Best val AUROC = {best_auroc:.4f}. Metrics -> {args.out_dir}/metrics.json')


if __name__ == '__main__':
    main()
