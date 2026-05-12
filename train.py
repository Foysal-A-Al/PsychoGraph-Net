"""
PsychoGraph-Net — training entry point.

Trains PsychoGraph-Net and LSTM/CNN baselines, saves the best checkpoint
for each model, and writes a results JSON to results/.

Usage
-----
    python train.py                          # default config
    python train.py --epochs 150 --lr 5e-4  # override hyperparameters
    python train.py --model pgn              # train one model only
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CFG
from data.synthetic_generator import generate_dataset, PsychDataset
from models import PsychoGraphNet, LSTMBaseline, CNNBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def make_loaders(cfg):
    t, nf, adj, y = generate_dataset(
        n_patients=cfg.data.n_patients,
        seq_len=cfg.data.seq_len,
        noise_std=cfg.data.noise_std,
        corr_thresh=cfg.data.corr_thresh,
        seed=cfg.data.seed,
    )
    yn = y.numpy()
    idx = torch.arange(len(y)).numpy()

    itr, itmp = train_test_split(idx, test_size=cfg.data.val_ratio + cfg.data.test_ratio,
                                  stratify=yn, random_state=cfg.data.seed)
    half = cfg.data.test_ratio / (cfg.data.val_ratio + cfg.data.test_ratio)
    iva, ite = train_test_split(itmp, test_size=half, stratify=yn[itmp],
                                 random_state=cfg.data.seed)

    def mkl(i, shuffle=False):
        ds = PsychDataset(t[i], nf[i], adj[i], y[i])
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    return mkl(itr, True), mkl(iva), mkl(ite), (t, nf, adj, y, ite)


def train_epoch(model, loader, optimizer, criterion, use_graph, grad_clip):
    model.train()
    total = 0.0
    for t, nf, adj, y in loader:
        optimizer.zero_grad()
        out  = model(t, nf, adj) if use_graph else model(t)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, use_graph):
    model.eval()
    yp, ypr, yt = [], [], []
    for t, nf, adj, y in loader:
        out  = model(t, nf, adj) if use_graph else model(t)
        prob = F.softmax(out, -1)[:, 1]
        yp  += out.argmax(1).tolist()
        ypr += prob.tolist()
        yt  += y.tolist()
    import numpy as np
    yp, ypr, yt = map(np.array, [yp, ypr, yt])
    return {
        'accuracy':  float(accuracy_score(yt, yp)),
        'precision': float(precision_score(yt, yp, average='macro', zero_division=0)),
        'recall':    float(recall_score(yt, yp, average='macro', zero_division=0)),
        'f1':        float(f1_score(yt, yp, average='macro', zero_division=0)),
        'auc':       float(roc_auc_score(yt, ypr)),
    }


def fit(model, tr_l, va_l, cfg, use_graph, tag, ckpt_path):
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                              weight_decay=cfg.train.weight_decay)
    sch  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
               opt, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult)
    crit = nn.CrossEntropyLoss()

    best_f1, best_st = 0.0, None
    t0 = time.time()

    for ep in range(1, cfg.train.epochs + 1):
        loss = train_epoch(model, tr_l, opt, crit, use_graph, cfg.train.grad_clip)
        sch.step()
        metrics = evaluate(model, va_l, use_graph)
        f1 = metrics['f1']

        if f1 >= best_f1:
            best_f1 = f1
            best_st = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0:
            log.info(f"[{tag:>18s}] ep {ep:3d}/{cfg.train.epochs} "
                     f"| loss {loss:.4f} | val-F1 {f1:.4f}")

    model.load_state_dict(best_st)
    torch.save({'model_state': best_st, 'val_f1': best_f1}, ckpt_path)
    log.info(f"  Saved checkpoint → {ckpt_path}  (best val-F1: {best_f1:.4f})")
    log.info(f"  Training time: {time.time() - t0:.1f}s")
    return model


def main(args):
    cfg = CFG
    cfg.train.epochs = args.epochs
    cfg.train.lr     = args.lr

    log.info("Generating dataset (%d patients, seq=%d)", cfg.data.n_patients, cfg.data.seq_len)
    tr_l, va_l, te_l, raw = make_loaders(cfg)
    t, nf, adj, y, ite = raw

    models_to_train = {
        'pgn':  (PsychoGraphNet(),   True,  "PsychoGraph-Net"),
        'lstm': (LSTMBaseline(),     False, "LSTM"),
        'cnn':  (CNNBaseline(),      False, "CNN"),
    }
    if args.model != 'all':
        models_to_train = {args.model: models_to_train[args.model]}

    results = {}
    for key, (model, use_graph, tag) in models_to_train.items():
        log.info("\n── %s ──────────────────────────────", tag)
        ckpt = cfg.paths.checkpoints / f"{key}_best.pt"
        model = fit(model, tr_l, va_l, cfg, use_graph, tag, ckpt)
        test_m = evaluate(model, te_l, use_graph)
        results[tag] = {k: round(v, 4) for k, v in test_m.items()}
        log.info("  Test — Acc %.4f | F1 %.4f | AUC %.4f",
                 test_m['accuracy'], test_m['f1'], test_m['auc'])

    out_path = cfg.paths.results_dir / "test_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("\n  Results saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PsychoGraph-Net")
    parser.add_argument('--epochs', type=int,  default=CFG.train.epochs)
    parser.add_argument('--lr',     type=float, default=CFG.train.lr)
    parser.add_argument('--model',  type=str,  default='all',
                        choices=['all', 'pgn', 'lstm', 'cnn'])
    main(parser.parse_args())
