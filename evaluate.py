"""
PsychoGraph-Net — evaluation and LIME explainability entry point.

Loads a saved checkpoint, evaluates on the test set, and runs LIME
attribution on the first rapid-cycling positive test case.

Usage
-----
    python evaluate.py                                       # PsychoGraph-Net
    python evaluate.py --model lstm --ckpt results/checkpoints/lstm_best.pt
"""

import argparse
import json
import logging

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, classification_report)
from torch.utils.data import DataLoader

from config import CFG
from data.synthetic_generator import generate_dataset, PsychDataset, SYMPTOM_NAMES
from models import PsychoGraphNet, LSTMBaseline, CNNBaseline
from explainability.lime_explainer import LIMEExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import numpy as np


def make_test_loader(cfg):
    t, nf, adj, y = generate_dataset(
        n_patients=cfg.data.n_patients,
        seq_len=cfg.data.seq_len,
        noise_std=cfg.data.noise_std,
        corr_thresh=cfg.data.corr_thresh,
        seed=cfg.data.seed,
    )
    idx = np.arange(len(y))
    yn  = y.numpy()
    _, itmp = train_test_split(idx, test_size=cfg.data.val_ratio + cfg.data.test_ratio,
                                stratify=yn, random_state=cfg.data.seed)
    half = cfg.data.test_ratio / (cfg.data.val_ratio + cfg.data.test_ratio)
    _, ite = train_test_split(itmp, test_size=half, stratify=yn[itmp],
                               random_state=cfg.data.seed)
    ds = PsychDataset(t[ite], nf[ite], adj[ite], y[ite])
    return DataLoader(ds, batch_size=32), t[ite], nf[ite], adj[ite], y[ite]


def main(args):
    cfg = CFG
    model_map = {'pgn': PsychoGraphNet, 'lstm': LSTMBaseline, 'cnn': CNNBaseline}
    use_graph_map = {'pgn': True, 'lstm': False, 'cnn': False}

    model = model_map[args.model]()
    use_graph = use_graph_map[args.model]

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        log.info("Loaded checkpoint from %s", args.ckpt)
    else:
        log.warning("No checkpoint provided — evaluating with random weights.")

    model.eval()
    loader, t, nf, adj, y = make_test_loader(cfg)

    yp, ypr, yt = [], [], []
    with torch.no_grad():
        for bt, bnf, badj, by in loader:
            out  = model(bt, bnf, badj) if use_graph else model(bt)
            prob = F.softmax(out, -1)[:, 1]
            yp  += out.argmax(1).tolist()
            ypr += prob.tolist()
            yt  += by.tolist()

    yp, ypr, yt = np.array(yp), np.array(ypr), np.array(yt)
    metrics = {
        'accuracy':  round(float(accuracy_score(yt, yp)), 4),
        'precision': round(float(precision_score(yt, yp, average='macro', zero_division=0)), 4),
        'recall':    round(float(recall_score(yt, yp, average='macro', zero_division=0)), 4),
        'f1':        round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(yt, ypr)), 4),
    }

    log.info("\n  Test Metrics")
    log.info("  %-12s %.4f", "Accuracy",  metrics['accuracy'])
    log.info("  %-12s %.4f", "Precision", metrics['precision'])
    log.info("  %-12s %.4f", "Recall",    metrics['recall'])
    log.info("  %-12s %.4f", "F1 (macro)",metrics['f1'])
    log.info("  %-12s %.4f", "AUC-ROC",   metrics['auc'])
    print("\n" + classification_report(yt, yp, target_names=["Non-cycling", "Rapid cycling"]))

    if use_graph:
        log.info("\n  Running LIME attribution on a positive test case...")
        pos_indices = (y == 1).nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            log.warning("No positive cases in test set.")
            return
        idx = pos_indices[0].item()
        explainer = LIMEExplainer(model, use_graph=True, n_perturbations=200)
        result = explainer.explain(t[idx], nf[idx], adj[idx])
        explainer.print_report(result)

        lime_path = cfg.paths.results_dir / "lime_attribution.json"
        with open(lime_path, 'w') as f:
            json.dump({
                'symptom_names': result['symptom_names'],
                'importances': result['importances'].tolist(),
                'base_prob': result['base_prob'],
                'ranked': result['ranked'],
            }, f, indent=2)
        log.info("  LIME results saved → %s", lime_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PsychoGraph-Net")
    parser.add_argument('--model', type=str, default='pgn',
                        choices=['pgn', 'lstm', 'cnn'])
    parser.add_argument('--ckpt',  type=str, default=None,
                        help="Path to checkpoint .pt file")
    main(parser.parse_args())
