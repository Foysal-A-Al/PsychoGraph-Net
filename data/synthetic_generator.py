"""
Synthetic psychiatric dataset generator.

Simulates patient EHR data for bipolar disorder rapid-cycling prediction.
Each patient has:
    - temporal_data : [seq_len, N_SYMPTOMS]  — longitudinal symptom assessments
    - node_features : [N_SYMPTOMS, 3]        — per-symptom aggregated statistics
    - adj            : [N_SYMPTOMS, N_SYMPTOMS] — normalised symptom co-occurrence graph

Discriminative signal is encoded in BOTH temporal dynamics and graph topology,
reflecting the clinical observation that rapid cyclers exhibit tightly coupled
mood–energy–impulsivity symptom clusters not present in non-cyclers.

In production, replace this generator with Neo4jSymptomGraphLoader (see neo4j_loader.py).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

SYMPTOM_NAMES = [
    "Mood Elevation",       # 0
    "Mood Depression",      # 1
    "Sleep Duration",       # 2
    "Energy Level",         # 3
    "Psychomotor Activity", # 4
    "Irritability",         # 5
    "Racing Thoughts",      # 6
    "Distractibility",      # 7
    "Grandiosity",          # 8
    "Impulsivity",          # 9
    "Anhedonia",            # 10
    "Appetite Change",      # 11
    "Concentration",        # 12
    "Social Withdrawal",    # 13
    "Suicidal Ideation",    # 14
]
N_SYMPTOMS = len(SYMPTOM_NAMES)


def generate_dataset(
    n_patients: int = 600,
    seq_len: int = 12,
    noise_std: float = 1.2,
    corr_thresh: float = 0.22,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic EHR-like dataset.

    Parameters
    ----------
    n_patients  : Total number of simulated patients.
    seq_len     : Number of longitudinal assessment time points.
    noise_std   : Standard deviation of baseline Gaussian noise.
    corr_thresh : Pearson |r| threshold for adjacency edge creation.
    seed        : Random seed for reproducibility.

    Returns
    -------
    temporal_data : FloatTensor [N, seq_len, N_SYMPTOMS]
    node_features : FloatTensor [N, N_SYMPTOMS, 3]
    adj_matrices  : FloatTensor [N, N_SYMPTOMS, N_SYMPTOMS]
    labels        : LongTensor  [N]  — 1 = rapid cycling, 0 = non-cycling
    """
    rng = np.random.RandomState(seed)
    labels = (rng.rand(n_patients) > 0.48).astype(int)

    temporal_data = np.zeros((n_patients, seq_len, N_SYMPTOMS), dtype=np.float32)
    node_features = np.zeros((n_patients, N_SYMPTOMS, 3), dtype=np.float32)
    adj_matrices  = np.zeros((n_patients, N_SYMPTOMS, N_SYMPTOMS), dtype=np.float32)
    t = np.arange(seq_len)

    for i in range(n_patients):
        base = rng.randn(seq_len, N_SYMPTOMS) * noise_std

        if labels[i] == 1:
            # ── Temporal signal (partial learnability by sequential models) ──
            cycle = np.sin(2 * np.pi * t / 3.8) * 0.55
            base[:, 0] += cycle          # Mood Elevation
            base[:, 1] -= cycle * 0.65   # Mood Depression (anti-phase)
            base[:, 3] += cycle * 0.45   # Energy Level
            base[:, 6] += cycle * 0.35   # Racing Thoughts

            # ── Graph-topology signal (shared latent mood factor) ──
            # Tight positive cluster: mood ↔ energy ↔ impulsivity ↔ psychomotor
            shared = rng.randn(seq_len)
            for idx, w in [(0, 1.1), (3, 0.9), (9, 0.85), (4, 0.75)]:
                base[:, idx] += shared * w
            # Negative cluster: depressive symptoms anti-correlate
            for idx, w in [(1, 0.75), (10, 0.65), (13, 0.50)]:
                base[:, idx] -= shared * w

        else:
            # Non-cyclers: slow low-amplitude drift, no tight clusters
            drift = np.sin(2 * np.pi * t * rng.uniform(0.05, 0.10)) * 0.4
            cols = rng.choice(N_SYMPTOMS, 5, replace=False)
            base[:, cols] += drift[:, None]
            noise_fac = rng.randn(seq_len) * 0.3
            for c in rng.choice(N_SYMPTOMS, 7, replace=False):
                base[:, c] += noise_fac * rng.uniform(0.2, 0.45)

        temporal_data[i] = base

        # ── Node features: [μ, σ, range] per symptom ──────────────
        node_features[i] = np.stack(
            [base.mean(0), base.std(0) + 1e-6, base.max(0) - base.min(0)],
            axis=1,
        )

        # ── Adjacency: D^{-1/2} A D^{-1/2} normalised graph ───────
        corr = np.corrcoef(base.T)
        adj  = (np.abs(corr) > corr_thresh).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        deg      = adj.sum(1)
        d_inv_sq = np.where(deg > 0, deg ** -0.5, 0.0)
        D        = np.diag(d_inv_sq)
        adj_matrices[i] = D @ adj @ D

    return (
        torch.FloatTensor(temporal_data),
        torch.FloatTensor(node_features),
        torch.FloatTensor(adj_matrices),
        torch.LongTensor(labels),
    )


class PsychDataset(Dataset):
    """
    PyTorch dataset wrapping the psychiatric patient tensors.

    Parameters
    ----------
    temporal    : FloatTensor [N, seq_len, N_SYMPTOMS]
    node_feats  : FloatTensor [N, N_SYMPTOMS, 3]
    adj         : FloatTensor [N, N_SYMPTOMS, N_SYMPTOMS]
    labels      : LongTensor  [N]
    """

    def __init__(
        self,
        temporal: torch.Tensor,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.temporal   = temporal
        self.node_feats = node_feats
        self.adj        = adj
        self.labels     = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.temporal[idx],
            self.node_feats[idx],
            self.adj[idx],
            self.labels[idx],
        )
