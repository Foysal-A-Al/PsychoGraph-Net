"""
PsychoGraph-Net — Central configuration.
All hyperparameters and paths defined here; train.py / evaluate.py import this.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class DataConfig:
    n_patients:   int   = 600
    seq_len:      int   = 12
    n_symptoms:   int   = 15
    noise_std:    float = 1.2
    corr_thresh:  float = 0.22      # adjacency edge threshold
    seed:         int   = 42
    val_ratio:    float = 0.15
    test_ratio:   float = 0.15


@dataclass
class ModelConfig:
    # GNN
    gnn_in_dim:   int   = 3         # node feature dim [mean, std, range]
    gnn_hidden:   int   = 48
    gnn_out_dim:  int   = 64
    # Temporal Transformer
    d_model:      int   = 64
    n_heads:      int   = 4
    n_layers:     int   = 2
    trm_dropout:  float = 0.15
    # Fusion head
    fusion_hidden: int  = 64
    fusion_out:   int   = 32
    head_dropout: float = 0.30
    n_classes:    int   = 2


@dataclass
class TrainConfig:
    epochs:       int   = 100
    batch_size:   int   = 32
    lr:           float = 8e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    # Cosine annealing with warm restarts
    T_0:          int   = 20
    T_mult:       int   = 2
    seed:         int   = 42


@dataclass
class PathConfig:
    results_dir:  Path = ROOT / "results"
    checkpoints:  Path = ROOT / "results" / "checkpoints"
    figures:      Path = ROOT / "results" / "figures"

    def __post_init__(self):
        for p in [self.results_dir, self.checkpoints, self.figures]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig  = field(default_factory=PathConfig)


CFG = Config()
