"""
Baseline models: LSTM and 1D-CNN.

Both baselines operate on temporal sequences ONLY — they do not receive
graph-structured input. This isolates the contribution of the GNN module
in PsychoGraph-Net's ablation analysis.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Bidirectional LSTM baseline.

    Architecture: 2-layer stacked LSTM → last hidden state → dropout → FC head.

    Parameters
    ----------
    n_symptoms : Number of symptom features (input size at each time step).
    hidden_dim : LSTM hidden state dimension.
    n_layers   : Number of stacked LSTM layers.
    dropout    : Dropout between LSTM layers and before FC head.
    n_classes  : Number of output classes.
    """

    def __init__(
        self,
        n_symptoms: int   = 15,
        hidden_dim: int   = 128,
        n_layers:   int   = 2,
        dropout:    float = 0.25,
        n_classes:  int   = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_symptoms,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        *_,   # absorbs node_feats, adj passed by unified training loop
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, seq_len, n_symptoms]

        Returns
        -------
        FloatTensor [B, n_classes]
        """
        _, (h, _) = self.lstm(x)       # h: [n_layers, B, hidden]
        return self.head(h[-1])        # last layer's final hidden state


class CNNBaseline(nn.Module):
    """
    1D-CNN baseline with three convolutional blocks and global average pooling.

    Architecture:
        Conv1d(n_symptoms → 32) → GELU → BN
        Conv1d(32 → 64)          → GELU → BN
        Conv1d(64 → 128)         → GELU → BN
        GlobalAvgPool → Dropout → FC head

    Parameters
    ----------
    n_symptoms : Number of input channels (symptom features).
    dropout    : Dropout before the fully-connected head.
    n_classes  : Number of output classes.
    """

    def __init__(
        self,
        n_symptoms: int   = 15,
        dropout:    float = 0.30,
        n_classes:  int   = 2,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_symptoms, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        *_,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, seq_len, n_symptoms]

        Returns
        -------
        FloatTensor [B, n_classes]
        """
        # Conv1d expects [B, C, L] — transpose from [B, L, C]
        features = self.backbone(x.permute(0, 2, 1)).squeeze(-1)
        return self.head(features)
