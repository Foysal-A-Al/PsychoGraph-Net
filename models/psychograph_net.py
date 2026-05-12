"""
PsychoGraph-Net — Graph-Augmented Transformer for Psychiatric Prediction.

Architecture
------------

             Node features [B, N, 3]          Temporal sequence [B, T, N_SYM]
                    │                                      │
             ┌──────▼──────┐                     ┌────────▼────────┐
             │  GraphEncoder│                     │TemporalTransformer│
             │ (2-layer GCN │                     │ (Pre-LN, 2L, 4H) │
             │  + attn pool)│                     └────────┬────────┘
             └──────┬──────┘                              │
                    │ [B, 64]                              │ [B, 64]
                    └────────────────┬────────────────────┘
                                     │ concat [B, 128]
                              ┌──────▼──────┐
                              │  Fusion MLP  │
                              │ 128→64→32→2  │
                              └──────┬──────┘
                                     │
                              Logits [B, 2]

Graph stream captures relational co-occurrence structure across symptoms;
temporal stream captures sequential dynamics. Fusion allows complementary
signal integration — the key advantage over LSTM/CNN baselines that
access temporal sequences only.
"""

import torch
import torch.nn as nn

from .gnn import GraphEncoder
from .transformer import TemporalTransformerEncoder


class PsychoGraphNet(nn.Module):
    """
    Graph-Augmented Transformer for rapid-cycling bipolar disorder prediction.

    Parameters
    ----------
    n_symptoms   : Number of symptom features (nodes in the graph).
    gnn_in_dim   : Input dimension of node features.
    gnn_hidden   : GNN intermediate hidden dimension.
    gnn_out_dim  : GNN output dimension (graph embedding size).
    d_model      : Transformer internal dimension.
    n_heads      : Transformer attention heads.
    n_layers     : Transformer encoder layers.
    trm_dropout  : Dropout in Transformer sub-layers.
    fusion_hidden: First hidden layer size in fusion MLP.
    fusion_out   : Second hidden layer size in fusion MLP.
    head_dropout : Dropout in fusion MLP.
    n_classes    : Output classes (2 for binary prediction).
    """

    def __init__(
        self,
        n_symptoms:    int   = 15,
        gnn_in_dim:    int   = 3,
        gnn_hidden:    int   = 48,
        gnn_out_dim:   int   = 64,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        trm_dropout:   float = 0.15,
        fusion_hidden: int   = 64,
        fusion_out:    int   = 32,
        head_dropout:  float = 0.30,
        n_classes:     int   = 2,
    ):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            in_dim=gnn_in_dim,
            hidden=gnn_hidden,
            out_dim=gnn_out_dim,
        )
        self.temporal_encoder = TemporalTransformerEncoder(
            n_symptoms=n_symptoms,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=trm_dropout,
        )
        fusion_in = gnn_out_dim + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_hidden, fusion_out),
            nn.GELU(),
            nn.Linear(fusion_out, n_classes),
        )

    def forward(
        self,
        temporal:    torch.Tensor,
        node_feats:  torch.Tensor,
        adj:         torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal   : FloatTensor [B, seq_len, n_symptoms]
        node_feats : FloatTensor [B, n_symptoms, gnn_in_dim]
        adj        : FloatTensor [B, n_symptoms, n_symptoms]

        Returns
        -------
        FloatTensor [B, n_classes] — raw logits
        """
        g = self.graph_encoder(node_feats, adj)    # [B, gnn_out_dim]
        t = self.temporal_encoder(temporal)        # [B, d_model]
        return self.fusion_head(torch.cat([g, t], dim=-1))

    def get_joint_embedding(
        self,
        temporal:   torch.Tensor,
        node_feats: torch.Tensor,
        adj:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the fused embedding before the classification head.
        Useful for downstream probing, visualisation, or transfer learning.

        Returns
        -------
        FloatTensor [B, gnn_out_dim + d_model]
        """
        g = self.graph_encoder(node_feats, adj)
        t = self.temporal_encoder(temporal)
        return torch.cat([g, t], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
