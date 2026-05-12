"""
Graph Neural Network module for PsychoGraph-Net.

Implements a 2-layer Graph Convolutional Network (GCN) with:
    - Symmetric normalised message passing (D^{-1/2} A D^{-1/2})
    - GELU activation + LayerNorm after each layer
    - Attention-based node pooling to produce a fixed-size graph embedding

Input
-----
x   : FloatTensor [B, N_nodes, in_dim]  — per-node feature vectors
adj : FloatTensor [B, N_nodes, N_nodes] — normalised adjacency matrix (batched)

Output
------
FloatTensor [B, out_dim] — graph-level embedding via attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Single GCN message-passing layer.

    h^{(l+1)} = GELU( LayerNorm( Â · h^{(l)} · W ) )
    where Â is the pre-normalised adjacency matrix (passed in at forward time).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : [B, N, in_dim]
        adj : [B, N, N]

        Returns
        -------
        [B, N, out_dim]
        """
        agg = torch.bmm(adj, x)               # neighbourhood aggregation
        return self.drop(F.gelu(self.norm(self.W(agg))))


class GraphEncoder(nn.Module):
    """
    Two-layer GCN followed by attention pooling over nodes.

    Architecture
    ------------
    GCNLayer(in_dim → hidden) → GCNLayer(hidden → out_dim) → Attention Pool

    Parameters
    ----------
    in_dim  : Dimension of input node features.
    hidden  : Hidden dimension after first GCN layer.
    out_dim : Output graph embedding dimension.
    dropout : Dropout probability applied inside each GCN layer.
    """

    def __init__(
        self,
        in_dim:  int = 3,
        hidden:  int = 48,
        out_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden, dropout)
        self.gcn2 = GCNLayer(hidden, out_dim, dropout)
        self.attn = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : [B, N_nodes, in_dim]
        adj : [B, N_nodes, N_nodes]

        Returns
        -------
        [B, out_dim]  — attention-pooled graph embedding
        """
        h = self.gcn1(x, adj)                 # [B, N, hidden]
        h = self.gcn2(h, adj)                 # [B, N, out_dim]
        # Soft-attention over nodes → graph-level representation
        weights = torch.softmax(self.attn(h), dim=1)   # [B, N, 1]
        return (weights * h).sum(dim=1)                # [B, out_dim]
