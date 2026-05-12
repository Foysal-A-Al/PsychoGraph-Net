"""
Temporal Transformer Encoder for PsychoGraph-Net.

Processes longitudinal symptom sequences [B, seq_len, N_SYMPTOMS] and
produces a fixed-size temporal embedding via mean pooling over time steps.

Architecture
------------
  Linear projection → Positional embedding → Pre-LN Transformer Encoder → Mean pool

Pre-norm (norm_first=True) is used for training stability with short sequences.
"""

import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder operating on a sequence of symptom assessments.

    Parameters
    ----------
    n_symptoms  : Number of symptom features per time step.
    d_model     : Internal model dimension.
    n_heads     : Number of self-attention heads.
    n_layers    : Number of stacked TransformerEncoderLayer blocks.
    dropout     : Dropout applied inside attention and feed-forward sub-layers.
    max_seq_len : Maximum sequence length supported by positional embeddings.
    """

    def __init__(
        self,
        n_symptoms:  int   = 15,
        d_model:     int   = 64,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        dropout:     float = 0.15,
        max_seq_len: int   = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_symptoms, d_model)
        self.pos_embed  = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, seq_len, n_symptoms]

        Returns
        -------
        FloatTensor [B, d_model]  — mean-pooled temporal embedding
        """
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)   # [1, T]
        h    = self.input_proj(x) + self.pos_embed(pos)         # [B, T, d_model]
        h    = self.encoder(h)                                   # [B, T, d_model]
        return self.norm(h.mean(dim=1))                          # [B, d_model]
