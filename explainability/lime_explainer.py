"""
LIME explainability for PsychoGraph-Net.

Implements perturbation-based local feature attribution over the temporal
symptom dimension. For each symptom, the time series is replaced with
Gaussian noise and the resulting change in predicted class probability is
measured. Feature importance is the mean absolute probability shift,
normalised to [0, 1].

This is a model-agnostic method compatible with both PsychoGraph-Net
and the LSTM/CNN baselines, enabling consistent cross-model attribution.

Reference
---------
Ribeiro et al. (2016). "Why Should I Trust You?" Explaining the Predictions
of Any Classifier. KDD 2016. https://arxiv.org/abs/1602.04938
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic_generator import SYMPTOM_NAMES


class LIMEExplainer:
    """
    Perturbation-based LIME explainer for temporal psychiatric models.

    Parameters
    ----------
    model         : Trained PsychoGraphNet, LSTMBaseline, or CNNBaseline.
    use_graph     : If True, passes node_feats and adj to the model.
    n_perturbations: Number of random perturbations per symptom.
    target_class  : Class index to explain (1 = rapid cycling by convention).
    """

    def __init__(
        self,
        model,
        use_graph:         bool = True,
        n_perturbations:   int  = 200,
        target_class:      int  = 1,
    ):
        self.model           = model
        self.use_graph       = use_graph
        self.n_perturbations = n_perturbations
        self.target_class    = target_class
        self.model.eval()

    @torch.no_grad()
    def explain(
        self,
        temporal:   torch.Tensor,
        node_feats: torch.Tensor | None = None,
        adj:        torch.Tensor | None = None,
    ) -> dict:
        """
        Compute LIME attributions for a single patient sample.

        Parameters
        ----------
        temporal   : FloatTensor [seq_len, N_SYMPTOMS]
        node_feats : FloatTensor [N_SYMPTOMS, 3]  — required if use_graph=True
        adj        : FloatTensor [N_SYMPTOMS, N_SYMPTOMS] — required if use_graph=True

        Returns
        -------
        dict with keys:
            'importances'     : np.ndarray [N_SYMPTOMS], normalised [0, 1]
            'base_prob'       : float, baseline P(target_class) before perturbation
            'symptom_names'   : list[str]
            'ranked'          : list of (symptom_name, importance) sorted descending
        """
        n_sym = temporal.shape[1]
        base_prob = self._predict(temporal, node_feats, adj)
        importances = np.zeros(n_sym, dtype=np.float32)

        for s in range(n_sym):
            shifts = []
            sigma  = temporal[:, s].std().item() + 1e-6
            for _ in range(self.n_perturbations):
                perturbed = temporal.clone()
                perturbed[:, s] = torch.randn(temporal.shape[0]) * sigma
                p = self._predict(perturbed, node_feats, adj)
                shifts.append(abs(base_prob - p))
            importances[s] = float(np.mean(shifts))

        # Normalise to [0, 1]
        mn, mx = importances.min(), importances.max()
        importances = (importances - mn) / (mx - mn + 1e-9)

        ranked = sorted(
            zip(SYMPTOM_NAMES[:n_sym], importances.tolist()),
            key=lambda x: -x[1],
        )

        return {
            'importances':   importances,
            'base_prob':     base_prob,
            'symptom_names': SYMPTOM_NAMES[:n_sym],
            'ranked':        ranked,
        }

    def _predict(self, temporal, node_feats, adj) -> float:
        t   = temporal.unsqueeze(0)
        nf  = node_feats.unsqueeze(0) if node_feats is not None else None
        a   = adj.unsqueeze(0)        if adj        is not None else None

        if self.use_graph:
            logits = self.model(t, nf, a)
        else:
            logits = self.model(t)

        return F.softmax(logits, dim=-1)[0, self.target_class].item()

    def print_report(self, result: dict, top_k: int = 8):
        """Pretty-print attribution results to stdout."""
        print(f"\n  P(rapid cycling) = {result['base_prob']:.3f}")
        print(f"  {'Symptom':<28} {'Importance':>10}  Bar")
        print("  " + "─" * 55)
        for sym, score in result['ranked'][:top_k]:
            bar = "█" * int(score * 25)
            print(f"  {sym:<28} {score:>10.4f}  {bar}")
