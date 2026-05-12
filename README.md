# PsychoGraph-Net

**Graph-Augmented Transformer for Psychiatric Prediction**

A novel deep learning architecture that integrates patient symptom co-occurrence graphs (Neo4j) with a temporal attention encoder for bipolar disorder rapid-cycling prediction, with LIME-based clinical decision support.

> **Published:** Al Foysal, A., de Filippis, R. (2024). *Italian Journal of Psychiatry*, 10(1).

---

## Motivation

Existing sequential models (LSTM, 1D-CNN) treat psychiatric assessments as independent time series, discarding the **relational structure** between co-occurring symptoms. Clinical evidence shows that rapid-cycling bipolar disorder is characterised by tightly coupled mood–energy–impulsivity clusters that manifest as distinctive patterns in the symptom co-occurrence graph — information that is lost when symptoms are processed channel-by-channel.

PsychoGraph-Net addresses this by jointly encoding:
- **Graph stream** — symptom co-occurrence topology extracted from patient EHR data via Neo4j
- **Temporal stream** — longitudinal symptom dynamics via a pre-norm Transformer encoder

---

## Architecture

```
  Node features [B, N, 3]          Temporal sequence [B, T, N_SYM]
         │                                     │
  ┌──────▼──────┐                    ┌─────────▼─────────┐
  │  GraphEncoder│                   │TemporalTransformer │
  │ 2-layer GCN  │                   │ Pre-LN · 2L · 4H   │
  │ + attn pool  │                   └─────────┬──────────┘
  └──────┬──────┘                              │
         │ [B, 64]                             │ [B, 64]
         └──────────────────┬─────────────────┘
                            │ concat [B, 128]
                     ┌──────▼──────┐
                     │  Fusion MLP  │
                     │ 128→64→32→2  │
                     └─────────────┘
```

**GraphEncoder** — Symmetric normalised GCN (`D⁻¹/² A D⁻¹/²`) with GELU + LayerNorm and soft attention pooling over nodes.

**TemporalTransformer** — Linear projection + learnable positional embedding → pre-norm Transformer encoder → mean pooling.

**Fusion MLP** — Concatenated joint embedding → two-layer classifier with GELU and dropout regularisation.

---

## Results

Evaluated on N=600 synthetic patients (15 symptoms, seq_len=12) with graph-topology-embedded discriminative signal — reflecting the clinical reality that relational structure is necessary to separate rapid cyclers from non-cyclers.

| Model | Accuracy | Precision | Recall | F1 (macro) | AUC-ROC |
|---|---|---|---|---|---|
| LSTM | 0.9556 | 0.9554 | 0.9554 | 0.9554 | 0.9876 |
| CNN | 0.9667 | 0.9674 | 0.9658 | 0.9665 | 0.9886 |
| **PsychoGraph-Net** | **0.9778** | **0.9800** | **0.9762** | **0.9776** | **0.9903** |

**ΔF1 vs LSTM: +2.22 pp · ΔF1 vs CNN: +1.11 pp**

> The published paper reports +11% F1 over LSTM/CNN baselines on the real clinical dataset.  
> Synthetic results above are for reproducibility demonstration only.

---

## LIME Explainability

Feature attribution via perturbation-based LIME over the temporal symptom dimension. For each symptom, the time series is replaced with Gaussian noise and the resulting change in P(rapid cycling) is measured.

Top attributions on a representative rapid-cycling patient:

| Rank | Symptom | LIME Score |
|---|---|---|
| 1 | Mood Elevation | 1.000 |
| 2 | Impulsivity | 0.067 |
| 3–15 | Other symptoms | ≈0.000 |

Consistent with clinical DSM-5 rapid-cycling criteria — mood elevation (hypomanic/manic episodes) is the primary diagnostic signal.

---

## Repository Structure

```
PsychoGraph-Net/
├── config.py                     # All hyperparameters and paths
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation + LIME entry point
├── requirements.txt
│
├── data/
│   ├── synthetic_generator.py    # Synthetic EHR dataset with graph signal
│   └── neo4j_loader.py           # Production Neo4j integration
│
├── models/
│   ├── psychograph_net.py        # Proposed model (main)
│   ├── gnn.py                    # GCNLayer + GraphEncoder
│   ├── transformer.py            # TemporalTransformerEncoder
│   └── baselines.py              # LSTMBaseline, CNNBaseline
│
├── explainability/
│   └── lime_explainer.py         # LIME attribution (model-agnostic)
│
├── results/
│   ├── checkpoints/              # Saved model weights (.pt)
│   └── figures/                  # Generated plots
│
└── notebooks/
    └── demo.ipynb                # End-to-end walkthrough
```

---

## Installation

```bash
git clone https://github.com/Abdullah-Al-Foysal/PsychoGraph-Net.git
cd PsychoGraph-Net
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required. CUDA is optional.

---

## Usage

### Train all models

```bash
python train.py
```

```bash
python train.py --epochs 150 --lr 5e-4 --model pgn
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 8e-4 | Learning rate |
| `--model` | all | `pgn`, `lstm`, `cnn`, or `all` |

### Evaluate with LIME

```bash
python evaluate.py --model pgn --ckpt results/checkpoints/pgn_best.pt
```

### Use the model programmatically

```python
import torch
from models import PsychoGraphNet

model = PsychoGraphNet()
print(f"Parameters: {model.count_parameters():,}")

# Dummy batch
B, T, N = 8, 12, 15
temporal   = torch.randn(B, T, N)
node_feats = torch.randn(B, N, 3)
adj        = torch.eye(N).unsqueeze(0).expand(B, -1, -1)

logits = model(temporal, node_feats, adj)   # [B, 2]
```

### Use the LIME explainer

```python
from explainability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model, use_graph=True, n_perturbations=200)
result = explainer.explain(temporal[0], node_feats[0], adj[0])
explainer.print_report(result)
```

### Connect to Neo4j (production)

```python
from data.neo4j_loader import Neo4jSymptomGraphLoader

loader = Neo4jSymptomGraphLoader(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password",
)
temporal, node_feats, adj, labels = loader.load_all()
loader.close()
```

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
from config import CFG

CFG.train.epochs     = 150
CFG.train.lr         = 5e-4
CFG.data.n_patients  = 1000
```

---

## Hyperparameters

| Component | Parameter | Default |
|---|---|---|
| GNN | Hidden dim | 48 |
| GNN | Output dim | 64 |
| GNN | Layers | 2 |
| Transformer | d_model | 64 |
| Transformer | Heads | 4 |
| Transformer | Layers | 2 |
| Transformer | Dropout | 0.15 |
| Fusion MLP | Hidden | 64 → 32 |
| Fusion MLP | Dropout | 0.30 |
| Optimizer | AdamW, lr=8e-4, wd=1e-4 | |
| Scheduler | CosineAnnealingWarmRestarts T₀=20 | |

---

## Clinical Context

Bipolar disorder rapid cycling is defined by ≥4 mood episodes per year and is associated with poor pharmacological response and elevated suicide risk. Early, accurate identification enables clinicians to adjust treatment protocols proactively.

This work was developed in collaboration with clinical researchers at the Istituto di Psicopatologia, Rome, and is part of a broader programme on trustworthy AI for psychiatric decision support.

---

## Citation

```bibtex
@article{foysal2024psychographnet,
  title   = {PsychoGraph-Net: Graph-Augmented Transformer for Rapid-Cycling 
             Bipolar Disorder Prediction with LIME Explainability},
  author  = {Al Foysal, Abdullah and de Filippis, Renato},
  journal = {Italian Journal of Psychiatry},
  volume  = {10},
  number  = {1},
  year    = {2024}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

## Author

**Abdullah Al Foysal**  
MSc Computer Engineering (AI) · University of Genoa  
Research Assistant · InfoMus Lab, Casa Paganini (DIBRIS)  

[Google Scholar](https://scholar.google.com/citations?user=cQ_zolQAAAAJ) · [LinkedIn](https://linkedin.com/in/abdullah-al-foysal1)
