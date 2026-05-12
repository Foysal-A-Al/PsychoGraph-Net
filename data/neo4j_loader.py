"""
Neo4j symptom graph loader — production integration.

Replaces synthetic_generator.py when a real Neo4j instance is available.
Extracts patient symptom graphs via Cypher queries and returns the same
tensor format expected by PsychDataset and the model.

Requirements
------------
    pip install neo4j torch numpy

Schema assumed in Neo4j
-----------------------
    (:Patient {id, label, seq_data_json})
        -[:HAS_SYMPTOM {weight, timestamp}]->
    (:Symptom {name, category})
        -[:CO_OCCURS_WITH {pearson_r, frequency}]->
    (:Symptom)

Usage
-----
    from data.neo4j_loader import Neo4jSymptomGraphLoader

    loader = Neo4jSymptomGraphLoader(uri="bolt://localhost:7687",
                                     user="neo4j", password="password")
    temporal, node_feats, adj, labels = loader.load_all()
    loader.close()
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

SYMPTOM_ORDER = [
    "Mood Elevation", "Mood Depression", "Sleep Duration", "Energy Level",
    "Psychomotor Activity", "Irritability", "Racing Thoughts", "Distractibility",
    "Grandiosity", "Impulsivity", "Anhedonia", "Appetite Change",
    "Concentration", "Social Withdrawal", "Suicidal Ideation",
]
N_SYMPTOMS = len(SYMPTOM_ORDER)


class Neo4jSymptomGraphLoader:
    """
    Loads patient symptom graphs from a Neo4j instance.

    Parameters
    ----------
    uri      : Neo4j bolt URI, e.g. "bolt://localhost:7687"
    user     : Neo4j username
    password : Neo4j password
    seq_len  : Number of longitudinal assessment time points to include
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        seq_len: int = 12,
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "neo4j-driver is required: pip install neo4j"
            ) from exc

        self.driver  = GraphDatabase.driver(uri, auth=(user, password))
        self.seq_len = seq_len
        logger.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    def load_all(self) -> tuple:
        """
        Execute Cypher queries and return tensors in the same format as
        generate_dataset() — drop-in replacement for synthetic data.

        Returns
        -------
        temporal    : FloatTensor [N, seq_len, N_SYMPTOMS]
        node_feats  : FloatTensor [N, N_SYMPTOMS, 3]
        adj_matrices: FloatTensor [N, N_SYMPTOMS, N_SYMPTOMS]
        labels      : LongTensor  [N]
        """
        with self.driver.session() as session:
            patients = session.run(
                "MATCH (p:Patient) RETURN p.id AS id, p.label AS label, "
                "p.seq_data_json AS seq_data ORDER BY p.id"
            ).data()

        n = len(patients)
        temporal_data = np.zeros((n, self.seq_len, N_SYMPTOMS), dtype=np.float32)
        node_features = np.zeros((n, N_SYMPTOMS, 3), dtype=np.float32)
        adj_matrices  = np.zeros((n, N_SYMPTOMS, N_SYMPTOMS), dtype=np.float32)
        labels        = np.zeros(n, dtype=np.int64)

        sym_idx = {name: i for i, name in enumerate(SYMPTOM_ORDER)}

        for i, p in enumerate(patients):
            seq = json.loads(p["seq_data"])          # [{symptom: value, ...}, ...]
            base = np.zeros((self.seq_len, N_SYMPTOMS), dtype=np.float32)
            for t, assessment in enumerate(seq[: self.seq_len]):
                for sym, val in assessment.items():
                    if sym in sym_idx:
                        base[t, sym_idx[sym]] = float(val)

            temporal_data[i] = base
            node_features[i] = np.stack(
                [base.mean(0), base.std(0) + 1e-6, base.max(0) - base.min(0)],
                axis=1,
            )

            # ── Per-patient co-occurrence graph ────────────────────
            corr = np.corrcoef(base.T)
            adj  = (np.abs(corr) > 0.22).astype(np.float32)
            np.fill_diagonal(adj, 1.0)
            deg      = adj.sum(1)
            d_inv_sq = np.where(deg > 0, deg ** -0.5, 0.0)
            D        = np.diag(d_inv_sq)
            adj_matrices[i] = D @ adj @ D

            labels[i] = int(p["label"])

        logger.info("Loaded %d patients from Neo4j", n)
        return (
            torch.FloatTensor(temporal_data),
            torch.FloatTensor(node_features),
            torch.FloatTensor(adj_matrices),
            torch.LongTensor(labels),
        )
