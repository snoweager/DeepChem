"""
ADMET-Net: Multi-Task Graph Neural Network for ADMET Property Prediction
=========================================================================
Aurigene Pharmaceutical Services | AIDD Group | Hyderabad

Architecture: AttentiveFP-inspired MPNN with task-specific heads and
learnable uncertainty weighting for multi-task loss balancing.

Reference:
  - Jiang et al. (2021) "Interactively Analyzing Graph Neural Networks
    for Drug Discovery" — AttentiveFP, JCIM
  - Kendall & Gal (2018) "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics" — NeurIPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, global_mean_pool, global_max_pool, BatchNorm
)
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions (used throughout training + evaluation)
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    # name: (type, loss_weight_init)  type: 'regression' | 'classification'
    "caco2":            ("regression",     1.0),
    "bioavailability":  ("classification", 1.0),
    "logP":             ("regression",     1.0),
    "bbb":              ("classification", 1.0),
    "cyp3a4":           ("classification", 1.0),
    "cyp2c9":           ("classification", 1.0),
    "cyp2d6":           ("classification", 1.0),
    "half_life":        ("regression",     1.0),
    "clearance":        ("regression",     1.0),
    "herg":             ("classification", 1.0),
    "ames":             ("classification", 1.0),
    "dili":             ("classification", 1.0),
}

TASK_NAMES      = list(TASK_CONFIG.keys())
REGRESSION_TASKS = [t for t, (kind, _) in TASK_CONFIG.items() if kind == "regression"]
CLASSIF_TASKS    = [t for t, (kind, _) in TASK_CONFIG.items() if kind == "classification"]


# ─────────────────────────────────────────────────────────────────────────────
# Atom & Bond feature dimensions (must match featurizer.py)
# ─────────────────────────────────────────────────────────────────────────────

ATOM_FEAT_DIM = 74   # See featurizer.py → atom_features()
BOND_FEAT_DIM = 12   # See featurizer.py → bond_features()
FP_DIM        = 2048 # Morgan fingerprint bits (r=2)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Configurable multi-layer perceptron with LayerNorm + Dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.2,
        activation: nn.Module = None,
    ):
        super().__init__()
        activation = activation or nn.GELU()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                activation,
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MPNNLayer(nn.Module):
    """
    Single message-passing layer using GATv2Conv (edge-feature aware).
    Includes a residual connection when input/output dims match.
    """

    def __init__(self, node_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert node_dim % heads == 0, "node_dim must be divisible by heads"
        self.conv = GATv2Conv(
            in_channels=node_dim,
            out_channels=node_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )
        self.norm = BatchNorm(node_dim)
        self.act  = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = self.act(out + x)  # residual
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────

class ADMETNet(nn.Module):
    """
    Multi-task ADMET prediction network.

    Pipeline:
        1. Linear projection: atom features → node_dim
        2. Linear projection: bond features → edge_dim
        3. N × MPNNLayer (GATv2Conv + residual + BatchNorm)
        4. Readout: mean_pool ⊕ max_pool → 2*node_dim
        5. Fingerprint MLP: FP_DIM → fp_embed_dim
        6. Fusion: concat(graph_embed, fp_embed) → shared_dim
        7. 12 task-specific MLP heads

    Uncertainty weighting:
        log_vars (learnable) per task → softmax-style balanced loss.
        Regression:     loss = (1/2σ²) * MSE + log σ
        Classification: loss = (1/2σ²) * BCE + log σ
    """

    def __init__(
        self,
        node_dim:     int   = 128,
        edge_dim:     int   = 64,
        n_layers:     int   = 4,
        fp_embed_dim: int   = 256,
        shared_dim:   int   = 512,
        head_hidden:  List[int] = [256, 128],
        dropout:      float = 0.2,
        gat_heads:    int   = 4,
    ):
        super().__init__()

        # ── Input projections ─────────────────────────────────────
        self.atom_proj = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.bond_proj = nn.Sequential(
            nn.Linear(BOND_FEAT_DIM, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
        )

        # ── Message passing ───────────────────────────────────────
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, heads=gat_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Fingerprint encoder ───────────────────────────────────
        self.fp_encoder = MLP(
            in_dim=FP_DIM,
            hidden_dims=[512],
            out_dim=fp_embed_dim,
            dropout=dropout,
        )

        # ── Fusion layer ──────────────────────────────────────────
        graph_out_dim = node_dim * 2          # mean + max pooling
        fusion_in_dim = graph_out_dim + fp_embed_dim
        self.fusion = MLP(
            in_dim=fusion_in_dim,
            hidden_dims=[shared_dim],
            out_dim=shared_dim,
            dropout=dropout,
        )

        # ── Task heads ────────────────────────────────────────────
        self.heads = nn.ModuleDict()
        for task in TASK_NAMES:
            task_type = TASK_CONFIG[task][0]
            out_dim   = 1  # single scalar for both regression and classification
            self.heads[task] = MLP(
                in_dim=shared_dim,
                hidden_dims=head_hidden,
                out_dim=out_dim,
                dropout=dropout,
            )

        # ── Learnable log-variance per task (uncertainty weighting) ─
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in TASK_NAMES
        })

    def encode(self, batch: Batch) -> torch.Tensor:
        """
        Returns the fused molecular embedding (batch_size, shared_dim).
        Useful for downstream tasks or UMAP visualization.
        """
        x         = self.atom_proj(batch.x)
        edge_attr = self.bond_proj(batch.edge_attr)

        for layer in self.mpnn_layers:
            x = layer(x, batch.edge_index, edge_attr)

        # Global readout
        mean_pool = global_mean_pool(x, batch.batch)
        max_pool  = global_max_pool(x, batch.batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*node_dim)

        # Fingerprint path
        fp_emb = self.fp_encoder(batch.fp)  # (B, fp_embed_dim)

        # Fuse
        fused = self.fusion(torch.cat([graph_emb, fp_emb], dim=-1))  # (B, shared_dim)
        return fused

    def forward(
        self, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: PyG Batch with attributes:
                - x:          (N_atoms_total, ATOM_FEAT_DIM)
                - edge_index: (2, N_bonds_total)
                - edge_attr:  (N_bonds_total, BOND_FEAT_DIM)
                - fp:         (B, FP_DIM)  Morgan fingerprint per molecule
                - batch:      (N_atoms_total,) molecule index

        Returns:
            dict mapping task_name → raw logit/scalar (B, 1)
        """
        fused = self.encode(batch)

        preds = {}
        for task in TASK_NAMES:
            preds[task] = self.heads[task](fused)  # (B, 1)
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task loss with uncertainty weighting
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Kendall & Gal (2018) homoscedastic uncertainty weighting.

    For regression:     L_i = (1/(2σ_i²)) * MSE_i + log σ_i
    For classification: L_i = (1/(2σ_i²)) * BCE_i + log σ_i

    log_vars is shared with ADMETNet to allow joint optimization.
    """

    def __init__(self, log_vars: nn.ParameterDict):
        super().__init__()
        self.log_vars = log_vars

    def forward(
        self,
        preds:  Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        masks:  Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            preds:  {task: (B,1) raw model output}
            labels: {task: (B,1) ground truth}
            masks:  {task: (B,1) bool — True where label is available}

        Returns:
            total_loss: scalar
            task_losses: {task: scalar} for logging
        """
        total_loss  = torch.tensor(0.0, device=next(iter(preds.values())).device)
        task_losses = {}

        for task in TASK_NAMES:
            pred  = preds[task]     # (B, 1)
            label = labels[task]    # (B, 1)
            mask  = masks[task]     # (B, 1) bool

            # Skip task if no labels available in this batch
            if mask.sum() == 0:
                task_losses[task] = torch.tensor(0.0)
                continue

            p = pred[mask]
            y = label[mask]

            task_type = TASK_CONFIG[task][0]
            log_var   = self.log_vars[task]
            precision = torch.exp(-log_var)   # 1/σ²

            if task_type == "regression":
                raw_loss = F.mse_loss(p, y)
                loss     = precision * raw_loss + log_var
            else:
                raw_loss = F.binary_cross_entropy_with_logits(p, y)
                loss     = precision * raw_loss + log_var

            total_loss += loss
            task_losses[task] = raw_loss.detach()

        return total_loss, task_losses


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: model factory from config dict
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> Tuple[ADMETNet, MultiTaskLoss]:
    """
    Build model and loss from a config dict (e.g., loaded from config.yaml).

    Expected keys under cfg['model']:
        node_dim, edge_dim, n_layers, fp_embed_dim,
        shared_dim, head_hidden, dropout, gat_heads
    """
    model = ADMETNet(**cfg["model"])
    loss_fn = MultiTaskLoss(model.log_vars)
    return model, loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    import random

    torch.manual_seed(42)

    def _fake_graph(n_atoms: int = 15, n_bonds: int = 20) -> Data:
        return Data(
            x          = torch.randn(n_atoms, ATOM_FEAT_DIM),
            edge_index = torch.randint(0, n_atoms, (2, n_bonds)),
            edge_attr  = torch.randn(n_bonds, BOND_FEAT_DIM),
            fp         = torch.randn(1, FP_DIM),
        )

    batch = Batch.from_data_list([_fake_graph() for _ in range(4)])

    model, loss_fn = build_model({
        "model": {
            "node_dim":     128,
            "edge_dim":     64,
            "n_layers":     4,
            "fp_embed_dim": 256,
            "shared_dim":   512,
            "head_hidden":  [256, 128],
            "dropout":      0.2,
            "gat_heads":    4,
        }
    })

    preds = model(batch)
    print("✅ Forward pass OK")
    for task, out in preds.items():
        print(f"  {task:20s}: shape {tuple(out.shape)}")

    # Fake labels + masks
    B = 4
    labels = {t: torch.randn(B, 1) if TASK_CONFIG[t][0]=="regression"
                 else torch.randint(0, 2, (B, 1)).float()
              for t in TASK_NAMES}
    masks  = {t: torch.ones(B, 1).bool() for t in TASK_NAMES}

    total, task_losses = loss_fn(preds, labels, masks)
    print(f"\n✅ Loss OK  |  total = {total.item():.4f}")
    print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
