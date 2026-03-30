"""
dataset.py — ADMET Dataset & DataLoader Construction
======================================================
Aurigene Pharmaceutical Services | AIDD Group

Handles:
  - Loading multiple public datasets (Tox21, ESOL, hERG, BBB, CYP450, DILI)
  - Merging on SMILES key with proper NaN handling for missing labels
  - Scaffold-based train/val/test splitting (prevents data leakage)
  - PyG DataLoader construction

Scaffold split is critical in pharma ML:
  Random splits overestimate generalization. Scaffold splits ensure
  molecules with similar core structures don't appear in both
  train and test sets — mimicking real prospective screening scenarios.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurizer import MolecularFeaturizer
from model import TASK_NAMES, TASK_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Dataset sources configuration
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (filename, smiles_col, {task_name: label_col})
DATASET_SOURCES = {
    "tox21": {
        "file":     "data/raw/tox21.csv",
        "smiles":   "smiles",
        "tasks":    {"ames": "NR-AR"},  # mapped to our task name
    },
    "esol": {
        "file":     "data/raw/esol.csv",
        "smiles":   "smiles",
        "tasks":    {"logP": "measured log solubility in mols per litre"},
    },
    "herg": {
        "file":     "data/raw/herg_central.csv",
        "smiles":   "SMILES",
        "tasks":    {"herg": "hERG_label"},
    },
    "bbb": {
        "file":     "data/raw/bbb_martins.csv",
        "smiles":   "Drug",
        "tasks":    {"bbb": "Y"},
    },
    "cyp450": {
        "file":     "data/raw/cyp_p450_2d6_inhibition.csv",
        "smiles":   "Drug",
        "tasks":    {
            "cyp3a4": "CYP3A4",
            "cyp2c9": "CYP2C9",
            "cyp2d6": "CYP2D6",
        },
    },
    "caco2": {
        "file":     "data/raw/caco2_wang.csv",
        "smiles":   "Drug",
        "tasks":    {"caco2": "Y"},
    },
    "dili": {
        "file":     "data/raw/dili.csv",
        "smiles":   "SMILES",
        "tasks":    {"dili": "Label"},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + merging
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_datasets(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads all dataset CSVs, standardizes SMILES, and merges into a single
    DataFrame with one row per unique SMILES and one column per task.
    Missing values are NaN (handled by masks during training).
    """
    all_dfs = []

    for source, cfg in DATASET_SOURCES.items():
        filepath = cfg["file"]
        if not os.path.exists(filepath):
            print(f"  ⚠️  Skipping {source}: file not found at {filepath}")
            continue

        df = pd.read_csv(filepath)
        smi_col = cfg["smiles"]

        # Standardize SMILES via RDKit
        df["smiles_std"] = df[smi_col].apply(_canonicalize_smiles)
        df = df.dropna(subset=["smiles_std"])

        # Rename task columns
        task_cols = {"smiles_std": "smiles"}
        for task_name, orig_col in cfg["tasks"].items():
            if orig_col in df.columns:
                task_cols[orig_col] = task_name

        df = df[list(task_cols.keys())].rename(columns=task_cols)
        df = df.groupby("smiles").first().reset_index()
        all_dfs.append(df)
        print(f"  Loaded {source:10s}: {len(df):6,} molecules  "
              f"tasks={list(cfg['tasks'].keys())}")

    if not all_dfs:
        raise FileNotFoundError(
            "No dataset files found. Download datasets with:\n"
            "  python data/download_datasets.py\n"
            "or place CSV files in data/raw/"
        )

    # Merge on SMILES (outer join preserves all molecules)
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on="smiles", how="outer")

    # Ensure all task columns exist (NaN if missing)
    for task in TASK_NAMES:
        if task not in merged.columns:
            merged[task] = np.nan

    print(f"\n  Total unique SMILES: {len(merged):,}")
    print(f"  Label density per task:")
    for task in TASK_NAMES:
        n_labeled = merged[task].notna().sum()
        print(f"    {task:20s}: {n_labeled:6,} / {len(merged):,} ({100*n_labeled/len(merged):.1f}%)")

    return merged[["smiles"] + TASK_NAMES]


def _canonicalize_smiles(smi: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold split
# ─────────────────────────────────────────────────────────────────────────────

def scaffold_split(
    smiles_list: List[str],
    val_frac:    float = 0.1,
    test_frac:   float = 0.1,
    seed:        int   = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Butina scaffold-based split. Groups molecules by Murcko scaffold,
    assigns scaffold groups to train/val/test ensuring no scaffold leakage.

    Returns:
        train_idx, val_idx, test_idx (lists of integer indices)
    """
    scaffolds: Dict[str, List[int]] = defaultdict(list)

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = "invalid"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except Exception:
                scaffold = smi  # fallback to SMILES itself

        scaffolds[scaffold].append(idx)

    # Sort scaffold groups by size (largest first → go to train)
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    n_total   = len(smiles_list)
    n_val     = int(n_total * val_frac)
    n_test    = int(n_total * test_frac)

    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_groups)

    train_idx, val_idx, test_idx = [], [], []

    for group in scaffold_groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    print(f"\n  Scaffold split: train={len(train_idx):,}  "
          f"val={len(val_idx):,}  test={len(test_idx):,}")
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ADMETDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-featurized molecules with multi-task labels.

    Args:
        graphs:  List of PyG Data objects (from MolecularFeaturizer)
        labels:  Dict mapping task_name → list of float labels (NaN = missing)
    """

    def __init__(
        self,
        graphs: List[Data],
        labels: Dict[str, List[float]],
    ):
        assert all(len(v) == len(graphs) for v in labels.values()), \
            "graphs and labels must have same length"
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        g = self.graphs[idx]
        # Attach task labels as graph attributes
        for task in TASK_NAMES:
            val = self.labels[task][idx]
            setattr(g, task, torch.tensor([val], dtype=torch.float))
        return g


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Full pipeline: load CSVs → featurize → scaffold split → DataLoaders.

    Expected cfg keys (under cfg['data'] and cfg['training']):
        data.data_dir:    Path to raw CSVs
        data.val_frac:    Validation fraction
        data.test_frac:   Test fraction
        data.seed:        Random seed
        training.batch_size: Batch size
        training.num_workers: DataLoader workers
    """
    dc = cfg.get("data", {})
    tc = cfg.get("training", {})

    # 1. Load + merge
    merged_df = load_and_merge_datasets(dc.get("data_dir", "data/raw"))

    # 2. Featurize
    print("\n  Featurizing molecules...")
    featurizer = MolecularFeaturizer(
        fp_radius=dc.get("fp_radius", 2),
        fp_bits=dc.get("fp_bits", 2048),
    )
    smiles_list = merged_df["smiles"].tolist()
    graphs, failed_smiles = featurizer.batch_featurize(smiles_list, verbose=True)

    # Track which indices are valid
    valid_mask = [smi not in set(failed_smiles) for smi in smiles_list]
    valid_df   = merged_df[valid_mask].reset_index(drop=True)

    # 3. Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        valid_df["smiles"].tolist(),
        val_frac=dc.get("val_frac", 0.1),
        test_frac=dc.get("test_frac", 0.1),
        seed=dc.get("seed", 42),
    )

    # 4. Build label dicts
    label_dict = {
        task: valid_df[task].tolist()
        for task in TASK_NAMES
    }

    def _subset(idxs: List[int]) -> ADMETDataset:
        sub_graphs = [graphs[i] for i in idxs]
        sub_labels = {t: [label_dict[t][i] for i in idxs] for t in TASK_NAMES}
        return ADMETDataset(sub_graphs, sub_labels)

    train_ds = _subset(train_idx)
    val_ds   = _subset(val_idx)
    test_ds  = _subset(test_idx)

    batch_size   = tc.get("batch_size", 32)
    num_workers  = tc.get("num_workers", 4)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Download helper script reference
# ─────────────────────────────────────────────────────────────────────────────
# All public datasets are freely available via DeepChem's MoleculeNet:
#
#   import deepchem as dc
#   tasks, datasets, transformers = dc.molnet.load_tox21()
#   tasks, datasets, transformers = dc.molnet.load_herg_central()
#   tasks, datasets, transformers = dc.molnet.load_bbbp()
#   tasks, datasets, transformers = dc.molnet.load_cyp_p450_2d6_inhibition()
#   tasks, datasets, transformers = dc.molnet.load_caco2_wang()
#   tasks, datasets, transformers = dc.molnet.load_thermosol()  # for logP proxy
#
# Save each to data/raw/<name>.csv
# ─────────────────────────────────────────────────────────────────────────────
