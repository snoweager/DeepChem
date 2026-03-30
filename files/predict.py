"""
predict.py — Inference Pipeline
=================================
Aurigene Pharmaceutical Services | AIDD Group

Provides:
  - ADMETPredictor class for clean inference on new SMILES
  - MC Dropout uncertainty estimation (20 stochastic forward passes)
  - Applicability domain flagging (Tanimoto-based)
  - Structured output: DataFrame with predictions, confidence, AD flag

Usage:
    from src.predict import ADMETPredictor

    predictor = ADMETPredictor(checkpoint="models/admet_net_best.pt")
    predictor.load_train_smiles("data/processed/train_smiles.txt")  # for AD

    results = predictor.predict([
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        "O=C(O)c1ccccc1O",     # Salicylic acid
    ])
    print(results)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ADMETNet, build_model, TASK_NAMES, TASK_CONFIG, REGRESSION_TASKS, CLASSIF_TASKS
from featurizer import MolecularFeaturizer, ApplicabilityDomain


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability labels for each task (for report-style output)
# ─────────────────────────────────────────────────────────────────────────────

TASK_LABELS = {
    "caco2":           ("Caco-2 Permeability (cm/s)",    "regression"),
    "bioavailability": ("Oral Bioavailability (%F)",      "classification"),
    "logP":            ("LogP (lipophilicity)",           "regression"),
    "bbb":             ("BBB Penetration",                "classification"),
    "cyp3a4":          ("CYP3A4 Inhibition",              "classification"),
    "cyp2c9":          ("CYP2C9 Inhibition",              "classification"),
    "cyp2d6":          ("CYP2D6 Inhibition",              "classification"),
    "half_life":       ("Half-life (h)",                  "regression"),
    "clearance":       ("Clearance (mL/min/kg)",          "regression"),
    "herg":            ("hERG Cardiotoxicity Risk",       "classification"),
    "ames":            ("AMES Mutagenicity",              "classification"),
    "dili":            ("DILI Hepatotoxicity Risk",       "classification"),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model:   ADMETNet,
    batch:   Batch,
    n_passes: int = 20,
    device:  torch.device = torch.device("cpu"),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Run N stochastic forward passes with dropout enabled to estimate uncertainty.

    Returns:
        dict mapping task → (mean_pred, std_pred) as numpy arrays (B,)
    """
    # Enable dropout during inference
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    all_preds = {task: [] for task in TASK_NAMES}

    with torch.no_grad():
        for _ in range(n_passes):
            preds = model(batch)
            for task in TASK_NAMES:
                all_preds[task].append(preds[task].cpu().numpy())

    results = {}
    for task in TASK_NAMES:
        stacked = np.stack(all_preds[task], axis=0)  # (n_passes, B, 1)
        stacked = stacked.squeeze(-1)                 # (n_passes, B)
        mean = stacked.mean(axis=0)  # (B,)
        std  = stacked.std(axis=0)   # (B,)
        results[task] = (mean, std)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ─────────────────────────────────────────────────────────────────────────────

class ADMETPredictor:
    """
    Clean inference interface for ADMET-Net.

    Args:
        checkpoint:   Path to .pt checkpoint file
        device:       'cuda', 'cpu', or 'auto'
        mc_passes:    Number of MC Dropout passes for uncertainty
        ad_threshold: Tanimoto threshold for applicability domain
    """

    def __init__(
        self,
        checkpoint:   str,
        device:       str  = "auto",
        mc_passes:    int  = 20,
        ad_threshold: float = 0.4,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.mc_passes    = mc_passes
        self.ad_threshold = ad_threshold
        self.featurizer   = MolecularFeaturizer()
        self.ad           = ApplicabilityDomain(threshold=ad_threshold)
        self._ad_fitted   = False

        self.model, _ = self._load_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(
        self, checkpoint: str
    ) -> Tuple[ADMETNet, dict]:
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg  = ckpt.get("cfg", {})

        # Rebuild model from config
        model, _ = build_model(cfg)
        model.load_state_dict(ckpt["model_state"])

        print(f"✅ Loaded checkpoint: {checkpoint}")
        print(f"   Trained for {ckpt.get('epoch', '?')} epochs  "
              f"|  val_loss = {ckpt.get('val_loss', '?'):.4f}")
        return model, cfg

    def load_train_smiles(self, path: str):
        """
        Load training set SMILES for applicability domain.
        File should have one SMILES per line.
        """
        with open(path) as f:
            smiles = [line.strip() for line in f if line.strip()]
        self.ad.fit(smiles)
        self._ad_fitted = True
        print(f"  AD fitted on {len(smiles):,} training molecules")

    def predict(
        self,
        smiles_list: List[str],
        verbose:     bool = True,
    ) -> pd.DataFrame:
        """
        Predict all 12 ADMET endpoints for a list of SMILES.

        Returns:
            pd.DataFrame with columns:
              - smiles
              - {task}_pred    (raw value or probability)
              - {task}_std     (MC Dropout uncertainty)
              - {task}_label   (human-readable class for classification)
              - in_domain      (bool)
              - max_tanimoto   (float)
        """
        # Featurize
        graphs, failed = self.featurizer.batch_featurize(smiles_list, verbose=verbose)
        valid_smiles   = [smi for smi in smiles_list if smi not in set(failed)]

        if not graphs:
            raise ValueError("No valid SMILES could be featurized.")

        # Applicability domain
        ad_results = {}
        if self._ad_fitted:
            for r in self.ad.check(valid_smiles):
                ad_results[r["smiles"]] = r
        else:
            for smi in valid_smiles:
                ad_results[smi] = {"in_domain": None, "max_tanimoto": None}

        # Build batch
        # Fix: fp needs shape (n_atoms_i, fp_dim) — actually fp is per-molecule
        # We stored fp as (1, FP_DIM) per graph. Batch handles this.
        batch = Batch.from_data_list(graphs).to(self.device)

        # MC Dropout inference
        mc_results = mc_dropout_predict(self.model, batch, n_passes=self.mc_passes)

        # Build output DataFrame
        rows = []
        for i, smi in enumerate(valid_smiles):
            row = {"smiles": smi}

            for task in TASK_NAMES:
                mean_raw, std_raw = mc_results[task][i], mc_results[task + "_std"][i] \
                    if f"{task}_std" in mc_results else (mc_results[task][0][i], mc_results[task][1][i])
                # Unpack properly
                mean_raw = mc_results[task][0][i]
                std_raw  = mc_results[task][1][i]

                task_type = TASK_CONFIG[task][0]

                if task_type == "classification":
                    prob = float(_sigmoid(mean_raw))
                    std  = float(_sigmoid(mean_raw + std_raw) - _sigmoid(mean_raw - std_raw)) / 2
                    row[f"{task}_prob"]  = round(prob, 4)
                    row[f"{task}_std"]   = round(std, 4)
                    row[f"{task}_label"] = "Positive" if prob >= 0.5 else "Negative"
                else:
                    row[f"{task}_pred"] = round(float(mean_raw), 4)
                    row[f"{task}_std"]  = round(float(std_raw), 4)

            # Applicability domain
            ad_info = ad_results.get(smi, {})
            row["in_domain"]    = ad_info.get("in_domain")
            row["max_tanimoto"] = ad_info.get("max_tanimoto")

            rows.append(row)

        df = pd.DataFrame(rows)

        # Add failed molecules
        if failed:
            fail_df = pd.DataFrame({"smiles": failed, "in_domain": False})
            df = pd.concat([df, fail_df], ignore_index=True)

        return df

    def predict_single(self, smiles: str) -> Dict:
        """Convenience wrapper for a single SMILES. Returns dict."""
        df = self.predict([smiles], verbose=False)
        if df.empty:
            return {"smiles": smiles, "error": "Invalid SMILES"}
        return df.iloc[0].to_dict()

    def print_report(self, smiles: str):
        """Pretty-print a single molecule's ADMET profile."""
        result = self.predict_single(smiles)
        if "error" in result:
            print(f"❌ {result['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  ADMET Profile | Aurigene AIDD")
        print(f"  SMILES: {smiles[:55]}")
        if result.get("in_domain") is not None:
            dom = "✅ In Domain" if result["in_domain"] else "⚠️  Out of Domain"
            print(f"  AD Status: {dom}  (Tanimoto = {result['max_tanimoto']:.3f})")
        print(f"{'='*60}")

        categories = {
            "Absorption":    ["caco2", "bioavailability"],
            "Distribution":  ["logP", "bbb"],
            "Metabolism":    ["cyp3a4", "cyp2c9", "cyp2d6"],
            "Excretion":     ["half_life", "clearance"],
            "Toxicity":      ["herg", "ames", "dili"],
        }

        for cat, tasks in categories.items():
            print(f"\n  [{cat}]")
            for task in tasks:
                task_type = TASK_CONFIG[task][0]
                label, _ = TASK_LABELS[task]
                if task_type == "classification":
                    prob  = result.get(f"{task}_prob", "N/A")
                    cls   = result.get(f"{task}_label", "")
                    std   = result.get(f"{task}_std", 0)
                    flag  = "🔴" if cls == "Positive" else "🟢"
                    print(f"    {flag} {label:35s}: {cls} (p={prob:.3f} ± {std:.3f})")
                else:
                    pred = result.get(f"{task}_pred", "N/A")
                    std  = result.get(f"{task}_std", 0)
                    print(f"    📊 {label:35s}: {pred:.3f} ± {std:.3f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI for quick inference
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADMET-Net Inference")
    parser.add_argument("--checkpoint", type=str, default="models/admet_net_best.pt")
    parser.add_argument("--smiles",     type=str, nargs="+",
                        default=["CC(=O)Nc1ccc(O)cc1"])
    parser.add_argument("--output",     type=str, default=None,
                        help="CSV output path (optional)")
    parser.add_argument("--train_smiles", type=str, default=None,
                        help="Path to training SMILES file for AD check")
    args = parser.parse_args()

    predictor = ADMETPredictor(checkpoint=args.checkpoint)

    if args.train_smiles:
        predictor.load_train_smiles(args.train_smiles)

    # Print report for each molecule
    for smi in args.smiles:
        predictor.print_report(smi)

    # Optionally save DataFrame
    if args.output:
        df = predictor.predict(args.smiles)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
