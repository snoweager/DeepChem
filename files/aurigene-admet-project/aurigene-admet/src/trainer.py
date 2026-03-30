"""
trainer.py — Multi-Task Training Loop
=======================================
Aurigene Pharmaceutical Services | AIDD Group

Features:
  - Multi-task learning with uncertainty-weighted loss (Kendall & Gal, 2018)
  - Cosine annealing with warm restarts (SGDR)
  - Early stopping on validation loss
  - Per-task metric logging (AUC for classification, RMSE for regression)
  - Gradient clipping + weight decay
  - Checkpoint saving (best val loss)

Run: python src/trainer.py --config config.yaml
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, List, Optional

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ADMETNet, MultiTaskLoss, build_model, TASK_NAMES, TASK_CONFIG, REGRESSION_TASKS, CLASSIF_TASKS
from dataset import ADMETDataset, build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds:  Dict[str, List[float]],
    all_labels: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Compute per-task metrics:
      - Classification: ROC-AUC (skipped if only one class present)
      - Regression: RMSE
    """
    metrics = {}
    for task in TASK_NAMES:
        preds  = np.array(all_preds[task])
        labels = np.array(all_labels[task])

        if len(preds) == 0:
            metrics[task] = float("nan")
            continue

        task_type = TASK_CONFIG[task][0]
        if task_type == "classification":
            probs = 1 / (1 + np.exp(-preds))   # sigmoid
            try:
                metrics[task] = roc_auc_score(labels, probs)
            except ValueError:
                metrics[task] = float("nan")
        else:
            metrics[task] = float(np.sqrt(np.mean((preds - labels) ** 2)))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch: train or eval
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:    ADMETNet,
    loss_fn:  MultiTaskLoss,
    loader:   DataLoader,
    optimizer: Optional[optim.Optimizer],
    device:   torch.device,
    is_train: bool,
    clip_grad: float = 1.0,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Returns:
        avg_loss:    average total loss over batches
        task_losses: {task: avg raw loss}
        task_metrics:{task: AUC or RMSE}
    """
    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    task_loss_accum = {t: 0.0 for t in TASK_NAMES}
    task_loss_count  = {t: 0   for t in TASK_NAMES}

    all_preds  = {t: [] for t in TASK_NAMES}
    all_labels = {t: [] for t in TASK_NAMES}

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)

            preds = model(batch)

            # Build label + mask dicts from batch attributes
            labels, masks = {}, {}
            for task in TASK_NAMES:
                if hasattr(batch, task):
                    y    = getattr(batch, task).view(-1, 1).float()
                    mask = ~torch.isnan(y)
                    y_clean = y.clone()
                    y_clean[~mask] = 0.0
                    labels[task] = y_clean
                    masks[task]  = mask
                else:
                    # Task not in this batch
                    B = preds[TASK_NAMES[0]].shape[0]
                    labels[task] = torch.zeros(B, 1, device=device)
                    masks[task]  = torch.zeros(B, 1, dtype=torch.bool, device=device)

            total_loss, tl = loss_fn(preds, labels, masks)

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            epoch_loss += total_loss.item()

            # Accumulate for metrics
            for task in TASK_NAMES:
                mask = masks[task].squeeze()
                if mask.sum() > 0:
                    p = preds[task].detach().cpu().squeeze()[mask.cpu()]
                    y = labels[task].detach().cpu().squeeze()[mask.cpu()]
                    all_preds[task].extend(p.tolist())
                    all_labels[task].extend(y.tolist())
                    task_loss_accum[task] += tl[task].item()
                    task_loss_count[task] += 1

    n_batches  = len(loader)
    avg_loss   = epoch_loss / max(n_batches, 1)
    avg_task_losses = {
        t: task_loss_accum[t] / max(task_loss_count[t], 1)
        for t in TASK_NAMES
    }
    task_metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, avg_task_losses, task_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training + validation + checkpointing orchestrator.

    Args:
        model:      ADMETNet instance
        loss_fn:    MultiTaskLoss instance
        train_dl:   Training DataLoader
        val_dl:     Validation DataLoader
        cfg:        Config dict (from config.yaml)
        device:     torch.device
        output_dir: Where to save checkpoints and logs
    """

    def __init__(
        self,
        model:      ADMETNet,
        loss_fn:    MultiTaskLoss,
        train_dl:   DataLoader,
        val_dl:     DataLoader,
        cfg:        dict,
        device:     torch.device,
        output_dir: str = "models/",
    ):
        self.model      = model.to(device)
        self.loss_fn    = loss_fn
        self.train_dl   = train_dl
        self.val_dl     = val_dl
        self.cfg        = cfg
        self.device     = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        tc = cfg["training"]
        self.n_epochs     = tc["n_epochs"]
        self.patience     = tc["patience"]
        self.clip_grad    = tc.get("clip_grad", 1.0)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=tc["lr"],
            weight_decay=tc.get("weight_decay", 1e-4),
        )

        # Scheduler: cosine warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=tc.get("T_0", 10),
            T_mult=tc.get("T_mult", 2),
            eta_min=tc.get("lr_min", 1e-6),
        )

        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            **{f"val_{t}": [] for t in TASK_NAMES},
        }

        self.best_val_loss  = float("inf")
        self.patience_count = 0

    def _log(self, epoch: int, train_loss: float, val_loss: float,
             val_metrics: Dict[str, float], elapsed: float):
        print(f"\nEpoch {epoch:03d}/{self.n_epochs}  "
              f"[{elapsed:.1f}s]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Print classification AUCs
        cls_str = "  ".join(
            f"{t}={val_metrics[t]:.3f}"
            for t in CLASSIF_TASKS
            if not np.isnan(val_metrics.get(t, float("nan")))
        )
        reg_str = "  ".join(
            f"{t}={val_metrics[t]:.3f}"
            for t in REGRESSION_TASKS
            if not np.isnan(val_metrics.get(t, float("nan")))
        )
        if cls_str:
            print(f"  [AUC]  {cls_str}")
        if reg_str:
            print(f"  [RMSE] {reg_str}")

    def save_checkpoint(self, epoch: int, tag: str = "best"):
        path = os.path.join(self.output_dir, f"admet_net_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "opt_state":   self.optimizer.state_dict(),
            "val_loss":    self.best_val_loss,
            "history":     self.history,
            "cfg":         self.cfg,
        }, path)
        print(f"  💾 Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["opt_state"])
        self.history     = ckpt.get("history", self.history)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  ✅ Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    def fit(self):
        """Main training loop. Returns training history dict."""
        print("=" * 70)
        print("  ADMET-Net Training | Aurigene AIDD Group")
        print(f"  Device: {self.device}  |  Tasks: {len(TASK_NAMES)}")
        print(f"  Train batches: {len(self.train_dl)}  |  Val batches: {len(self.val_dl)}")
        print("=" * 70)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            train_loss, _, _ = run_epoch(
                self.model, self.loss_fn, self.train_dl,
                self.optimizer, self.device, is_train=True,
                clip_grad=self.clip_grad,
            )
            val_loss, _, val_metrics = run_epoch(
                self.model, self.loss_fn, self.val_dl,
                None, self.device, is_train=False,
            )

            self.scheduler.step(epoch)
            elapsed = time.time() - t0

            # Logging
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            for t in TASK_NAMES:
                self.history[f"val_{t}"].append(val_metrics.get(t, float("nan")))

            self._log(epoch, train_loss, val_loss, val_metrics, elapsed)

            # Early stopping + checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss  = val_loss
                self.patience_count = 0
                self.save_checkpoint(epoch, tag="best")
            else:
                self.patience_count += 1
                print(f"  ⏳ No improvement ({self.patience_count}/{self.patience})")

            if self.patience_count >= self.patience:
                print(f"\n  ⛔ Early stopping at epoch {epoch}.")
                break

        # Always save final checkpoint
        self.save_checkpoint(epoch, tag="final")
        print("\n✅ Training complete.")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ADMET-Net")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume",  type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and loss
    model, loss_fn = build_model(cfg)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataloaders
    train_dl, val_dl, test_dl = build_dataloaders(cfg)

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_dl=train_dl, val_dl=val_dl,
        cfg=cfg, device=device,
        output_dir=cfg.get("output_dir", "models/"),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.fit()

    # Save history for plotting
    import json
    hist_path = os.path.join(cfg.get("output_dir", "models/"), "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
