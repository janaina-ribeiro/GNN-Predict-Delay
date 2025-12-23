from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch_geometric.loader import DataLoader

from .dataset_builder import DelayGraphDataset, temporal_split
from .gnn_model import DelayGNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: torch.device):
    return batch.to(device)


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    acc = accuracy_score(true, preds)
    f1_macro = f1_score(true, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(true, preds, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(true, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
    }


def _run_epoch(
    loader: DataLoader,
    model: DelayGNN,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
) -> Tuple[float, Dict[str, float]]:
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_collect = []
    target_collect = []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            loss = criterion(logits, batch.y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * batch.y.numel()
            total_samples += int(batch.y.numel())
            logits_collect.append(logits)
            target_collect.append(batch.y)
    if not logits_collect:
        return 0.0, {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision": 0.0, "recall": 0.0}
    stacked_logits = torch.cat(logits_collect, dim=0)
    stacked_targets = torch.cat(target_collect, dim=0)
    metrics = _compute_metrics(stacked_logits, stacked_targets)
    avg_loss = total_loss / max(total_samples, 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _flatten_labels(dataset: DelayGraphDataset, indices) -> torch.Tensor:
    labels = dataset.labels[indices]
    return labels.view(-1)


def _make_class_weights(dataset: DelayGraphDataset, split_indices) -> torch.Tensor:
    flattened = _flatten_labels(dataset, split_indices)
    counts = torch.bincount(flattened, minlength=2).float()
    weights = counts.sum() / (counts * len(counts))
    weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
    return weights


def train_model(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)
    dataset = DelayGraphDataset(
        data_dir=args.data_dir,
        links=args.links,
        window_size=args.window_size,
        horizon_minutes=args.horizon_minutes,
        min_corr=args.min_corr,
        limit_samples=args.limit_samples,
    )
    split = temporal_split(dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_indices = split.train.indices  # type: ignore[attr-defined]
    class_weights = _make_class_weights(dataset, train_indices).to(device)
    train_loader = DataLoader(split.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(split.val, batch_size=args.batch_size)
    test_loader = DataLoader(split.test, batch_size=args.batch_size)
    model = DelayGNN(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=2,
        conv_type=args.conv_type,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        _, train_metrics = _run_epoch(train_loader, model, criterion, optimizer, device, train_mode=True)
        _, val_metrics = _run_epoch(val_loader, model, criterion, optimizer, device, train_mode=False)
        val_f1 = val_metrics.get("f1_macro", 0.0)
        if val_f1 > best_metric:
            best_metric = val_f1
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
        print(
            f"Epoch {epoch:03d} | Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1_macro']:.4f} | Val F1 {val_metrics['f1_macro']:.4f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    _, test_metrics = _run_epoch(test_loader, model, criterion, optimizer, device, train_mode=False)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    artifact = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "links": dataset.link_order,
            "num_node_features": dataset.num_node_features,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "conv_type": args.conv_type,
            "gat_heads": args.gat_heads,
            "dropout": args.dropout,
            "window_size": args.window_size,
            "horizon_minutes": args.horizon_minutes,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "min_corr": args.min_corr,
            "timestamp_range": {
                "start": dataset.sample_timestamps[0].isoformat(),
                "end": dataset.sample_timestamps[-1].isoformat(),
                "count": len(dataset.sample_timestamps),
            },
            "best_val_f1": best_metric,
            "test_metrics": test_metrics,
        },
    }
    torch.save(artifact, args.model_path)
    print(f"Model saved to {args.model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN to classify link delay levels")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=["ac-am", "ac-ap", "ac-ba", "ac-ce"])
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--min-corr", type=float, default=0.3)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--conv-type", type=str, default="gat")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", type=Path, default=Path("test_gnn_delay_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
