from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .dataset_builder import DelayGraphDataset
from .gnn_model import DelayGNN


def load_artifact(path: Path, device: torch.device):
    artifact = torch.load(path, map_location=device)
    if "model_state_dict" not in artifact or "metadata" not in artifact:
        raise RuntimeError("Model artifact is missing required keys")
    return artifact


def build_dataset(artifact: dict, args: argparse.Namespace) -> DelayGraphDataset:
    meta = artifact["metadata"]
    links = args.links if args.links else meta["links"]
    dataset = DelayGraphDataset(
        data_dir=args.data_dir,
        links=links,
        window_size=meta["window_size"],
        horizon_minutes=meta["horizon_minutes"],
        min_corr=meta["min_corr"],
        limit_samples=None,
        column_delay="Atraso",
    )
    return dataset


def build_model(artifact: dict, device: torch.device) -> DelayGNN:
    meta = artifact["metadata"]
    model = DelayGNN(
        in_channels=meta["num_node_features"],
        hidden_channels=meta["hidden_channels"],
        num_layers=meta["num_layers"],
        num_classes=2,
        conv_type=meta["conv_type"],
        dropout=meta["dropout"],
        gat_heads=meta["gat_heads"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _resolve_timestamp(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    artifact = load_artifact(args.model_path, device)
    dataset = build_dataset(artifact, args)
    model = build_model(artifact, device)
    if args.last_n <= 0 or args.last_n > len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = list(range(len(dataset) - args.last_n, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    records = []
    link_order: List[str] = dataset.link_order
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            timestamp = _resolve_timestamp(data.timestamp)
            edge_weight = getattr(data, "edge_weight", None)
            edge_weight = edge_weight.to(device) if edge_weight is not None else None
            logits = model(data.x.to(device), data.edge_index.to(device), edge_weight)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1)
            targets = data.y.cpu()
            for node_idx, link in enumerate(link_order):
                records.append(
                    {
                        "sample": indices[batch_idx],
                        "timestamp": timestamp,
                        "link": link,
                        "target": int(targets[node_idx].item()),
                        "prediction": int(preds[node_idx].item()),
                        "prob_low": float(probs[node_idx, 0].item()),
                        "prob_high": float(probs[node_idx, 1].item()),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        print("No predictions generated")
        return
    frame = frame.sort_values(["timestamp", "link"])
    frame["correct"] = frame["target"] == frame["prediction"]
    overall_acc = frame["correct"].mean()
    per_link = frame.groupby("link")["correct"].mean().to_dict()
    print(frame.tail(args.print_rows).to_string(index=False))
    print(f"Overall accuracy: {overall_acc:.3f}")
    print("Per-link accuracy:")
    for link, value in per_link.items():
        print(f"  {link}: {value:.3f}")
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run delay predictions with a trained GNN model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=None)
    parser.add_argument("--last-n", type=int, default=24)
    parser.add_argument("--print-rows", type=int, default=20)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
