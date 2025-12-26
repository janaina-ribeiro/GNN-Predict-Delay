from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from test_gnn.dataset_builder import DelayGraphDataset


def build_and_save_cache(
    data_dir: Path,
    output_dir: Path,
    links: Optional[List[str]] = None,
    window_size: int = 6,
    horizon_minutes: int = 30,
    min_corr: float = 0.3,
    delay_percentile: float = 85.0,
    use_traceroute: bool = True,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"BUILDING DATASET CACHE - Horizon {horizon_minutes}min")
    print(f"{'=' * 60}")

    timing = {}

    start = time.perf_counter()
    print(f"[INFO] Loading data from: {data_dir}")

    dataset = DelayGraphDataset(
        data_dir=data_dir,
        links=links,
        window_size=window_size,
        horizon_minutes=horizon_minutes,
        min_corr=min_corr,
        limit_samples=None,
        delay_percentile=delay_percentile,
        use_traceroute=use_traceroute,
    )

    timing["build_time_s"] = time.perf_counter() - start
    print(f"[INFO] Dataset built in {timing['build_time_s']:.2f}s")
    print(f"[INFO] Total samples: {len(dataset)}")
    print(f"[INFO] Links: {len(dataset.link_order)}")
    print(f"[INFO] Node features: {dataset.num_node_features}")

    cache_data = {
        "features": dataset.features,
        "labels": dataset.labels,
        "edge_index": dataset.edge_index,
        "edge_weight": dataset.edge_weight,
        "metadata": {
            "links": dataset.link_order,
            "num_samples": len(dataset),
            "num_node_features": dataset.num_node_features,
            "num_nodes": dataset.num_nodes,
            "window_size": window_size,
            "horizon_minutes": horizon_minutes,
            "min_corr": min_corr,
            "delay_threshold": dataset.delay_threshold,
            "delay_percentile": delay_percentile,
            "use_traceroute": use_traceroute,
            "data_dir": str(data_dir),
        },
        "timestamps": dataset.sample_timestamps,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / f"dataset_cache_horizon_{horizon_minutes}min.pt"

    start_save = time.perf_counter()
    torch.save(cache_data, cache_file)
    timing["save_time_s"] = time.perf_counter() - start_save

    file_size_mb = cache_file.stat().st_size / (1024 * 1024)

    print(f"[INFO] Cache saved at: {cache_file}")
    print(f"[INFO] File size: {file_size_mb:.2f} MB")
    print(f"[INFO] Save time: {timing['save_time_s']:.2f}s")

    return {
        "cache_file": str(cache_file),
        "num_samples": len(dataset),
        "num_links": len(dataset.link_order),
        "num_features": dataset.num_node_features,
        "file_size_mb": file_size_mb,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Builds and saves dataset cache for fast prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets_generated_prediction"),
        help="Directory with data CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_gnn/dataset_cache"),
        help="Directory to save the cache",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 30],
        help="Prediction horizons in minutes",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Temporal window size",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.3,
        help="Minimum correlation to create edges",
    )
    parser.add_argument(
        "--delay-percentile",
        type=float,
        default=85.0,
        help="Percentile for high delay threshold",
    )
    parser.add_argument(
        "--links",
        nargs="*",
        default=None,
        help="Specific links (None = all)",
    )
    parser.add_argument(
        "--no-traceroute",
        action="store_true",
        help="Disable traceroute features",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BUILD DATASET CACHE")
    print("=" * 60)
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Window size: {args.window_size}")
    print(f"  Min correlation: {args.min_corr}")
    print(f"  Delay percentile: {args.delay_percentile}")
    print(f"  Use traceroute: {not args.no_traceroute}")
    print("=" * 60 + "\n")

    results = []
    total_start = time.perf_counter()

    for horizon in args.horizons:
        result = build_and_save_cache(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            links=args.links,
            window_size=args.window_size,
            horizon_minutes=horizon,
            min_corr=args.min_corr,
            delay_percentile=args.delay_percentile,
            use_traceroute=not args.no_traceroute,
        )
        results.append(result)

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("SUMMARY - CACHES CREATED")
    print("=" * 60)
    for i, (horizon, result) in enumerate(zip(args.horizons, results)):
        print(f"\n  Horizon {horizon}min:")
        print(f"    File: {result['cache_file']}")
        print(f"    Samples: {result['num_samples']}")
        print(f"    Links: {result['num_links']}")
        print(f"    Features: {result['num_features']}")
        print(f"    Size: {result['file_size_mb']:.2f} MB")
        print(f"    Build time: {result['timing']['build_time_s']:.2f}s")

    print(f"\n  Total time: {total_time:.2f}s")
    print("=" * 60 + "\n")

    print("[INFO] Use the caches for prediction with:")
    print(
        "  python -m test_gnn.predict_delay --model-path <model.pt> --cache-path <cache.pt>"
    )


if __name__ == "__main__":
    main()
