import argparse
import sys
from pathlib import Path

import torch
from .hyperparameter_optimizer import HyperparameterOptimizer
from optimization_analysis import OptimizationAnalyzer

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))


def run_quick_optimization():
    print("=" * 60)
    print("QUICK EXAMPLE OPTIMIZATION")
    print("=" * 60)
    print("This is a demonstration with reduced parameters for quick testing.")
    print("For full optimization, adjust the parameters as needed.\n")

    data_dir = PROJECT_ROOT / "datasets_generated"
    output_dir = SCRIPT_DIR / "optimization_results"

    base_args = argparse.Namespace(
        data_dir=data_dir,
        links=None,
        window_size=6,
        horizon_minutes=30,
        min_corr=0.3,
        limit_samples=2000,
        use_traceroute=True,
        topology_weight=0.4,
        train_ratio=0.7,
        val_ratio=0.15,
        epochs=15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_channels=64,
        num_layers=2,
        conv_type="gat",
        dropout=0.2,
        gat_heads=4,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        loss_fn="cross_entropy",
        focal_alpha=0.25,
        focal_gamma=2.0,
        balance_mode="class",
        delay_threshold=None,
        delay_percentile=85.0,
        scheduler="none",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_t0=10,
        undersample_ratio=None,
        patience=5,
        model_path=output_dir / "temp_model.pt",
    )

    if not base_args.data_dir.exists():
        print(f"Data directory not found: {base_args.data_dir}")
        print(" Adjust the 'data_dir' parameter in the script.")
        return

    print(f"Data directory: {base_args.data_dir}")
    print(f"Device: {base_args.device}")
    print(f"Sample limit: {base_args.limit_samples}")

    optimizer = HyperparameterOptimizer(
        base_args=base_args,
        n_trials=15,
        timeout=1800,
        study_name="quick_optimization_example",
        metric="val_f1_macro",
        pruner_patience=3,
    )

    try:
        print("\nStarting optimization...")

        best_params, best_value = optimizer.optimize()

        print("\nOptimization finished!")
        print(f"Best F1 Score: {best_value:.4f}")

        study_path = optimizer.save_study()

        print("\nTraining final model with best hyperparameters...")
        best_model_path = optimizer.train_best_model()

        print("\nResults analysis:")
        analyzer = OptimizationAnalyzer(study_path)
        analyzer.print_summary()
        analyzer.analyze_top_trials(5)

        analysis_dir = SCRIPT_DIR / "optimization_results" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        history_path = analysis_dir / "optimization_history.png"
        params_path = analysis_dir / "parameter_distributions.png"

        analyzer.plot_optimization_history(history_path)
        analyzer.plot_parameter_distributions(params_path)

        report_path = analysis_dir / "analysis_report.json"
        analyzer.export_summary_report(report_path)

        print("\n" + "=" * 60)
        print("OPTIMIZATION SUCCESSFULLY COMPLETED!")
        print("=" * 60)
        print(f"Best model: {best_model_path}")
        print(f"Study saved: {study_path}")
        print(f"Report: {report_path}")
        print(f"Plots: {analysis_dir}")
        print("\nNext steps:")
        print("   1. Analyze the generated plots")
        print("   2. Use the final model for predictions")
        print("   3. Run a longer optimization if needed")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        optimizer.save_study()

    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("   Check:")
        print("   - If the dataset is in the correct directory")
        print("   - If dependencies are installed")
        print("   - If there is enough data for training")
        raise


def run_full_optimization():
    print("=" * 60)
    print("FULL OPTIMIZATION")
    print("=" * 60)
    print("This optimization may take several hours.")
    print("Configure the parameters as needed.\n")

    data_dir = PROJECT_ROOT / "datasets_generated"
    output_dir = SCRIPT_DIR / "optimization_results"

    base_args = argparse.Namespace(
        data_dir=data_dir,
        links=None,
        window_size=6,
        horizon_minutes=30,
        min_corr=0.3,
        limit_samples=None,
        use_traceroute=True,
        topology_weight=0.4,
        train_ratio=0.7,
        val_ratio=0.15,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_channels=64,
        num_layers=2,
        conv_type="gat",
        dropout=0.2,
        gat_heads=4,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        loss_fn="focal",
        focal_alpha=0.75,
        focal_gamma=2.5,
        balance_mode="both",
        delay_threshold=None,
        delay_percentile=85.0,
        scheduler="plateau",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_t0=10,
        undersample_ratio=3.0,
        patience=15,
        model_path=output_dir / "temp_model.pt",
    )

    optimizer = HyperparameterOptimizer(
        base_args=base_args,
        n_trials=20,
        timeout=14400,
        study_name="full_gnn_optimization",
        metric="val_f1_macro",
        pruner_patience=5,
    )

    try:
        best_params, best_value = optimizer.optimize()
        study_path = optimizer.save_study()
        best_model_path = optimizer.train_best_model()

        print("\nAnalyzing results...")
        analyzer = OptimizationAnalyzer(study_path)
        analyzer.print_summary()
        analyzer.analyze_top_trials(10)

        analysis_dir = SCRIPT_DIR / "optimization_results" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        history_path = analysis_dir / "optimization_history.png"
        params_path = analysis_dir / "parameter_distributions.png"

        analyzer.plot_optimization_history(history_path)
        analyzer.plot_parameter_distributions(params_path)

        report_path = analysis_dir / "analysis_report.json"
        analyzer.export_summary_report(report_path)

        print("\n" + "=" * 60)
        print("FULL OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"Best model: {best_model_path}")
        print(f"Study saved: {study_path}")
        print(f"Report: {report_path}")
        print(f"Plots: {analysis_dir}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Saving progress...")
        optimizer.save_study()

        if len(optimizer.study.trials) > 0:
            print("Training the best model found so far...")
            try:
                best_model_path = optimizer.train_best_model()
                print(f"Best model saved at: {best_model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Optimization mode: 'quick' for testing, 'full' for production",
    )

    args = parser.parse_args()

    if args.mode == "quick":
        run_quick_optimization()
    else:
        run_full_optimization()
