import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import safe_path_resolution, validate_file_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CNN training history")
    parser.add_argument("--hist", type=str, required=True, help="Path to CNN training history .npz file")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "visualize"), help="Visualization output directory")
    return parser.parse_args()


def model_name_from_history(hist_path):
    name = Path(hist_path).name
    for suffix in (
        "_cnn_gpu_training_history.npz",
        "_cnn_training_history.npz",
        "_training_history.npz",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(hist_path).stem


def load_history(hist_path):
    data = np.load(hist_path)
    required = ("train_loss", "val_loss", "train_acc", "val_acc")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"History file missing keys: {', '.join(missing)}")
    return tuple(np.asarray(data[key], dtype=np.float64) for key in required)


def save_cnn_visualizations(hist_path, outdir=None, model_name=None):
    hist_path = Path(hist_path)
    outdir = Path(outdir) if outdir is not None else PROJECT_ROOT / "visualize"
    model_name = model_name or model_name_from_history(hist_path)
    output_dir = outdir / f"{model_name}_cnn_visualize"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loss, val_loss, train_acc, val_acc = load_history(hist_path)
    epochs = np.arange(1, len(train_loss) + 1)

    overview_path = output_dir / f"{model_name}_cnn_training_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"CNN Training Overview - {model_name}", fontsize=16, fontweight="bold")

    axes[0, 0].plot(epochs, train_loss, color="#2563eb", linewidth=2, marker="o", markersize=4)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.28)

    axes[0, 1].plot(epochs, val_loss, color="#dc2626", linewidth=2, marker="o", markersize=4)
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.28)

    axes[1, 0].plot(epochs, train_acc * 100, color="#16a34a", linewidth=2, marker="o", markersize=4)
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.28)

    axes[1, 1].plot(epochs, val_acc * 100, color="#f97316", linewidth=2, marker="o", markersize=4)
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.28)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    comparison_path = output_dir / f"{model_name}_cnn_loss_accuracy.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"CNN Loss and Accuracy - {model_name}", fontsize=16, fontweight="bold")

    axes[0].plot(epochs, train_loss, color="#2563eb", linewidth=2, marker="o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, val_loss, color="#dc2626", linewidth=2, marker="o", markersize=4, label="Validation Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    axes[1].plot(epochs, train_acc * 100, color="#16a34a", linewidth=2, marker="o", markersize=4, label="Train Accuracy")
    axes[1].plot(epochs, val_acc * 100, color="#f97316", linewidth=2, marker="o", markersize=4, label="Validation Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.28)
    axes[1].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    gap_path = output_dir / f"{model_name}_cnn_generalization_gap.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    gap = (train_acc - val_acc) * 100
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
    ax.plot(epochs, gap, color="#7c3aed", linewidth=2, marker="o", markersize=4)
    ax.set_title("Generalization Gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy - Validation Accuracy (%)")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(gap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "overview": overview_path,
        "comparison": comparison_path,
        "gap": gap_path,
    }


def main():
    args = parse_args()
    try:
        hist_path = Path(safe_path_resolution(args.hist))
        validate_file_path(hist_path)
        outputs = save_cnn_visualizations(hist_path, args.outdir)
    except Exception as exc:
        print(f"Error during CNN visualization: {exc}")
        raise SystemExit(1)

    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
