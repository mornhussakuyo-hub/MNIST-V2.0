import argparse
import os
import sys
import time
from pathlib import Path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_numpy_threads():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-threads", type=int, default=None)
    known_args, _ = parser.parse_known_args()

    if known_args.num_threads is not None:
        if known_args.num_threads < 1:
            parser.error("--num-threads must be a positive integer")
        for env_var in THREAD_ENV_VARS:
            os.environ[env_var] = str(known_args.num_threads)
        return known_args.num_threads

    if any(os.environ.get(env_var) for env_var in THREAD_ENV_VARS):
        return None

    num_threads = max(1, os.cpu_count() or 1)
    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = str(num_threads)
    return num_threads


CONFIGURED_NUM_THREADS = configure_numpy_threads()

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_model import TwoConvNet
from common import safe_path_resolution, validate_file_path
from utils import load_data, normalize_data, one_hot_encode, split_train_val
from visualize_cnn import save_cnn_visualizations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a two-convolution MNIST model")
    parser.add_argument("--traindt", type=str, default=str(PROJECT_ROOT / "data" / "mnist_train.csv"), help="Path to train data CSV file")
    parser.add_argument("--testdt", type=str, default=str(PROJECT_ROOT / "data" / "mnist_test.csv"), help="Path to test data CSV file")
    parser.add_argument("--model-name", type=str, default="cnn_model", help="Model name (default: cnn_model)")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models"), help="Model save directory")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results"), help="Results save directory")
    parser.add_argument("--epochs", type=int, default=3, help="Train epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--conv1-filters", type=int, default=8, help="First convolution filters (default: 8)")
    parser.add_argument("--conv2-filters", type=int, default=16, help="Second convolution filters (default: 16)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proportion for validation set (default: 0.2)")
    parser.add_argument("--usage-ratio", type=float, default=1.0, help="Proportion used from training data (default: 1.0)")
    parser.add_argument("--reg-rate", type=float, default=0.0, help="L2 regularization rate (default: 0.0)")
    parser.add_argument("--num-threads", type=int, default=CONFIGURED_NUM_THREADS, help="CPU threads used by NumPy/BLAS matrix operations")
    parser.add_argument("--no-save", action="store_true", help="Do not save model or history")
    return parser.parse_args()


def prepare_images(images):
    return images.astype(np.float32).reshape(-1, 1, 28, 28)


def main():
    args = parse_args()

    print("\nCNN training start...")
    print(f"Network: Conv({args.conv1_filters}) -> Pool -> Conv({args.conv2_filters}) -> Pool -> FC(10)")
    print(f"Hyperparameter: LR={args.learning_rate}, Epochs={args.epochs}, Batch={args.batch_size}")
    print(f"Regularization rate: {args.reg_rate}")
    print(f"NumPy/BLAS thread setting: {args.num_threads if args.num_threads is not None else 'environment default'}")

    try:
        train_path = safe_path_resolution(args.traindt)
        test_path = safe_path_resolution(args.testdt)
        validate_file_path(train_path)
        validate_file_path(test_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Please check if the path was right")
        raise SystemExit(1)

    train_images, train_labels, test_images, test_labels = load_data(train_path, test_path)

    train_images = normalize_data(train_images, "train_images")
    test_images = normalize_data(test_images, "test_images")
    train_images, train_labels, val_images, val_labels = split_train_val(
        train_images,
        train_labels,
        args.val_ratio,
        args.usage_ratio,
    )

    train_images = prepare_images(train_images)
    val_images = prepare_images(val_images)
    test_images = prepare_images(test_images)
    train_labels = one_hot_encode(train_labels).astype(np.float32)
    val_labels = one_hot_encode(val_labels).astype(np.float32)
    test_labels = one_hot_encode(test_labels).astype(np.float32)

    model = TwoConvNet(
        model_name=args.model_name,
        conv1_filters=args.conv1_filters,
        conv2_filters=args.conv2_filters,
    )

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    num_train = train_images.shape[0]
    if num_train < 1:
        raise ValueError("No training samples available")

    print("\nStart CNN training.")
    for epoch in range(args.epochs):
        start_time = time.time()
        indices = np.random.permutation(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            x_batch = train_images[batch_indices]
            y_batch = train_labels[batch_indices]

            probs = model.forward(x_batch)
            loss = model.compute_loss(probs, y_batch, args.reg_rate)
            model.backward(x_batch, y_batch, args.reg_rate)
            model.update_parameters(args.learning_rate)

            epoch_loss += loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_acc = model.accuracy(train_images, train_labels)
        val_probs = model.forward(val_images)
        val_loss = model.compute_loss(val_probs, val_labels, args.reg_rate)
        val_acc = model.accuracy(val_images, val_labels)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch: {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {avg_train_loss:.4f} (train)/{val_loss:.4f} (valid) | "
            f"Acc: {train_acc * 100:.2f}% (train)/{val_acc * 100:.2f}% (valid) | "
            f"Time: {elapsed:.1f}s"
        )

    print("\nRunning on test data...")
    test_acc = model.accuracy(test_images, test_labels)
    print(f"Test data acc: {test_acc * 100:.2f}%")

    if args.no_save:
        print("No saving was on.")
        return

    results_dir = Path(args.results_dir)
    models_dir = Path(args.model_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{args.model_name}_cnn_training_history.npz"
    np.savez(
        results_path,
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        train_acc=train_acc_history,
        val_acc=val_acc_history,
    )
    print(f"Training history has been saved at: {results_path}")

    print("\nGenerating CNN training visualizations...")
    visualization_paths = save_cnn_visualizations(results_path, PROJECT_ROOT / "visualize", args.model_name)
    for label, path in visualization_paths.items():
        print(f"{label.capitalize()} visualization saved to: {path}")

    model_path = models_dir / f"{args.model_name}.npz"
    model.save(model_path)
    print(f"CNN model has been saved at: {model_path}")


if __name__ == "__main__":
    main()
