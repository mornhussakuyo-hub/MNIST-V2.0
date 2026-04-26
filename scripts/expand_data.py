import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "mnist_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "mnist_train_expand.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Expand MNIST CSV with scale and shift augmentation")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input MNIST CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output expanded CSV")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Rows processed per chunk")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for shifted samples")
    parser.add_argument("--enlarge-scale", type=float, default=1.18, help="Scale used for enlarged digits")
    parser.add_argument("--shrink-scale", type=float, default=0.82, help="Scale used for shrunken digits")
    parser.add_argument("--shift-scale", type=float, default=0.78, help="Scale used before shifting digits")
    parser.add_argument("--max-shift", type=int, default=4, help="Maximum absolute shift in pixels")
    return parser.parse_args()


def digit_bbox(image, threshold=8):
    ys, xs = np.where(image > threshold)
    if len(xs) == 0:
        return 0, 0, 28, 28
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


def resize_digit(image, scale, shift_y=0, shift_x=0):
    y0, x0, y1, x1 = digit_bbox(image)
    crop = image[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape

    new_h = max(1, min(28, int(round(crop_h * scale))))
    new_w = max(1, min(28, int(round(crop_w * scale))))
    resized = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized = np.asarray(resized, dtype=np.uint8)

    center_y = (y0 + y1 - 1) / 2.0
    center_x = (x0 + x1 - 1) / 2.0
    top = int(round(center_y - new_h / 2.0 + shift_y))
    left = int(round(center_x - new_w / 2.0 + shift_x))
    top = max(0, min(28 - new_h, top))
    left = max(0, min(28 - new_w, left))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def make_shift(rng, max_shift):
    shift_y = 0
    shift_x = 0
    while shift_y == 0 and shift_x == 0:
        shift_y = int(rng.integers(-max_shift, max_shift + 1))
        shift_x = int(rng.integers(-max_shift, max_shift + 1))
    return shift_y, shift_x


def augment_row(row, rng, args):
    label = int(row[0])
    image = row[1:].astype(np.uint8).reshape(28, 28)
    shift_y, shift_x = make_shift(rng, args.max_shift)

    variants = (
        image,
        resize_digit(image, args.enlarge_scale),
        resize_digit(image, args.shrink_scale),
        resize_digit(image, args.shift_scale, shift_y=shift_y, shift_x=shift_x),
    )

    expanded = np.empty((len(variants), 785), dtype=np.uint8)
    expanded[:, 0] = label
    for index, variant in enumerate(variants):
        expanded[index, 1:] = variant.reshape(-1)
    return expanded


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    rng = np.random.default_rng(args.seed)
    total_input = 0
    total_output = 0
    first_chunk = True

    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=args.chunk_size), start=1):
        columns = chunk.columns
        rows = chunk.to_numpy(dtype=np.uint16)
        expanded_rows = [augment_row(row, rng, args) for row in rows]
        expanded = np.vstack(expanded_rows)
        expanded_df = pd.DataFrame(expanded, columns=columns)
        expanded_df.to_csv(output_path, mode="w" if first_chunk else "a", index=False, header=first_chunk)

        first_chunk = False
        total_input += len(rows)
        total_output += len(expanded)
        print(f"Chunk {chunk_index}: input {total_input}, output {total_output}")

    print(f"Expanded data saved to: {output_path}")
    print(f"Input rows: {total_input}")
    print(f"Output rows: {total_output}")


if __name__ == "__main__":
    main()
