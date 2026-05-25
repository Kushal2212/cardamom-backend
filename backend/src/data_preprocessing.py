import os
import shutil
import argparse
import random
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Image quality check will be skipped.")
    print("Install with: pip install opencv-python")

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMG_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(path: str) -> bool:
    """Return True if the image can be opened and is not blurry."""
    if not CV2_AVAILABLE:
        return True
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 50  # threshold for blurriness


def split_and_copy(raw_dir: str, output_dir: str, seed: int = 42):
    """
    raw_dir structure expected:
        raw_dir/
            healthy/  img1.jpg  img2.jpg ...
            chhirke/   ...
            leaf_blight/ ...

    Creates:
        output_dir/train/healthy/  ...
        output_dir/validation/healthy/ ...
        output_dir/test/healthy/   ...
    """
    random.seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    classes = [d for d in raw_path.iterdir() if d.is_dir()]
    if not classes:
        print(f"No sub-folders found in {raw_dir}")
        return

    total_copied = 0
    for class_dir in classes:
        class_name = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        # Filter blurry / corrupt images
        valid_images = [str(img) for img in images if is_valid_image(str(img))]
        skipped = len(images) - len(valid_images)
        if skipped:
            print(f"  [{class_name}] Skipped {skipped} blurry/corrupt images.")

        random.shuffle(valid_images)
        n = len(valid_images)
        if n == 0:
            print(f"  [{class_name}] No valid images found. Skipping.")
            continue

        n_train = max(1, int(n * SPLITS["train"]))
        n_val = max(1, int(n * SPLITS["validation"]))
        # rest goes to test
        splits_map = {
            "train":      valid_images[:n_train],
            "validation": valid_images[n_train: n_train + n_val],
            "test":       valid_images[n_train + n_val:],
        }

        for split_name, files in splits_map.items():
            dest_dir = out_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = dest_dir / Path(src).name
                shutil.copy2(src, dst)
            total_copied += len(files)
            print(f"  [{class_name}] {split_name:10s}: {len(files)} images")

    print(f"\nDone! {total_copied} images copied to '{output_dir}'")


def resize_dataset(dataset_dir: str):
    """Resize all images in dataset to IMG_SIZE in-place."""
    if not CV2_AVAILABLE:
        print("OpenCV not available — skipping resize.")
        return

    count = 0
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    resized = cv2.resize(img, IMG_SIZE)
                    cv2.imwrite(fpath, resized)
                    count += 1
    print(f"Resized {count} images to {IMG_SIZE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess cardamom dataset")

    parser.add_argument(
        "--raw_dir",
        default="raw_images",
        help="Folder containing class sub-folders with raw images",
    )

    parser.add_argument(
        "--output_dir",
        default="dataset",
        help="Output folder for train/validation/test splits",
    )

    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize images after splitting",
    )

    args, unknown = parser.parse_known_args()

    print(f"Splitting '{args.raw_dir}' → '{args.output_dir}' ...")
    split_and_copy(args.raw_dir, args.output_dir)

    if args.resize:
        print("\nResizing images...")
        resize_dataset(args.output_dir)



