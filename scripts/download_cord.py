#!/usr/bin/env python3
"""Download CORD-v2 receipt dataset from HuggingFace.

Dataset: CORD (Consolidated Receipt Dataset for Post-OCR Parsing)
HuggingFace: naver-clova-ix/cord-v2
License: CC BY 4.0
Splits: train (800), validation (100), test (100)

Saves:
  data/cord/{split}/images/{image_id:04d}.png
  data/cord/{split}/annotations/{image_id:04d}.json
  data/cord/manifest.json
"""

import argparse
import json
import sys
from pathlib import Path


def download(output_dir: Path, splits: list[str], force: bool) -> None:
    try:
        import datasets as hf_datasets
    except ImportError:
        print("Error: 'datasets' package not installed. Run: uv sync")
        sys.exit(1)

    manifest: dict[str, int] = {}

    for split in splits:
        img_dir = output_dir / split / "images"
        ann_dir = output_dir / split / "annotations"

        if img_dir.exists() and not force:
            existing = list(img_dir.glob("*.png"))
            if existing:
                print(
                    f"[{split}] Already downloaded ({len(existing)} images). Use --force to re-download."
                )
                manifest[split] = len(existing)
                continue

        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{split}] Downloading from HuggingFace naver-clova-ix/cord-v2 ...")
        ds = hf_datasets.load_dataset("naver-clova-ix/cord-v2", split=split)

        count = 0
        for i, row in enumerate(ds):
            img_path = img_dir / f"{i:04d}.png"
            ann_path = ann_dir / f"{i:04d}.json"

            # Save image as PNG
            pil_image = row["image"]
            pil_image.save(img_path, format="PNG")

            # Parse and save ground truth annotation
            gt_raw = row["ground_truth"]
            try:
                gt_data = json.loads(gt_raw)
            except (json.JSONDecodeError, TypeError):
                gt_data = {"raw": gt_raw}

            with ann_path.open("w") as f:
                json.dump({"image_id": i, "split": split, "gt": gt_data}, f, indent=2)

            count += 1
            if count % 50 == 0:
                print(f"  [{split}] {count}/{len(ds)} ...")

        print(f"[{split}] Done: {count} images saved to {img_dir}/")
        manifest[split] = count

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {"dataset": "naver-clova-ix/cord-v2", "splits": manifest}, f, indent=2
        )

    total = sum(manifest.values())
    print(f"\nCORD download complete: {total} images across {list(manifest.keys())}")
    print(f"Manifest: {manifest_path}")
    print(
        f"\nTo evaluate:\n  uv run python scripts/eval_cord.py --data-dir {output_dir} --dry-run"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CORD-v2 receipt dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all splits
  uv run python scripts/download_cord.py

  # Download test split only (100 images)
  uv run python scripts/download_cord.py --splits test

  # Re-download
  uv run python scripts/download_cord.py --force
""",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cord",
        help="Output directory (default: data/cord)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
        help="Which splits to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download(output_dir, args.splits, args.force)


if __name__ == "__main__":
    main()
