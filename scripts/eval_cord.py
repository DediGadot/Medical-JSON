"""Evaluate lab-extractor pipeline against CORD-v2 receipt ground truth.

Usage:
    uv run python scripts/eval_cord.py [--data-dir data/cord] [--split test]
        [--output results_cord.json] [--limit N] [--dry-run] [--device cuda]

Produces per-image and aggregate precision/recall/F1 metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lab_extractor.cord import (
    CORD_EXTRACT_PROMPT,
    evaluate_cord_from_pipeline_output,
    load_cord_annotation,
)
from lab_extractor.pipeline.runner import run_pipeline
from lab_extractor.schemas.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _aggregate(per_image: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean metrics across all images."""
    if not per_image:
        return {}

    keys = ["precision", "recall", "f1", "price_accuracy"]
    totals: dict[str, float] = {k: 0.0 for k in keys}
    counts: dict[str, int] = {k: 0 for k in keys}

    for r in per_image:
        if r.get("error"):
            continue
        metrics = r.get("metrics", {})
        item_name = metrics.get("item_name", {})
        for k in ["precision", "recall", "f1"]:
            if k in item_name:
                totals[k] += item_name[k]
                counts[k] += 1
        if "price_accuracy" in metrics:
            totals["price_accuracy"] += metrics["price_accuracy"]
            counts["price_accuracy"] += 1

    return {k: totals[k] / counts[k] if counts[k] else 0.0 for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline on CORD-v2 receipt dataset"
    )
    parser.add_argument(
        "--data-dir",
        default="data/cord",
        help="Root directory of downloaded CORD data (default: data/cord)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output",
        default="results_cord.json",
        help="Output JSON file (default: results_cord.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock engine — no GPU required (for CI/dev)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype for model (default: bfloat16)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"

    if not images_dir.exists():
        logger.error(
            "Images directory not found: %s\n"
            "Run: uv run python scripts/download_cord.py --splits %s",
            images_dir,
            args.split,
        )
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        logger.error("No PNG images found in %s", images_dir)
        sys.exit(1)

    logger.info(
        "Evaluating %d images from CORD-%s split (dry_run=%s)",
        len(image_paths),
        args.split,
        args.dry_run,
    )

    config = PipelineConfig(
        device=args.device,
        torch_dtype=args.torch_dtype,
        dry_run=args.dry_run,
        custom_extract_prompt=CORD_EXTRACT_PROMPT,
    )

    per_image: list[dict[str, Any]] = []

    for i, img_path in enumerate(image_paths, 1):
        image_id = img_path.stem  # e.g. "00000"
        ann_path = annotations_dir / f"{image_id}.json"

        logger.info("[%d/%d] Processing %s", i, len(image_paths), img_path.name)

        record: dict[str, Any] = {"image_id": image_id, "image_path": str(img_path)}

        # Load ground truth
        if not ann_path.exists():
            logger.warning("Annotation missing for %s — skipping", image_id)
            record["error"] = "annotation_missing"
            per_image.append(record)
            continue

        try:
            gt = load_cord_annotation(str(ann_path))
        except Exception as exc:
            logger.warning("Failed to load annotation %s: %s", ann_path, exc)
            record["error"] = f"annotation_load_error: {exc}"
            per_image.append(record)
            continue

        # Run pipeline
        try:
            result = run_pipeline(str(img_path), config)
        except Exception as exc:
            logger.warning("Pipeline failed on %s: %s", img_path.name, exc)
            record["error"] = f"pipeline_error: {exc}"
            per_image.append(record)
            continue

        if not result.success:
            record["error"] = f"pipeline_failed: {result.error}"
            per_image.append(record)
            continue

        # Extract stage output
        extract_stage = next(
            (s for s in result.stages if s.stage_name == "extract"), None
        )
        if extract_stage is None:
            record["error"] = "extract_stage_missing"
            per_image.append(record)
            continue

        pipeline_output = extract_stage.output  # has "tests", "patient_info", etc.

        # Evaluate
        try:
            metrics = evaluate_cord_from_pipeline_output(pipeline_output, gt)
        except Exception as exc:
            logger.warning("Evaluation failed on %s: %s", image_id, exc)
            record["error"] = f"eval_error: {exc}"
            per_image.append(record)
            continue

        record["metrics"] = metrics
        record["ground_truth_count"] = metrics.get("ground_truth_count", 0)
        record["predicted_count"] = metrics.get("predicted_count", 0)
        per_image.append(record)

        logger.info(
            "  F1=%.3f  P=%.3f  R=%.3f  price_acc=%.3f  (%d/%d items matched)",
            metrics["item_name"]["f1"],
            metrics["item_name"]["precision"],
            metrics["item_name"]["recall"],
            metrics["price_accuracy"],
            metrics["matched_count"],
            metrics["ground_truth_count"],
        )

    aggregate = _aggregate(per_image)
    errors = sum(1 for r in per_image if r.get("error"))

    output = {
        "split": args.split,
        "n_images": len(image_paths),
        "n_errors": errors,
        "n_evaluated": len(image_paths) - errors,
        "dry_run": args.dry_run,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== CORD-v2 Evaluation Results ({args.split} split) ===")
    print(f"Images evaluated : {len(image_paths) - errors}/{len(image_paths)}")
    if errors:
        print(f"Errors           : {errors}")
    print(f"Precision        : {aggregate.get('precision', 0):.3f}")
    print(f"Recall           : {aggregate.get('recall', 0):.3f}")
    print(f"F1               : {aggregate.get('f1', 0):.3f}")
    print(f"Price accuracy   : {aggregate.get('price_accuracy', 0):.3f}")
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
