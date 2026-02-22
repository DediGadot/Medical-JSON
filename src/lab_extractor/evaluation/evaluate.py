"""Evaluation harness for the lab report extraction pipeline."""

from __future__ import annotations
import logging
from pathlib import Path
from lab_extractor.schemas.config import PipelineConfig
from lab_extractor.evaluation.metrics import (
    self_consistency_score,
    report_level_metrics,
)

logger = logging.getLogger(__name__)


def evaluate(image_dir: str, config: PipelineConfig, sample_size: int = 10) -> dict:
    """Evaluate the pipeline on a directory of images.

    For each image (up to sample_size):
    - Run pipeline twice for self-consistency scoring
    - Collect all results for report-level metrics

    Returns evaluation report as dict with:
    - self_consistency: average self-consistency score across all images
    - report_metrics: aggregate report-level metrics
    - per_image: list of per-image results
    """
    from lab_extractor.pipeline.runner import run_pipeline

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Find image files
    image_files = sorted(
        list(image_dir_path.glob("*.png"))
        + list(image_dir_path.glob("*.jpg"))
        + list(image_dir_path.glob("*.jpeg"))
        + list(image_dir_path.glob("*.pdf"))
    )[:sample_size]

    if not image_files:
        logger.warning("No image files found in %s", image_dir)
        return {
            "self_consistency": 0.0,
            "report_metrics": report_level_metrics([]),
            "per_image": [],
            "sample_size": 0,
        }

    all_results = []
    consistency_scores = []
    per_image = []

    for img_path in image_files:
        logger.info("Evaluating: %s", img_path.name)

        # Run twice for self-consistency
        result1 = run_pipeline(str(img_path), config)
        result2 = run_pipeline(str(img_path), config)

        score = self_consistency_score(result1, result2)
        consistency_scores.append(score)
        all_results.append(result1)

        per_image.append(
            {
                "image": img_path.name,
                "success": result1.success,
                "tests_extracted": len(result1.final_report.tests),
                "self_consistency": score,
                "pipeline_time": result1.total_time_seconds,
            }
        )

        logger.info(
            "  consistency=%.2f, tests=%d", score, len(result1.final_report.tests)
        )

    avg_consistency = (
        sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    )

    return {
        "self_consistency": avg_consistency,
        "report_metrics": report_level_metrics(all_results),
        "per_image": per_image,
        "sample_size": len(image_files),
    }
