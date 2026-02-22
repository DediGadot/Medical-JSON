"""Evaluation metrics for lab report extraction pipeline."""

from __future__ import annotations
import logging
from lab_extractor.schemas.lab_report import LabReport, LabTest
from lab_extractor.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def self_consistency_score(result1: PipelineResult, result2: PipelineResult) -> float:
    """Compute self-consistency between two pipeline runs on the same image.

    Compares extracted tests field by field:
    - test_name: exact match (case-insensitive)
    - value: fuzzy match (within 1% for numeric, exact for non-numeric)
    - unit: exact match (case-insensitive)

    Score = matching_fields / total_fields (0.0 to 1.0)
    Returns 1.0 if both reports have 0 tests (consistent empty).
    """
    tests1 = result1.final_report.tests
    tests2 = result2.final_report.tests

    if not tests1 and not tests2:
        return 1.0

    if not tests1 or not tests2:
        return 0.0

    # Build lookup by test_name (lowercase)
    lookup1 = {t.test_name.lower(): t for t in tests1}
    lookup2 = {t.test_name.lower(): t for t in tests2}

    all_names = set(lookup1.keys()) | set(lookup2.keys())

    matching = 0
    total = 0

    for name in all_names:
        t1 = lookup1.get(name)
        t2 = lookup2.get(name)

        if t1 is None or t2 is None:
            total += 3  # test_name, value, unit all missing
            continue

        # test_name match (already matched by key)
        matching += 1
        total += 1

        # value match (fuzzy for numeric)
        total += 1
        try:
            v1 = float(t1.value.replace("<", "").replace(">", "").strip())
            v2 = float(t2.value.replace("<", "").replace(">", "").strip())
            if v1 == 0 and v2 == 0:
                matching += 1
            elif v1 != 0 and abs(v1 - v2) / abs(v1) <= 0.01:
                matching += 1
        except (ValueError, AttributeError):
            if t1.value == t2.value:
                matching += 1

        # unit match
        total += 1
        u1 = (t1.unit or "").lower().strip()
        u2 = (t2.unit or "").lower().strip()
        if u1 == u2:
            matching += 1

    return matching / total if total > 0 else 1.0


def field_level_metrics(predicted: LabReport, ground_truth: LabReport) -> dict:
    """Compute field-level precision, recall, F1 for lab test extraction.

    Compares predicted tests against ground truth.
    Returns dict with per-field and aggregate metrics.

    Note: In the absence of ground truth annotations, this can be used
    with manually annotated samples.
    """
    pred_tests = {t.test_name.lower(): t for t in predicted.tests}
    gt_tests = {t.test_name.lower(): t for t in ground_truth.tests}

    # Test name detection metrics
    pred_names = set(pred_tests.keys())
    gt_names = set(gt_tests.keys())

    tp_names = len(pred_names & gt_names)
    fp_names = len(pred_names - gt_names)
    fn_names = len(gt_names - pred_names)

    precision = tp_names / (tp_names + fp_names) if (tp_names + fp_names) > 0 else 0.0
    recall = tp_names / (tp_names + fn_names) if (tp_names + fn_names) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Value accuracy for matched tests
    value_matches = 0
    matched_count = len(pred_names & gt_names)
    for name in pred_names & gt_names:
        pred_val = pred_tests[name].value
        gt_val = gt_tests[name].value
        try:
            pv = float(pred_val.replace("<", "").replace(">", "").strip())
            gv = float(gt_val.replace("<", "").replace(">", "").strip())
            if gv == 0:
                if pv == 0:
                    value_matches += 1
            elif abs(pv - gv) / abs(gv) <= 0.05:  # 5% tolerance
                value_matches += 1
        except (ValueError, AttributeError):
            if pred_val == gt_val:
                value_matches += 1

    value_accuracy = value_matches / matched_count if matched_count > 0 else 0.0

    return {
        "test_name": {"precision": precision, "recall": recall, "f1": f1},
        "value_accuracy": value_accuracy,
        "predicted_count": len(pred_tests),
        "ground_truth_count": len(gt_tests),
        "matched_count": matched_count,
    }


def report_level_metrics(results: list[PipelineResult]) -> dict:
    """Compute report-level aggregate metrics across multiple pipeline results.

    Returns:
    - success_rate: fraction of reports where pipeline succeeded
    - avg_tests_extracted: average number of tests per report
    - avg_pipeline_time: average total pipeline time in seconds
    - reports_with_tests: fraction of reports with at least 1 test
    """
    if not results:
        return {
            "success_rate": 0.0,
            "avg_tests_extracted": 0.0,
            "avg_pipeline_time": 0.0,
            "reports_with_tests": 0.0,
            "total_reports": 0,
        }

    n = len(results)
    successes = sum(1 for r in results if r.success)
    total_tests = sum(len(r.final_report.tests) for r in results)
    total_time = sum(r.total_time_seconds for r in results)
    with_tests = sum(1 for r in results if len(r.final_report.tests) > 0)

    return {
        "success_rate": successes / n,
        "avg_tests_extracted": total_tests / n,
        "avg_pipeline_time": total_time / n,
        "reports_with_tests": with_tests / n,
        "total_reports": n,
    }
