"""Lab Report Structured Data Extraction CLI.

Usage:
    uv run python -m lab_extractor --input <path> [options]
    uv run python -m lab_extractor --batch <dir> [options]
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab_extractor",
        description="Extract structured JSON from lab report images using MedGemma 1.5 4B",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        metavar="PATH",
        help="Path to a single lab report image (PNG/JPG/PDF)",
    )
    group.add_argument(
        "--batch", metavar="DIR", help="Directory of lab report images to process"
    )

    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSON output to file (default: stdout)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Use mock engine (no GPU required)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage-by-stage reasoning to stderr",
    )
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format: json (machine-readable) or summary (human-readable table)",
    )
    return parser


def format_summary(result) -> str:
    """Format PipelineResult as human-readable text table."""
    lines = []
    image_name = Path(result.image_path).name
    lines.append(f"Lab Report Analysis -- {image_name}")
    lines.append("=" * (len(lines[0])))

    report = result.final_report
    classify_stage = next(
        (s for s in result.stages if s.stage_name == "classify"), None
    )

    if classify_stage:
        report_type = classify_stage.output.get("report_type", "unknown").upper()
        confidence = classify_stage.output.get("confidence", 0)
        lines.append(f"Report Type: {report_type} (confidence: {confidence:.0%})")

    if report.lab_name:
        lines.append(f"Lab: {report.lab_name}")
    if report.report_date:
        lines.append(f"Date: {report.report_date}")
    if report.patient_info and report.patient_info.name:
        lines.append(f"Patient: {report.patient_info.name}")

    lines.append("")
    lines.append("Test Results:")

    # Table header
    col_widths = [20, 10, 12, 20, 12]
    header = f"| {'Test':<{col_widths[0]}} | {'Value':<{col_widths[1]}} | {'Unit':<{col_widths[2]}} | {'Reference':<{col_widths[3]}} | {'Flag':<{col_widths[4]}} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|{'-' * (col_widths[3] + 2)}|{'-' * (col_widths[4] + 2)}|"
    lines.append(header)
    lines.append(separator)

    for test in report.tests:
        ref_str = ""
        if test.reference_range:
            low = test.reference_range.low
            high = test.reference_range.high
            if low is not None and high is not None:
                ref_str = f"{low}-{high}"
            elif high is not None:
                ref_str = f"<{high}"
            elif low is not None:
                ref_str = f">{low}"

        flag_str = (test.flag or "").upper() if test.flag else ""

        row = (
            f"| {test.test_name:<{col_widths[0]}} "
            f"| {test.value:<{col_widths[1]}} "
            f"| {(test.unit or ''):<{col_widths[2]}} "
            f"| {ref_str:<{col_widths[3]}} "
            f"| {flag_str:<{col_widths[4]}} |"
        )
        lines.append(row)

    lines.append("")
    flag_stage = next((s for s in result.stages if s.stage_name == "flag"), None)
    flagged = flag_stage.output.get("flagged_count", 0) if flag_stage else 0
    critical = flag_stage.output.get("critical_count", 0) if flag_stage else 0
    lines.append(
        f"Pipeline: {len(result.stages)} stages completed in {result.total_time_seconds:.2f}s"
    )
    lines.append(f"Flagged: {flagged} abnormal ({critical} critical)")

    return "\n".join(lines)


def process_single(image_path: str, args, config):
    """Process a single image and return result dict."""
    from lab_extractor.pipeline.runner import run_pipeline

    result = run_pipeline(image_path, config)

    if args.verbose:
        for stage in result.stages:
            print(f"[{stage.stage_name}] {stage.reasoning}", file=sys.stderr)

    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    # Build config
    from lab_extractor.schemas.config import PipelineConfig

    config = PipelineConfig(dry_run=args.dry_run)

    results = []

    if args.input:
        # Single image mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 2

        result = process_single(str(input_path), args, config)
        results = [result]

    elif args.batch:
        # Batch mode
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            print(f"Error: batch directory not found: {args.batch}", file=sys.stderr)
            return 2

        image_files = sorted(
            list(batch_dir.glob("*.png"))
            + list(batch_dir.glob("*.jpg"))
            + list(batch_dir.glob("*.jpeg"))
            + list(batch_dir.glob("*.pdf"))
        )

        if not image_files:
            print(f"Error: no image files found in {args.batch}", file=sys.stderr)
            return 2

        for img_path in image_files:
            result = process_single(str(img_path), args, config)
            results.append(result)

    # Format output
    if args.format == "summary":
        output_text = "\n\n".join(format_summary(r) for r in results)
    else:
        # JSON output
        if len(results) == 1:
            output_data = results[0].model_dump()  # type: ignore
        else:
            output_data = [r.model_dump() for r in results]  # type: ignore
        output_text = json.dumps(output_data, indent=2, default=str)

    # Write output
    if args.output:
        Path(args.output).write_text(output_text)
    else:
        print(output_text)

    # Exit code based on success
    if any(not r.success for r in results):  # type: ignore
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
