#!/usr/bin/env python3
"""Preflight checks before running full GPU benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

torch = importlib.import_module("torch")


def default_pipeline_config() -> dict[str, Any]:
    schemas_config = importlib.import_module("lab_extractor.schemas.config")
    return asdict(schemas_config.PipelineConfig())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU preflight for lab-extractor")
    parser.add_argument(
        "--input-dir",
        default="data/samples",
        help="Directory with benchmark input files",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write JSON preflight report",
    )
    parser.add_argument(
        "--require-4060",
        action="store_true",
        help="Return non-zero if GPU name does not contain '4060'",
    )
    return parser.parse_args()


def supported_file_count(input_dir: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    return sum(1 for p in input_dir.iterdir() if p.suffix.lower() in exts)


def hf_token_present() -> bool:
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))


@dataclass
class PreflightReport:
    python_version: str
    platform: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    input_dir: str
    input_exists: bool
    input_supported_files: int
    hf_token_present: bool
    recommended_config: dict[str, Any]
    gpu_name: str | None = None
    gpu_count: int | None = None
    gpu_total_memory_gb: float | None = None
    bf16_supported: bool | None = None


def build_report(input_dir: Path) -> PreflightReport:
    report = PreflightReport(
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        input_dir=str(input_dir),
        input_exists=input_dir.is_dir(),
        input_supported_files=supported_file_count(input_dir)
        if input_dir.is_dir()
        else 0,
        hf_token_present=hf_token_present(),
        recommended_config=default_pipeline_config(),
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.gpu_name = torch.cuda.get_device_name(0)
        report.gpu_count = torch.cuda.device_count()
        report.gpu_total_memory_gb = round(props.total_memory / (1024**3), 2)
        report.bf16_supported = torch.cuda.is_bf16_supported()
    return report


def print_report(report: PreflightReport) -> None:
    print("=== lab-extractor GPU preflight ===")
    print(f"Python: {report.python_version}")
    print(f"Torch: {report.torch_version} | CUDA: {report.torch_cuda_version}")
    print(f"CUDA available: {report.cuda_available}")
    if report.cuda_available:
        print(
            f"GPU: {report.gpu_name} | VRAM: {report.gpu_total_memory_gb} GB | "
            f"BF16: {report.bf16_supported}"
        )
    print(
        f"Input dir: {report.input_dir} | exists={report.input_exists} | "
        f"supported files={report.input_supported_files}"
    )
    print(f"HF token present: {report.hf_token_present}")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    report = build_report(input_dir)
    print_report(report)

    failures: list[str] = []
    warnings: list[str] = []

    if not report.input_exists:
        failures.append("Input directory does not exist")
    elif report.input_supported_files == 0:
        failures.append("Input directory has no supported files (.png/.jpg/.jpeg/.pdf)")

    if not report.cuda_available:
        failures.append("CUDA is not available")
    else:
        gpu_name = report.gpu_name or ""
        if args.require_4060 and "4060" not in gpu_name:
            failures.append(f"Expected RTX 4060, got: {gpu_name}")
        gpu_memory = report.gpu_total_memory_gb or 0.0
        if gpu_memory < 8:
            warnings.append("GPU memory < 8GB; MedGemma may fail without quantization")

    if not report.hf_token_present:
        warnings.append(
            "No HF token in environment (set HF_TOKEN or HUGGINGFACE_TOKEN if model access fails)"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"Saved report: {output_path}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"- {f}")
        return 1

    print("\nPASS: system is ready for GPU benchmark")
    print("Suggested next command:")
    print(
        "uv run python scripts/run_benchmark.py --input-dir data/samples --repeats 5 "
        "--warmup-runs 2 --device cuda --torch-dtype bfloat16"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
