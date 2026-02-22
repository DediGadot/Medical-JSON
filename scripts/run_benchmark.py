#!/usr/bin/env python3
"""Run reproducible benchmark suite for lab-extractor pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

torch = importlib.import_module("torch")


def load_pipeline_config() -> Any:
    schemas_config = importlib.import_module("lab_extractor.schemas.config")
    return schemas_config.PipelineConfig


def load_run_pipeline() -> Any:
    runner = importlib.import_module("lab_extractor.pipeline.runner")
    return runner.run_pipeline


def load_evaluate() -> Any:
    evaluate_mod = importlib.import_module("lab_extractor.evaluation.evaluate")
    return evaluate_mod.evaluate


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}


@dataclass
class RunRow:
    image: str
    repeat: int
    success: bool
    error: str | None
    pipeline_time_seconds: float
    wall_time_seconds: float
    tests_extracted: int
    flagged_count: int
    critical_count: int
    stage_timings: dict[str, float]
    stage_confidence: dict[str, float]
    gpu: dict[str, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lab_extractor pipeline")
    parser.add_argument(
        "--input-dir",
        default="data/samples",
        help="Directory containing report images/PDFs",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_runs",
        help="Directory where benchmark artifacts will be saved",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Limit number of input files (0 means all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per image",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs before measurement",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Torch dtype for model inference",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Enable 4-bit quantization path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max generated tokens for MedGemma calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Validation retries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic mock engine (no GPU/model required)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluate() self-consistency pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PyTorch random seed",
    )
    return parser.parse_args()


def list_input_files(input_dir: Path, sample_size: int) -> list[Path]:
    files = [
        p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS
    ]
    if sample_size > 0:
        return files[:sample_size]
    return files


def gpu_snapshot() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


def system_info(args: argparse.Namespace) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "device_requested": args.device,
        "torch_dtype": args.torch_dtype,
        "quantize_4bit": args.quantize_4bit,
        "dry_run": args.dry_run,
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_device_count": torch.cuda.device_count(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def main() -> int:
    args = parse_args()
    _ = torch.manual_seed(int(args.seed))

    PipelineConfig = load_pipeline_config()
    run_pipeline = load_run_pipeline()
    evaluate = load_evaluate()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = list_input_files(input_dir, args.sample_size)
    if not files:
        raise ValueError(f"No supported files found in {input_dir}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        quantize_4bit=args.quantize_4bit,
    )

    print(f"Benchmark run_id={run_id}")
    print(
        f"Input files: {len(files)} | repeats: {args.repeats} | warmup: {args.warmup_runs}"
    )
    print(f"Config: {config}")

    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    warmup_target = files[0]
    for _ in range(args.warmup_runs):
        _ = run_pipeline(str(warmup_target), config)

    rows: list[RunRow] = []
    wall_start = time.perf_counter()
    success_count = 0

    for image_path in files:
        for repeat_idx in range(args.repeats):
            if torch.cuda.is_available() and args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = run_pipeline(str(image_path), config)
            if torch.cuda.is_available() and args.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            flagged_count = int(
                next(
                    (
                        s.output.get("flagged_count", 0)
                        for s in result.stages
                        if s.stage_name == "flag"
                    ),
                    0,
                )
            )
            critical_count = int(
                next(
                    (
                        s.output.get("critical_count", 0)
                        for s in result.stages
                        if s.stage_name == "flag"
                    ),
                    0,
                )
            )
            row = RunRow(
                image=image_path.name,
                repeat=repeat_idx,
                success=result.success,
                error=result.error,
                pipeline_time_seconds=float(result.total_time_seconds),
                wall_time_seconds=float(t1 - t0),
                tests_extracted=len(result.final_report.tests),
                flagged_count=flagged_count,
                critical_count=critical_count,
                stage_timings={
                    s.stage_name: float(s.timing_seconds) for s in result.stages
                },
                stage_confidence={
                    s.stage_name: float(s.confidence) for s in result.stages
                },
                gpu=gpu_snapshot(),
            )
            rows.append(row)
            success_count += int(result.success)

    wall_total = time.perf_counter() - wall_start

    pipeline_times = [r.pipeline_time_seconds for r in rows]
    wall_times = [r.wall_time_seconds for r in rows]
    tests_counts = [r.tests_extracted for r in rows]

    aggregate = {
        "total_runs": len(rows),
        "success_rate": success_count / len(rows),
        "avg_pipeline_time_seconds": statistics.mean(pipeline_times),
        "p50_pipeline_time_seconds": percentile(pipeline_times, 50),
        "p95_pipeline_time_seconds": percentile(pipeline_times, 95),
        "avg_wall_time_seconds": statistics.mean(wall_times),
        "p50_wall_time_seconds": percentile(wall_times, 50),
        "p95_wall_time_seconds": percentile(wall_times, 95),
        "avg_tests_extracted": statistics.mean(tests_counts),
        "images_per_second": len(rows) / wall_total if wall_total > 0 else 0.0,
        "tests_per_second": sum(tests_counts) / wall_total if wall_total > 0 else 0.0,
        "benchmark_wall_time_seconds": wall_total,
    }

    summary: dict[str, Any] = {
        "run_id": run_id,
        "system": system_info(args),
        "config": asdict(config),
        "inputs": {
            "input_dir": str(input_dir),
            "file_count": len(files),
            "files": [p.name for p in files],
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
        },
        "aggregate": aggregate,
    }

    if not args.skip_eval:
        summary["self_consistency_eval"] = evaluate(
            image_dir=str(input_dir),
            config=config,
            sample_size=min(len(files), 10),
        )

    results_path = out_dir / "benchmark_results.json"
    runs_path = out_dir / "per_run.jsonl"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with runs_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")

    print("\nBenchmark complete")
    print(f"Results: {results_path}")
    print(f"Per-run: {runs_path}")
    print(
        "Avg pipeline time: "
        f"{aggregate['avg_pipeline_time_seconds']:.3f}s | "
        f"P95: {aggregate['p95_pipeline_time_seconds']:.3f}s | "
        f"Throughput: {aggregate['images_per_second']:.2f} img/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
