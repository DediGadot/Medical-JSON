# Benchmark Runbook

This runbook is for a full benchmark session on a GPU machine (target: RTX 4060).

## 1) Preflight

Run this first on the GPU box:

```bash
uv sync
uv run python scripts/preflight_gpu.py --input-dir data/samples --require-4060 --output benchmark_runs/preflight.json
```

Preflight checks:
- CUDA availability
- GPU name and VRAM
- BF16 support
- input file availability
- HuggingFace token presence

## 2) Full benchmark (recommended)

```bash
uv run python scripts/run_benchmark.py \
  --input-dir data/samples \
  --output-dir benchmark_runs \
  --repeats 5 \
  --warmup-runs 2 \
  --device cuda \
  --torch-dtype bfloat16
```

## 3) Optional variant sweeps

### Quantized run (memory-focused)

```bash
uv run python scripts/run_benchmark.py \
  --input-dir data/samples \
  --output-dir benchmark_runs \
  --repeats 5 \
  --warmup-runs 2 \
  --device cuda \
  --torch-dtype bfloat16 \
  --quantize-4bit
```

### FP16 run (latency comparison)

```bash
uv run python scripts/run_benchmark.py \
  --input-dir data/samples \
  --output-dir benchmark_runs \
  --repeats 5 \
  --warmup-runs 2 \
  --device cuda \
  --torch-dtype float16
```

## 4) Artifacts produced

Each run creates `benchmark_runs/<run_id>/` with:

- `benchmark_results.json` (aggregate report)
- `per_run.jsonl` (one JSON object per image repeat)

### `benchmark_results.json` includes

- `system`: python, torch, CUDA, GPU metadata
- `config`: exact `PipelineConfig` used
- `inputs`: file list, repeats, warmups
- `aggregate`:
  - `success_rate`
  - `avg_pipeline_time_seconds`
  - `p50_pipeline_time_seconds`
  - `p95_pipeline_time_seconds`
  - `avg_wall_time_seconds`
  - `p50_wall_time_seconds`
  - `p95_wall_time_seconds`
  - `avg_tests_extracted`
  - `images_per_second`
  - `tests_per_second`
- `self_consistency_eval` (unless `--skip-eval`)

### `per_run.jsonl` per record includes

- input image name + repeat index
- success/error
- pipeline and wall-clock time
- extracted/flagged/critical test counts
- per-stage timings and confidences
- GPU memory snapshot (`allocated`, `reserved`, peak values)

## 5) Interpreting results quickly

- Prefer `p95_pipeline_time_seconds` for reliability (not just average).
- If throughput is low and GPU peaks are low, bottleneck is likely preprocessing or Python overhead.
- If `success_rate < 1.0`, inspect `error` fields in `per_run.jsonl` first.
- Compare non-quantized vs `--quantize-4bit` for speed/memory tradeoff on 4060.

## 6) Ground-truth benchmark extension (next)

Current benchmark gives performance + self-consistency. To measure extraction accuracy:

1. Build a labeled set (`image -> expected LabReport JSON`)
2. Evaluate with `field_level_metrics()` from `src/lab_extractor/evaluation/metrics.py`
3. Track precision/recall/F1 by panel and by test name
