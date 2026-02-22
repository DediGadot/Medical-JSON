# lab-extractor

> Extract structured, queryable data from clinical lab report images using MedGemma 1.5 4B.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![MedGemma 1.5 4B](https://img.shields.io/badge/model-MedGemma%201.5%204B-green)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow)
![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)

---

## TL;DR

Lab reports are the backbone of clinical medicine — yet they arrive as scanned PDFs, photographed paper slips, or proprietary printouts that no database can query. This tool takes a raw lab report image and returns a validated, structured JSON object with every test result, reference range, and abnormality flag extracted by MedGemma 1.5 4B. A 5-stage pipeline (classify → preprocess → extract → validate → flag) handles noise, layout variation, and plausibility checking automatically. No fine-tuning required; works on any standard lab report in under 5 seconds on a consumer GPU.

```bash
uv run python -m lab_extractor --input report.png --format summary
```

```
Lab Report Analysis -- report.png
==================================
Report Type: CBC (confidence: 92%)

| Test                 | Value      | Unit         | Reference            | Flag         |
|----------------------|------------|--------------|----------------------|--------------|
| Hemoglobin           | 6.8        | g/dL         | 13.5-17.5            | CRITICAL_LOW |
| WBC                  | 11.4       | 10^3/uL      | 4.5-11.0             | HIGH         |
| Platelets            | 187        | 10^3/uL      | 150-400              | NORMAL       |

Pipeline: 5 stages completed in 3.84s
Flagged: 2 abnormal (1 critical)
```

---

## The Problem

### Lab results drive clinical decisions — but the data is trapped

Clinical laboratory testing is the single most information-dense activity in modern medicine. Roughly **70% of all medical decisions** — whether to start a medication, escalate care, or discharge a patient — are informed by lab results. A complete blood count tells you if a patient is anemic or fighting an infection. A metabolic panel reveals kidney failure before symptoms appear. Liver enzymes catch drug toxicity weeks before the patient feels anything.

The volume is staggering. US labs alone process over **14 billion tests per year**. Each one produces a result that, ideally, gets reviewed, acted on, documented, and stored in a way that future clinicians can access. In practice, a large fraction of this data is immediately locked away in a format that is functionally useless for anything beyond a single human reading it once.

### The format problem

Most lab reports are generated as PDF printouts or thermal paper slips by aging laboratory information systems (LIS). They get scanned, faxed, photographed with a phone, or simply filed in a folder. The result is an image — a picture of data, not data itself. No database can query it. No EHR system can ingest it automatically. No algorithm can flag the critical hemoglobin before the on-call physician notices it at 3am.

This is not a niche problem. It is the **default state** of lab data in most of the world:

- **Private clinics and small hospitals** often run on standalone LIS software that exports PDFs with no structured API
- **Low- and middle-income countries (LMICs)** frequently rely entirely on paper-based reporting, where the physical lab slip is the only record of the result
- **Cross-institution transfers** routinely involve faxing or photographing reports because different hospital systems cannot talk to each other
- **Historical records** pre-dating EHR adoption are almost universally scanned images with no machine-readable index

### What gets lost

When lab data is trapped in images, the downstream failures are concrete:

**Missed critical values.** A hemoglobin of 5.2 g/dL requires immediate intervention. If the lab slip is scanned and filed rather than parsed, the alert never fires. Studies estimate that **critical lab value notification failures** contribute to preventable adverse events in 1-2% of hospitalized patients — a small percentage that represents tens of thousands of people annually.

**Duplicated testing.** Without structured historical data, clinicians cannot quickly check whether a test was run last week. The default is to order it again. Unnecessary duplicate testing in the US costs an estimated **$8 billion per year**.

**Population health blindness.** Aggregate analysis of lab trends — tracking HbA1c across a diabetic population, monitoring kidney function in patients on nephrotoxic drugs, detecting geographic clusters of thyroid disease — requires structured, queryable data. Image-locked records make this kind of analysis impossible without expensive manual abstraction.

**Research bottlenecks.** Clinical research depends on extracting structured phenotype data from patient records. Manual chart review to extract lab values costs $50-200 per patient chart. A study requiring 10,000 patients spends more on data extraction than on analysis.

### Why this is hard to solve

The obvious fix — structured data from the LIS — requires that the lab system support it, that the receiving system implement the interface, and that someone pays for the integration. In practice, HL7 and FHIR interfaces exist but are unevenly implemented, expensive to deploy, and unavailable for legacy systems or resource-constrained settings.

OCR-based approaches have been tried for decades. The problem is that lab reports have no standardized layout. A CBC from a hospital in Cairo looks completely different from one in São Paulo or Seoul. Column positions, test name abbreviations, reference range formats, and units vary by instrument vendor, region, and year. A regex-based or template-matching OCR system requires manual configuration for every layout variant — which is a maintenance nightmare and fails silently on edge cases.

What's needed is a system that understands medical content semantically, not just visually — one that knows that "Hgb" and "Hemoglobin" are the same test, that "g/dL" and "gm/dL" are the same unit, and that a value of 450 for hemoglobin is physically impossible regardless of what the image says.

### Why now

Vision-language models pre-trained on medical data have changed what's possible. **MedGemma 1.5 4B** was trained on a large corpus of medical imaging and clinical text, giving it the domain vocabulary and visual reasoning needed to handle layout variation and semantic normalization that defeated prior approaches. It can read a lab report the way a clinician reads one — understanding context, catching implausible values, and normalizing terminology — without requiring per-layout configuration.

This project is an implementation of that capability as a production-ready extraction pipeline: reliable enough to process real reports, fast enough to run at scale, and transparent enough (stage-by-stage reasoning) to audit when it gets something wrong.

---

## Solution

A 5-stage agentic pipeline, each stage a plain Python function with typed inputs and outputs:

```
image/PDF
    │
    ▼
┌─────────────┐
│  1. Classify │  → report_type: "cbc" | "metabolic" | "lipid" | "thyroid" | ...
└──────┬──────┘
       │
    ▼
┌──────────────────┐
│  2. Preprocess    │  → 896×896 RGB PIL Image (EXIF-corrected, contrast-enhanced)
└────────┬─────────┘
         │
      ▼
┌──────────────┐
│  3. Extract   │  → LabReport (raw MedGemma pass: names, values, units, ranges)
└──────┬───────┘
       │
    ▼
┌──────────────┐
│  4. Validate  │  → LabReport (second MedGemma pass: plausibility check + gap fill)
└──────┬───────┘
       │
    ▼
┌─────────────┐
│  5. Flag     │  → LabReport with flag on each LabTest (deterministic thresholds)
└──────┬──────┘
       │
    ▼
structured JSON
```

Each stage emits a `StageResult` with `reasoning`, `confidence`, and `time_seconds`. The final output is a fully typed `PipelineResult` (Pydantic v2).

---

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd medical
uv sync
```

For GPU inference (production):
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.8+
- HuggingFace account with access to `google/medgemma-1.5-4b-it`

```bash
# Authenticate with HuggingFace (required for MedGemma)
uv run huggingface-cli login
```

For development and testing, no GPU is needed — use `--dry-run`.

---

## Usage

### Single report → JSON

```bash
uv run python -m lab_extractor --input report.png --output result.json
```

### Single report → human-readable summary

```bash
uv run python -m lab_extractor --input report.png --format summary
```

### Batch directory

```bash
uv run python -m lab_extractor --batch ./reports/ --output results.json
```

### Without a GPU (dry-run mode)

```bash
uv run python -m lab_extractor --input report.png --dry-run
```

### Debug: see stage-by-stage reasoning

```bash
uv run python -m lab_extractor --input report.png --verbose 2>reasoning.log
```

**All flags:**

| Flag | Description |
|------|-------------|
| `--input PATH` | Single image (PNG/JPG/PDF) |
| `--batch DIR` | Directory of images |
| `--output PATH` | Write output to file (default: stdout) |
| `--format json\|summary` | Output format (default: json) |
| `--dry-run` | Use mock engine — no GPU, no model download |
| `--verbose` | Print stage reasoning to stderr |

**Exit codes:** `0` = success · `1` = pipeline error · `2` = file not found

---

## Output Schema

```json
{
  "success": true,
  "image_path": "report.png",
  "total_time_seconds": 3.84,
  "stages": [
    {
      "stage_name": "classify",
      "success": true,
      "output": {"report_type": "cbc", "confidence": 0.92},
      "reasoning": "Report header shows CBC panel with WBC, RBC, Hemoglobin...",
      "confidence": 0.92,
      "time_seconds": 0.71
    }
  ],
  "final_report": {
    "patient_info": {
      "name": "Jane Smith",
      "age": "34",
      "gender": "F",
      "patient_id": "MR-00482"
    },
    "lab_name": "Regional Medical Center",
    "report_date": "2026-01-15",
    "report_type": "cbc",
    "tests": [
      {
        "test_name": "Hemoglobin",
        "value": "6.8",
        "unit": "g/dL",
        "reference_range": {"low": 12.0, "high": 16.0},
        "flag": "critical_low"
      },
      {
        "test_name": "WBC",
        "value": "11.4",
        "unit": "10^3/uL",
        "reference_range": {"low": 4.5, "high": 11.0},
        "flag": "high"
      }
    ],
    "metadata": {}
  }
}
```

**Flag values:** `normal` · `high` · `low` · `critical_high` · `critical_low`

Critical thresholds (examples): hemoglobin < 7 g/dL → `critical_low`; potassium > 6.5 mEq/L → `critical_high`.

---

## Supported Tests

Reference ranges are built-in for all major panels:

| Panel | Tests |
|-------|-------|
| CBC | Hemoglobin (M/F), WBC, Platelets, RBC (M/F), Hematocrit (M/F) |
| Metabolic | Glucose, BUN, Creatinine, Sodium, Potassium, Calcium, CO2 |
| Lipid | Total Cholesterol, LDL, HDL, Triglycerides |
| Liver Function | ALT, AST, ALP, Bilirubin (total) |
| Thyroid | TSH, Free T4, Free T3 |

Tests not in this table are still extracted — they just won't have a reference range auto-filled.

---

## Evaluation

Run the self-consistency evaluator on a directory of images:

```bash
uv run python -c "
from lab_extractor.evaluation import evaluate
results = evaluate('data/samples/', dry_run=True)
print(f'Self-consistency: {results[\"self_consistency\"]:.2f}')
print(f'Sample size: {results[\"sample_size\"]}')
"
```

**Self-consistency scoring** runs each image through the pipeline twice and compares outputs field-by-field:
- Numeric values match within 1% tolerance
- String values require exact match after normalization
- Missing tests are penalized (+3 per missing test)
- Score ≥ 0.85 indicates reliable extraction

In dry-run mode (deterministic mock), score = 1.0 by construction.

---

## Development

```bash
# Run all tests (no GPU required)
uv run pytest tests/ -v

# Run one test module
uv run pytest tests/test_pipeline.py -v

# Generate placeholder sample images
uv run python scripts/create_samples.py

# Download real dataset (Mendeley, CC BY 4.0)
uv run python scripts/download_dataset.py --output-dir data/raw/ --sample
```

### Project structure

```
medical/
├── src/lab_extractor/
│   ├── __main__.py          # CLI (argparse, batch mode, format rendering)
│   ├── schemas/
│   │   ├── lab_report.py    # LabTest, LabReport, PatientInfo, ReferenceRange
│   │   ├── pipeline.py      # StageResult, PipelineResult
│   │   ├── config.py        # PipelineConfig dataclass
│   │   └── reference_ranges.py  # Built-in ranges + lookup()
│   ├── engine/
│   │   ├── medgemma.py      # MedGemmaEngine (pipeline() wrapper + dry-run mock)
│   │   └── prompts.py       # CLASSIFY_PROMPT, EXTRACT_PROMPT, VALIDATE_PROMPT
│   ├── preprocessing/
│   │   └── image_ops.py     # preprocess_image(), load_image(), convert_pdf_to_images()
│   ├── pipeline/
│   │   ├── classify.py      # Stage 1
│   │   ├── preprocess.py    # Stage 2
│   │   ├── extract.py       # Stage 3
│   │   ├── validate.py      # Stage 4
│   │   ├── flag.py          # Stage 5
│   │   └── runner.py        # run_pipeline() orchestrator
│   └── evaluation/
│       ├── metrics.py       # self_consistency_score(), field_level_metrics()
│       └── evaluate.py      # evaluate() — runs pipeline twice per image
├── tests/                   # 27 pytest tests, all pass with --dry-run
├── scripts/
│   ├── create_samples.py    # Generate 896×896 placeholder PNGs
│   └── download_dataset.py  # Mendeley dataset downloader
└── data/
    └── samples/             # 5 placeholder lab report images (896×896 PNG)
```

### Key design constraints

- **No abstract base classes** — every pipeline stage is a plain function
- **No web UI** — CLI only (`argparse`)
- **No fine-tuning** — MedGemma base model as-is
- **No YAML configs** — single `PipelineConfig` dataclass
- **No utils/helpers modules** — code lives in the module that uses it
- **Image size is always 896×896** — MedGemma's required input resolution

---

## Limitations

- **GPU required for production.** MedGemma 1.5 4B needs ~8GB VRAM. `--dry-run` works without any GPU.
- **English only.** Prompts and reference ranges are in English. Non-English reports will partially extract but may miss terminology normalization.
- **US-standard reference ranges.** Built-in ranges follow US clinical guidelines. Ranges differ by country, lab instrument, and patient demographics (age, sex). The validate stage fills gaps from built-in ranges — which may not match the ranges printed on the report.
- **Handwritten values.** The pipeline handles printed and scanned typed reports well. Handwritten annotations may require OCR preprocessing before the pipeline.
- **No ground truth validation.** Self-consistency measures reproducibility, not accuracy. There is no labeled benchmark for this exact dataset.

---

## Medical Disclaimer

**This tool is for research purposes only. It is not FDA-approved or CE-marked. It must not be used to make clinical decisions without review by a qualified healthcare professional. Extracted values and flags should always be verified against the original report. The authors accept no liability for clinical outcomes.**

---

## Dataset

[Mendeley Clinical Lab Reports](https://data.mendeley.com/datasets/bygfmk4rx9/2) (DOI: 10.17632/bygfmk4rx9.2) — CC BY 4.0.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
