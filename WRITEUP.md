# Lab Report to Structured JSON Extraction — Competition Writeup

**Kaggle MedGemma Impact Challenge | Problem #3**

---

## Problem Statement

Clinical laboratory reports are generated in enormous volumes daily — yet they remain locked in unstructured formats: scanned PDFs, photographed paper reports, and proprietary lab system printouts. Extracting structured data from these documents is a prerequisite for EHR integration, population health analytics, and clinical decision support. Manual transcription is error-prone and expensive. Automated extraction using vision-language models offers a scalable alternative.

This project builds a production-ready pipeline that takes a raw lab report image and returns a validated, structured JSON object containing patient information, test results, reference ranges, and abnormality flags.

---

## Overall Solution

The system uses **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`) — a multimodal model pre-trained on medical imaging and clinical text — as the core reasoning engine. Rather than a single monolithic prompt, we decompose the task into a **5-stage agentic pipeline** where each stage has a focused responsibility and produces a typed output.

### Why MedGemma?

MedGemma's pre-training on medical data gives it domain-specific vocabulary (test names, units, reference ranges) and visual understanding of clinical document layouts. This reduces hallucination compared to general-purpose VLMs and eliminates the need for fine-tuning.

---

## Technical Details

Each stage is a plain Python function — no abstract base classes, no framework overhead.

### Stage 1: Classify
Determines the report type (CBC, metabolic panel, lipid panel, thyroid, urinalysis, liver function, or other). This gates downstream extraction: a CBC report uses different reference ranges than a lipid panel.

**Input**: 896×896 preprocessed image  
**Output**: `{"report_type": "cbc", "confidence": 0.92, "reasoning": "..."}`

### Stage 2: Preprocess
Loads the image, auto-orients via EXIF metadata, converts to RGB, resizes with aspect-ratio-preserving white padding to exactly 896×896 pixels, and applies contrast enhancement (factor 1.4) to improve scanned document legibility.

**Input**: Raw file path (PNG/JPG/PDF)  
**Output**: PIL Image at 896×896 RGB

### Stage 3: Extract
The core extraction stage. Sends the preprocessed image to MedGemma with a structured prompt requesting a JSON array of lab tests. Each test includes name, value, unit, and reference range as printed on the report.

**Input**: 896×896 image + EXTRACT_PROMPT  
**Output**: `LabReport` Pydantic model with `tests: list[LabTest]`

### Stage 4: Validate
Runs a second MedGemma pass with the extracted JSON as context, asking the model to verify plausibility (e.g., hemoglobin of 450 g/dL is impossible). Corrects obvious extraction errors and fills in missing reference ranges from a curated lookup table of 40+ common tests.

**Input**: Extracted JSON + VALIDATE_PROMPT  
**Output**: Corrected `LabReport`

### Stage 5: Flag
Deterministic post-processing — no model call. Compares each test value against reference ranges to assign flags: `normal`, `high`, `low`, `critical_high`, `critical_low`. Critical thresholds are hardcoded (e.g., hemoglobin < 7 g/dL = critical_low).

**Input**: Validated `LabReport`  
**Output**: `LabReport` with `flag` field populated on each `LabTest`

---

## HAI-DEF Model Usage

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},   # PIL Image, 896x896
    {"type": "text", "text": prompt},
]}]

output = pipe(text=messages, max_new_tokens=2000)
```

All prompts end with "Respond ONLY with valid JSON" to ensure parseable output. The engine includes a JSON parse fallback that returns a structured error object rather than crashing.

## Agentic Workflow

The 5-stage pipeline implements an agentic reasoning pattern where each stage produces typed outputs that feed into the next:

1. **Classify** → report type (gates downstream extraction)
2. **Preprocess** → normalized 896×896 image (improves model input quality)
3. **Extract** → raw JSON from MedGemma (first reasoning pass)
4. **Validate** → corrected JSON with filled ranges (second reasoning pass, error correction)
5. **Flag** → final report with abnormality flags (deterministic post-processing)

Each stage's reasoning is logged to stderr in verbose mode, enabling debugging and transparency into model decisions.
---

## Evaluation

### Self-Consistency Scoring
Each image is processed twice independently. The two `LabReport` outputs are compared field-by-field:
- **Numeric values**: fuzzy match within 1% tolerance
- **String values**: exact match after normalization
- **Missing tests**: penalized (+3 per missing test across both runs)

Score = `matched_fields / total_fields`. A score ≥ 0.85 indicates reliable extraction.

### Field-Level Metrics
When ground-truth annotations are available, we compute precision and recall per field type (test_name, value, unit, flag). This enables targeted debugging of which extraction step is failing.

In dry-run mode (deterministic mock engine), self-consistency = 1.0 by construction.

---

## Medical Safety

The flag stage uses conservative thresholds derived from standard clinical references:
- **Critical low**: hemoglobin < 7 g/dL, WBC < 2.0 × 10³/µL, platelets < 50 × 10³/µL
- **Critical high**: potassium > 6.5 mEq/L, sodium > 155 mEq/L, glucose > 500 mg/dL

Flagged tests are surfaced prominently in both JSON and summary output formats. The pipeline never suppresses or normalizes critical values.

---

## Limitations & Future Work

- **GPU required for production**: MedGemma 1.5 4B requires ~8GB VRAM. The `--dry-run` flag enables testing without GPU using a deterministic mock engine.
- **English-only**: Prompts and reference ranges are in English. Multi-language support would require prompt translation.
- **Handwritten reports**: The preprocessing pipeline handles printed and scanned reports well; handwritten values may require additional OCR preprocessing.
- **Fine-tuning opportunity**: Domain-specific fine-tuning on annotated lab reports would improve extraction accuracy, particularly for non-standard report layouts.

---

## Medical Disclaimer

**This system is for research and educational purposes only. It is not validated for clinical use and must not be used to make medical decisions. Always consult a qualified healthcare professional for interpretation of laboratory results.**

---

## Dataset

Mendeley Clinical Lab Reports dataset (DOI: 10.17632/bygfmk4rx9.2), licensed CC BY 4.0.
