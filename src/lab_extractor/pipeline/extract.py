from __future__ import annotations

import json
import logging
import time
from typing import Any

from PIL import Image

from lab_extractor.engine.medgemma import MedGemmaEngine
from lab_extractor.engine.prompts import EXTRACT_PROMPT
from lab_extractor.schemas.pipeline import StageResult

logger = logging.getLogger(__name__)

TYPE_HINTS = {
    "cbc": " Focus on: Hemoglobin, WBC, Platelets, RBC, Hematocrit, MCV, MCH, MCHC.",
    "metabolic": " Focus on: Glucose, BUN, Creatinine, Sodium, Potassium, Calcium, CO2, Chloride.",
    "lipid": " Focus on: Total Cholesterol, LDL, HDL, Triglycerides, VLDL.",
    "thyroid": " Focus on: TSH, Free T4, Free T3, Total T4, Total T3.",
    "urinalysis": " Focus on: pH, Specific Gravity, Protein, Glucose, Ketones, Blood, Nitrites, Leukocytes.",
    "liver": " Focus on: ALT, AST, ALP, Bilirubin Total, Bilirubin Direct, Albumin, Total Protein.",
    "other": "",
}


def extract_lab_data(
    engine: MedGemmaEngine,
    image: Image.Image,
    report_type: str,
    custom_prompt: str | None = None,
) -> StageResult:
    start = time.time()

    if custom_prompt is not None:
        prompt = custom_prompt
    else:
        prompt = EXTRACT_PROMPT + TYPE_HINTS.get(report_type, "")
    response = engine.query(image, prompt)

    tests_data: list[dict[str, Any]] = []
    patient_info: dict[str, Any] = {}
    extraction_confidence = 0.7
    parse_error: str | None = None

    try:
        data = json.loads(response)
        if isinstance(data, list):
            tests_data = data
        elif isinstance(data, dict):
            tests_data = data.get("tests", [])
            patient_info = data.get("patient_info", {}) or {}
        extraction_confidence = 0.87
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = str(exc)
        extraction_confidence = 0.3
        logger.warning("extract: JSON parse failed: %s", exc)

    reasoning = (
        f"Extracted {len(tests_data)} lab tests from {report_type} report. "
        f"Used {report_type}-specific prompt hints: '{TYPE_HINTS.get(report_type, 'none')}'. "
        f"Extraction confidence: {extraction_confidence:.0%}."
    )
    if parse_error:
        reasoning += f" WARNING: JSON parse error: {parse_error}"

    logger.info(
        "extract: %d tests extracted from %s report", len(tests_data), report_type
    )

    return StageResult(
        stage_name="extract",
        input_summary=f"Preprocessed {report_type} report image",
        output={
            "tests": tests_data,
            "patient_info": patient_info,
            "extraction_confidence": extraction_confidence,
            "test_count": len(tests_data),
            "parse_error": parse_error,
        },
        reasoning=reasoning,
        confidence=extraction_confidence,
        timing_seconds=time.time() - start,
    )
