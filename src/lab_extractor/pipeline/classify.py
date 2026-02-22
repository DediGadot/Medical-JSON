from __future__ import annotations

import json
import logging
import time

from PIL import Image

from lab_extractor.engine.medgemma import MedGemmaEngine
from lab_extractor.engine.prompts import CLASSIFY_PROMPT
from lab_extractor.schemas.pipeline import StageResult

logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = {
    "cbc",
    "metabolic",
    "lipid",
    "thyroid",
    "urinalysis",
    "liver",
    "other",
}


def classify_report(engine: MedGemmaEngine, image: Image.Image) -> StageResult:
    """Stage 1: Classify the lab report type using MedGemma.

    Returns StageResult with output={"report_type": str, "confidence": float}
    """
    start = time.time()

    response = engine.query(image, CLASSIFY_PROMPT)

    try:
        data = json.loads(response)
        report_type = data.get("report_type", "other")
        confidence = float(data.get("confidence", 0.5))
        reasoning_from_model = data.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        report_type = "other"
        confidence = 0.3
        reasoning_from_model = f"JSON parse failed, raw response: {response[:100]}"

    if report_type not in VALID_REPORT_TYPES:
        report_type = "other"

    reasoning = (
        f"Classified as '{report_type}' with {confidence:.0%} confidence. "
        f"Model reasoning: {reasoning_from_model}"
    )

    logger.info("classify: %s (%.0f%%)", report_type, confidence * 100)

    return StageResult(
        stage_name="classify",
        input_summary=f"Lab report image ({image.size[0]}x{image.size[1]} px)",
        output={"report_type": report_type, "confidence": confidence},
        reasoning=reasoning,
        confidence=confidence,
        timing_seconds=time.time() - start,
    )
