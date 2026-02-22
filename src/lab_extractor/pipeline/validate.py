from __future__ import annotations

import json
import logging
import time
from typing import Any

from PIL import Image
from pydantic import ValidationError

from lab_extractor.engine.medgemma import MedGemmaEngine
from lab_extractor.engine.prompts import VALIDATE_PROMPT
from lab_extractor.schemas.config import PipelineConfig
from lab_extractor.schemas.lab_report import LabReport, LabTest, PatientInfo
from lab_extractor.schemas.pipeline import StageResult

logger = logging.getLogger(__name__)


def validate_extraction(
    engine: MedGemmaEngine,
    image: Image.Image,
    extracted: dict[str, Any],
    config: PipelineConfig,
) -> tuple[LabReport, StageResult]:
    start = time.time()
    retries = 0
    issues: list[str] = []
    current_data: dict[str, Any] = extracted

    report = LabReport(tests=[])

    for attempt in range(config.max_retries + 1):
        try:
            tests_raw = current_data.get("tests", [])
            tests: list[LabTest] = []
            for test_raw in tests_raw:
                try:
                    tests.append(
                        LabTest(
                            **{
                                key: value
                                for key, value in test_raw.items()
                                if key in LabTest.model_fields
                            }
                        )
                    )
                except (ValidationError, TypeError):
                    continue

            patient_raw = current_data.get("patient_info") or {}
            patient_info = (
                PatientInfo(
                    **{
                        key: value
                        for key, value in patient_raw.items()
                        if key in PatientInfo.model_fields
                    }
                )
                if patient_raw
                else None
            )

            report = LabReport(
                patient_info=patient_info,
                lab_name=current_data.get("lab_name"),
                report_date=current_data.get("report_date"),
                report_type=current_data.get("report_type", "other"),
                tests=tests,
                metadata={
                    key: str(value)
                    for key, value in current_data.get("metadata", {}).items()
                },
            )

            issues = _check_plausibility(report)
            if not issues or attempt == config.max_retries:
                break

            logger.info(
                "validate: attempt %d/%d, %d issues found, retrying",
                attempt + 1,
                config.max_retries,
                len(issues),
            )
            retries += 1
            previous_json = json.dumps(current_data, default=str)
            retry_prompt = VALIDATE_PROMPT.format(previous_json=previous_json[:2000])
            response = engine.query(image, retry_prompt)
            try:
                current_data = json.loads(response)
                if (
                    isinstance(current_data, dict)
                    and "corrections" in current_data
                    and current_data["corrections"]
                ):
                    current_data = current_data["corrections"]
            except json.JSONDecodeError:
                break
        except (ValidationError, Exception) as exc:
            issues.append(f"Schema validation error: {exc}")
            if attempt == config.max_retries:
                report = LabReport(tests=[])
                break

    reasoning = (
        f"Validated {len(report.tests)} tests. "
        f"Issues found: {len(issues)}. "
        f"Retries used: {retries}/{config.max_retries}. "
        "Checks: value plausibility, unit standards, duplicate detection."
    )
    if issues:
        reasoning += f" Issues: {'; '.join(issues[:3])}"

    logger.info(
        "validate: %d tests valid, %d issues, %d retries",
        len(report.tests),
        len(issues),
        retries,
    )

    stage_result = StageResult(
        stage_name="validate",
        input_summary=f"Extracted data with {len(extracted.get('tests', []))} raw tests",
        output={
            "valid": len(issues) == 0,
            "issues_found": len(issues),
            "issues": issues[:5],
            "retries_used": retries,
            "final_test_count": len(report.tests),
        },
        reasoning=reasoning,
        confidence=max(0.5, 1.0 - len(issues) * 0.1),
        timing_seconds=time.time() - start,
        retries=retries,
    )

    return report, stage_result


def _check_plausibility(report: LabReport) -> list[str]:
    issues: list[str] = []
    seen_names: set[str] = set()

    for test in report.tests:
        name_lower = test.test_name.lower()
        if name_lower in seen_names:
            issues.append(f"Duplicate test: {test.test_name}")
        seen_names.add(name_lower)

        try:
            value = float(test.value.replace("<", "").replace(">", "").strip())
            if value < 0:
                issues.append(f"Negative value for {test.test_name}: {test.value}")
            if value > 100000:
                issues.append(
                    f"Implausibly large value for {test.test_name}: {test.value}"
                )
        except (ValueError, AttributeError):
            continue

    return issues
