from __future__ import annotations

import logging
import time
from typing import Literal

from lab_extractor.schemas.lab_report import LabTest
from lab_extractor.schemas.pipeline import StageResult
from lab_extractor.schemas.reference_ranges import lookup_reference

logger = logging.getLogger(__name__)

FlagType = Literal["normal", "high", "low", "critical_high", "critical_low"]


def flag_abnormals(tests: list[LabTest]) -> StageResult:
    start = time.time()

    flags: list[dict[str, object]] = []
    flagged_count = 0
    critical_count = 0
    reasoning_parts: list[str] = []

    for test in tests:
        ref = lookup_reference(test.test_name)
        flag: FlagType | None = None

        try:
            raw = test.value.replace("<", "").replace(">", "").strip()
            value = float(raw)
        except (ValueError, AttributeError):
            reasoning_parts.append(
                f"{test.test_name}: non-numeric value '{test.value}', skipped"
            )
            continue

        if ref is None:
            reasoning_parts.append(f"{test.test_name}: no reference range available")
            continue

        if ref.high is not None and value > ref.high * 2:
            flag = "critical_high"
            critical_count += 1
            flagged_count += 1
            reasoning_parts.append(
                f"{test.test_name}={value}: CRITICAL HIGH (ref high={ref.high}, 2x={ref.high * 2})"
            )
        elif ref.low is not None and ref.low > 0 and value < ref.low / 2:
            flag = "critical_low"
            critical_count += 1
            flagged_count += 1
            reasoning_parts.append(
                f"{test.test_name}={value}: CRITICAL LOW (ref low={ref.low}, half={ref.low / 2})"
            )
        elif ref.high is not None and value > ref.high:
            flag = "high"
            flagged_count += 1
            reasoning_parts.append(
                f"{test.test_name}={value}: HIGH (ref high={ref.high})"
            )
        elif ref.low is not None and value < ref.low:
            flag = "low"
            flagged_count += 1
            reasoning_parts.append(f"{test.test_name}={value}: LOW (ref low={ref.low})")
        else:
            flag = "normal"
            reasoning_parts.append(
                f"{test.test_name}={value}: normal (ref {ref.low}-{ref.high})"
            )

        test.flag = flag

        if flag != "normal":
            flags.append(
                {
                    "test_name": test.test_name,
                    "value": test.value,
                    "flag": flag,
                    "reference_low": ref.low,
                    "reference_high": ref.high,
                }
            )

    reasoning = (
        f"Flagged {flagged_count}/{len(tests)} tests as abnormal "
        f"({critical_count} critical). "
        f"Details: {'; '.join(reasoning_parts[:5])}"
    )

    logger.info(
        "flag: %d/%d tests flagged (%d critical)",
        flagged_count,
        len(tests),
        critical_count,
    )

    return StageResult(
        stage_name="flag",
        input_summary=f"{len(tests)} validated lab tests",
        output={
            "flagged_count": flagged_count,
            "critical_count": critical_count,
            "total_tests": len(tests),
            "flags": flags,
        },
        reasoning=reasoning,
        confidence=0.9,
        timing_seconds=time.time() - start,
    )
