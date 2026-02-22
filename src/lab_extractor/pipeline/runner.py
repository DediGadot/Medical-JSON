from __future__ import annotations

import logging
import time

from PIL import Image

from lab_extractor.engine.medgemma import MedGemmaEngine
from lab_extractor.pipeline.classify import classify_report
from lab_extractor.pipeline.extract import extract_lab_data
from lab_extractor.pipeline.flag import flag_abnormals
from lab_extractor.pipeline.preprocess import preprocess_report
from lab_extractor.pipeline.validate import validate_extraction
from lab_extractor.schemas.config import PipelineConfig
from lab_extractor.schemas.lab_report import LabReport
from lab_extractor.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def run_pipeline(image_path: str, config: PipelineConfig) -> PipelineResult:
    pipeline_start = time.time()
    stages = []

    try:
        image = Image.open(image_path).convert("RGB")
        logger.info(
            "pipeline: loaded image %s (%dx%d)",
            image_path,
            image.width,
            image.height,
        )

        engine = MedGemmaEngine(config)

        logger.info("pipeline: [1/5] classify")
        classify_result = classify_report(engine, image)
        stages.append(classify_result)
        report_type = classify_result.output["report_type"]

        logger.info("pipeline: [2/5] preprocess (type=%s)", report_type)
        processed_img, preprocess_result = preprocess_report(image, report_type)
        stages.append(preprocess_result)

        logger.info("pipeline: [3/5] extract")
        extract_result = extract_lab_data(engine, processed_img, report_type)
        stages.append(extract_result)
        extracted_data = {
            "tests": extract_result.output.get("tests", []),
            "patient_info": extract_result.output.get("patient_info", {}),
            "report_type": report_type,
        }

        logger.info("pipeline: [4/5] validate")
        final_report, validate_result = validate_extraction(
            engine,
            processed_img,
            extracted_data,
            config,
        )
        stages.append(validate_result)

        logger.info("pipeline: [5/5] flag")
        flag_result = flag_abnormals(final_report.tests)
        stages.append(flag_result)

        total_time = time.time() - pipeline_start
        logger.info(
            "pipeline: complete in %.2fs - %d tests, %d flagged",
            total_time,
            len(final_report.tests),
            flag_result.output["flagged_count"],
        )

        return PipelineResult(
            image_path=str(image_path),
            stages=stages,
            final_report=final_report,
            total_time_seconds=total_time,
            success=True,
        )
    except Exception as exc:
        logger.error("pipeline: failed - %s", exc)
        total_time = time.time() - pipeline_start
        return PipelineResult(
            image_path=str(image_path),
            stages=stages,
            final_report=LabReport(tests=[]),
            total_time_seconds=total_time,
            success=False,
            error=str(exc),
        )
