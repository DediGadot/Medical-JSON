from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from lab_extractor.schemas.lab_report import LabReport


class StageResult(BaseModel):
    stage_name: str
    input_summary: str
    output: dict[str, Any]
    reasoning: str
    confidence: float
    timing_seconds: float
    retries: int = 0


class PipelineResult(BaseModel):
    image_path: str
    stages: list[StageResult]
    final_report: LabReport
    total_time_seconds: float
    success: bool
    error: str | None = None
