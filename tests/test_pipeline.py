"""Tests for the 5-stage pipeline."""

import pytest
from lab_extractor.pipeline.runner import run_pipeline
from lab_extractor.pipeline.flag import flag_abnormals
from lab_extractor.schemas import LabTest
from lab_extractor.schemas.config import PipelineConfig


def test_pipeline_dry_run_success(dry_run_config, sample_image_path):
    """Full dry-run pipeline produces successful PipelineResult."""
    result = run_pipeline(sample_image_path, dry_run_config)
    assert result.success is True
    assert result.error is None


def test_pipeline_has_five_stages(dry_run_config, sample_image_path):
    """Pipeline produces exactly 5 StageResults."""
    result = run_pipeline(sample_image_path, dry_run_config)
    assert len(result.stages) == 5
    stage_names = [s.stage_name for s in result.stages]
    assert stage_names == ["classify", "preprocess", "extract", "validate", "flag"]


def test_pipeline_stages_have_reasoning(dry_run_config, sample_image_path):
    """Each stage has non-empty reasoning and confidence > 0."""
    result = run_pipeline(sample_image_path, dry_run_config)
    for stage in result.stages:
        assert stage.reasoning, f"Stage {stage.stage_name} has empty reasoning"
        assert stage.confidence > 0, f"Stage {stage.stage_name} has zero confidence"
        assert stage.timing_seconds >= 0


def test_flag_stage_flags_abnormal_values():
    """Flag stage correctly identifies high and low values."""
    tests = [
        LabTest(test_name="Hemoglobin", value="18.0", unit="g/dL"),  # HIGH
        LabTest(test_name="Glucose", value="85", unit="mg/dL"),  # NORMAL
        LabTest(test_name="WBC", value="2.0", unit="x10^3/uL"),  # LOW
    ]
    result = flag_abnormals(tests)
    assert result.output["flagged_count"] >= 2
    assert result.output["total_tests"] == 3
