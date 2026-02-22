"""Tests for Pydantic schema models."""

import json
import pytest
from pydantic import ValidationError
from lab_extractor.schemas.lab_report import (
    LabTest,
    LabReport,
    ReferenceRange,
    PatientInfo,
)
from lab_extractor.schemas.pipeline import StageResult, PipelineResult
from lab_extractor.schemas.config import PipelineConfig


def test_labtest_minimal():
    """LabTest can be created with just test_name and value."""
    t = LabTest(test_name="Glucose", value="95")
    assert t.test_name == "Glucose"
    assert t.value == "95"
    assert t.unit is None
    assert t.flag is None


def test_labtest_full():
    """LabTest accepts all optional fields."""
    t = LabTest(
        test_name="Hemoglobin",
        value="14.2",
        unit="g/dL",
        reference_range=ReferenceRange(low=13.5, high=17.5),
        flag="normal",
    )
    assert t.flag == "normal"
    assert t.reference_range.low == 13.5


def test_labreport_with_tests():
    """LabReport can be created with a list of tests."""
    report = LabReport(
        tests=[
            LabTest(test_name="Glucose", value="95", unit="mg/dL"),
            LabTest(test_name="WBC", value="7.5", unit="x10^3/uL"),
        ]
    )
    assert len(report.tests) == 2
    assert report.report_type == "other"  # default


def test_labreport_json_schema():
    """LabReport.model_json_schema() returns valid schema dict."""
    schema = LabReport.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "tests" in schema["properties"]


def test_labtest_validation_error():
    """LabTest raises ValidationError when test_name is missing."""
    with pytest.raises(ValidationError):
        LabTest(value="95")  # Missing required test_name


def test_pipeline_result_serialization(dry_run_config):
    """PipelineResult can be serialized to JSON."""
    from lab_extractor.pipeline.runner import run_pipeline

    result = run_pipeline("data/samples/sample_001.png", dry_run_config)
    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert "success" in data
    assert "stages" in data
    assert "final_report" in data


def test_pipeline_config_defaults():
    """PipelineConfig has correct defaults."""
    config = PipelineConfig()
    assert config.model_name == "google/medgemma-1.5-4b-it"
    assert config.image_size == 896
    assert config.dry_run is False
    assert config.max_retries == 3
