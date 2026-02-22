"""Tests for MedGemma engine (dry-run mode only)."""

import json
import pytest
from PIL import Image
from lab_extractor.engine import MedGemmaEngine
from lab_extractor.engine.prompts import (
    CLASSIFY_PROMPT,
    EXTRACT_PROMPT,
    VALIDATE_PROMPT,
)
from lab_extractor.schemas.config import PipelineConfig


def test_engine_dry_run_instantiation(dry_run_config):
    """MedGemmaEngine can be instantiated in dry-run mode without GPU."""
    engine = MedGemmaEngine(dry_run_config)
    assert engine.pipe is None  # No model loaded in dry-run


def test_engine_classify_returns_json(dry_run_engine, white_image):
    """Dry-run classify returns parseable JSON with report_type."""
    result = dry_run_engine.query(white_image, "Classify this lab report type")
    data = json.loads(result)
    assert "report_type" in data
    assert data["report_type"] in {
        "cbc",
        "metabolic",
        "lipid",
        "thyroid",
        "urinalysis",
        "liver",
        "other",
    }


def test_engine_extract_returns_tests(dry_run_engine, white_image):
    """Dry-run extract returns JSON with tests array."""
    result = dry_run_engine.query(white_image, "Extract all lab test results as JSON")
    data = json.loads(result)
    tests = data.get("tests", [])
    assert len(tests) >= 5
    assert all("test_name" in t and "value" in t for t in tests)


def test_prompt_templates_non_empty():
    """All prompt templates are non-empty strings with JSON instruction."""
    assert len(CLASSIFY_PROMPT) > 50
    assert len(EXTRACT_PROMPT) > 50
    assert len(VALIDATE_PROMPT) > 50
    assert "JSON" in EXTRACT_PROMPT
    assert "{previous_json}" in VALIDATE_PROMPT
