"""Shared pytest fixtures for lab_extractor tests."""

import pytest
from pathlib import Path
from PIL import Image
from lab_extractor.schemas.config import PipelineConfig
from lab_extractor.engine.medgemma import MedGemmaEngine


@pytest.fixture
def sample_image_path() -> str:
    """Path to sample_001.png (896x896 placeholder lab report)."""
    path = Path("data/samples/sample_001.png")
    assert path.exists(), f"Sample image not found: {path}"
    return str(path)


@pytest.fixture
def dry_run_config() -> PipelineConfig:
    """PipelineConfig with dry_run=True (no GPU needed)."""
    return PipelineConfig(dry_run=True)


@pytest.fixture
def dry_run_engine(dry_run_config) -> MedGemmaEngine:
    """MedGemmaEngine in dry-run mode."""
    return MedGemmaEngine(dry_run_config)


@pytest.fixture
def white_image() -> Image.Image:
    """896x896 white RGB image for testing."""
    return Image.new("RGB", (896, 896), "white")
