"""Tests for image preprocessing utilities."""

import pytest
import tempfile
import os
from PIL import Image
from lab_extractor.preprocessing import preprocess_image, load_image


def test_preprocess_returns_896x896(sample_image_path):
    """preprocess_image returns 896x896 RGB image."""
    result = preprocess_image(sample_image_path)
    assert result.size == (896, 896)
    assert result.mode == "RGB"


def test_preprocess_non_square_image():
    """preprocess_image handles non-square images (pads to 896x896)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (400, 600), "lightblue")
        img.save(f.name)
        result = preprocess_image(f.name)
        os.unlink(f.name)
    assert result.size == (896, 896)
    assert result.mode == "RGB"


def test_load_image_raises_for_missing_file():
    """load_image raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.png")


def test_load_image_valid_png(sample_image_path):
    """load_image returns RGB PIL Image for valid PNG."""
    img = load_image(sample_image_path)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
