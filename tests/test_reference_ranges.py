"""Tests for reference range lookup."""

import pytest
from lab_extractor.schemas.reference_ranges import lookup_reference, REFERENCE_RANGES


def test_lookup_hemoglobin():
    """Hemoglobin lookup returns correct range."""
    ref = lookup_reference("Hemoglobin")
    assert ref is not None
    assert ref.low == 13.5
    assert ref.high == 17.5


def test_lookup_case_insensitive():
    """Lookup is case-insensitive."""
    ref = lookup_reference("glucose")
    assert ref is not None
    ref2 = lookup_reference("GLUCOSE")
    assert ref2 is not None
    assert ref.low == ref2.low


def test_lookup_unknown_returns_none():
    """Unknown test name returns None (no exception)."""
    result = lookup_reference("NONEXISTENT_TEST_XYZ")
    assert result is None


def test_all_major_panels_covered():
    """All major lab panels have reference ranges."""
    # CBC
    assert lookup_reference("WBC") is not None
    assert lookup_reference("Platelets") is not None
    # Metabolic
    assert lookup_reference("Glucose") is not None
    assert lookup_reference("Creatinine") is not None
    # Lipid
    assert lookup_reference("LDL") is not None
    # Liver
    assert lookup_reference("ALT") is not None
    # Thyroid
    assert lookup_reference("TSH") is not None


def test_lookup_empty_string():
    """Empty string returns None gracefully."""
    result = lookup_reference("")
    assert result is None
