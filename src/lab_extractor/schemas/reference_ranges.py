from lab_extractor.schemas.lab_report import ReferenceRange


REFERENCE_RANGES = {
    # CBC - Complete Blood Count
    "hemoglobin": ReferenceRange(low=13.5, high=17.5),  # Male default
    "hemoglobin m": ReferenceRange(low=13.5, high=17.5),
    "hemoglobin f": ReferenceRange(low=12.0, high=16.0),
    "wbc": ReferenceRange(low=4.5, high=11.0),
    "white blood cells": ReferenceRange(low=4.5, high=11.0),
    "platelets": ReferenceRange(low=150.0, high=400.0),
    "rbc": ReferenceRange(low=4.7, high=6.1),  # Male default
    "rbc m": ReferenceRange(low=4.7, high=6.1),
    "rbc f": ReferenceRange(low=4.2, high=5.4),
    "hematocrit": ReferenceRange(low=38.3, high=48.6),  # Male default
    "hematocrit m": ReferenceRange(low=38.3, high=48.6),
    "hematocrit f": ReferenceRange(low=35.5, high=44.9),
    # Metabolic Panel
    "glucose": ReferenceRange(low=70.0, high=100.0),
    "bun": ReferenceRange(low=7.0, high=20.0),
    "creatinine": ReferenceRange(low=0.7, high=1.3),
    "sodium": ReferenceRange(low=136.0, high=145.0),
    "potassium": ReferenceRange(low=3.5, high=5.0),
    "calcium": ReferenceRange(low=8.5, high=10.5),
    "co2": ReferenceRange(low=23.0, high=29.0),
    # Lipid Panel
    "total cholesterol": ReferenceRange(low=None, high=200.0),
    "ldl": ReferenceRange(low=None, high=100.0),
    "hdl": ReferenceRange(low=40.0, high=None),
    "triglycerides": ReferenceRange(low=None, high=150.0),
    # Liver Function Tests
    "alt": ReferenceRange(low=7.0, high=56.0),
    "ast": ReferenceRange(low=10.0, high=40.0),
    "alp": ReferenceRange(low=44.0, high=147.0),
    "bilirubin": ReferenceRange(low=0.1, high=1.2),
    "bilirubin total": ReferenceRange(low=0.1, high=1.2),
    # Thyroid Function Tests
    "tsh": ReferenceRange(low=0.4, high=4.0),
    "free t4": ReferenceRange(low=0.8, high=1.8),
    "free t3": ReferenceRange(low=2.3, high=4.2),
}


def lookup_reference(test_name: str) -> ReferenceRange | None:
    """
    Look up reference range for a test name.

    Normalizes the test name (lowercase, strip whitespace) and tries:
    1. Exact match
    2. Partial/substring match

    Returns None if no match found.
    """
    if not test_name:
        return None

    normalized = test_name.lower().strip()

    # Try exact match first
    if normalized in REFERENCE_RANGES:
        return REFERENCE_RANGES[normalized]

    # Try partial/substring match
    for key, value in REFERENCE_RANGES.items():
        if key in normalized or normalized in key:
            return value

    return None
