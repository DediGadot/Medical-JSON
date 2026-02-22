from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel


class ReferenceRange(BaseModel):
    low: float | None = None
    high: float | None = None
    text: str | None = None  # For non-numeric ranges like "Negative"


class LabTest(BaseModel):
    test_name: str  # e.g., "Hemoglobin", "Glucose"
    value: str  # String to handle "<5", ">100", "Positive"
    unit: str | None = None  # e.g., "g/dL", "mg/dL"
    reference_range: ReferenceRange | None = None
    flag: Literal["normal", "high", "low", "critical_high", "critical_low"] | None = (
        None
    )


class PatientInfo(BaseModel):
    name: str | None = None
    age: str | None = None
    gender: str | None = None
    patient_id: str | None = None


class LabReport(BaseModel):
    patient_info: PatientInfo | None = None
    lab_name: str | None = None
    report_date: str | None = None
    report_type: Literal[
        "cbc", "metabolic", "lipid", "thyroid", "urinalysis", "liver", "other"
    ] = "other"
    tests: list[LabTest]
    metadata: dict[str, str] = {}
