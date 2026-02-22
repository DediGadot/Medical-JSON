"""Prompt templates for MedGemma lab report extraction pipeline."""

CLASSIFY_PROMPT = (
    "Analyze this lab report image. What type of lab report is this? "
    "Respond ONLY with valid JSON in this exact format: "
    '{"report_type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}. '
    "Types: cbc, metabolic, lipid, thyroid, urinalysis, liver, other. "
    "Do not include any text outside the JSON."
)

EXTRACT_PROMPT = (
    "Extract ALL lab test results from this lab report image. "
    "For each test provide: test_name, value (as string), unit, and reference_range (with low and high as numbers if shown). "
    "Also extract patient_info (name, age, gender, patient_id), lab_name, and report_date if visible. "
    "Respond ONLY with valid JSON matching this structure: "
    '{"patient_info": {"name": null, "age": null, "gender": null, "patient_id": null}, '
    '"lab_name": null, "report_date": null, "report_type": "other", '
    '"tests": [{"test_name": "...", "value": "...", "unit": "...", "reference_range": {"low": null, "high": null}}]}. '
    "Do not include any text outside the JSON. Include ALL visible test results."
)

VALIDATE_PROMPT = (
    "Review this extracted lab data for accuracy and completeness. "
    "Check: (1) all values are numerically plausible for their test type, "
    "(2) units are standard medical units, "
    "(3) no duplicate test names, "
    "(4) reference ranges are clinically reasonable. "
    "Previous extraction: {previous_json}. "
    "If corrections are needed, respond with the corrected full JSON. "
    "If data is valid, respond with: "
    '{"valid": true, "issues": [], "corrections": null, "confidence": <0.0-1.0>}. '
    "Respond ONLY with valid JSON."
)
