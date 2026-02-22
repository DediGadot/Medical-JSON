#!/usr/bin/env python3
"""Generate placeholder clinical lab report images for testing."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Sample lab values for variety
LAB_VALUES = [
    [
        ("Hemoglobin", "14.2", "g/dL", "[13.5-17.5]"),
        ("Hematocrit", "42.1", "%", "[40.0-54.0]"),
        ("RBC", "4.8", "M/µL", "[4.5-5.9]"),
        ("WBC", "7.2", "K/µL", "[4.5-11.0]"),
        ("Platelets", "245", "K/µL", "[150-400]"),
        ("Glucose", "95", "mg/dL", "[70-100]"),
        ("Creatinine", "0.9", "mg/dL", "[0.7-1.3]"),
    ],
    [
        ("Hemoglobin", "13.8", "g/dL", "[13.5-17.5]"),
        ("Hematocrit", "41.5", "%", "[40.0-54.0]"),
        ("RBC", "4.7", "M/µL", "[4.5-5.9]"),
        ("WBC", "6.9", "K/µL", "[4.5-11.0]"),
        ("Platelets", "250", "K/µL", "[150-400]"),
        ("Sodium", "138", "mEq/L", "[136-145]"),
        ("Potassium", "4.1", "mEq/L", "[3.5-5.0]"),
    ],
    [
        ("Hemoglobin", "15.1", "g/dL", "[13.5-17.5]"),
        ("Hematocrit", "45.2", "%", "[40.0-54.0]"),
        ("RBC", "5.1", "M/µL", "[4.5-5.9]"),
        ("WBC", "7.5", "K/µL", "[4.5-11.0]"),
        ("Platelets", "260", "K/µL", "[150-400]"),
        ("ALT", "28", "U/L", "[7-56]"),
        ("AST", "32", "U/L", "[10-40]"),
        ("Bilirubin", "0.8", "mg/dL", "[0.1-1.2]"),
    ],
    [
        ("Hemoglobin", "14.5", "g/dL", "[13.5-17.5]"),
        ("Hematocrit", "43.8", "%", "[40.0-54.0]"),
        ("RBC", "4.9", "M/µL", "[4.5-5.9]"),
        ("WBC", "7.1", "K/µL", "[4.5-11.0]"),
        ("Platelets", "240", "K/µL", "[150-400]"),
        ("Calcium", "9.2", "mg/dL", "[8.5-10.2]"),
        ("Phosphorus", "3.5", "mg/dL", "[2.5-4.5]"),
        ("Magnesium", "2.1", "mg/dL", "[1.7-2.2]"),
        ("Albumin", "4.0", "g/dL", "[3.5-5.0]"),
    ],
    [
        ("Hemoglobin", "13.9", "g/dL", "[13.5-17.5]"),
        ("Hematocrit", "41.8", "%", "[40.0-54.0]"),
        ("RBC", "4.8", "M/µL", "[4.5-5.9]"),
        ("WBC", "7.3", "K/µL", "[4.5-11.0]"),
        ("Platelets", "255", "K/µL", "[150-400]"),
        ("Cholesterol", "185", "mg/dL", "[<200]"),
        ("Triglycerides", "120", "mg/dL", "[<150]"),
        ("HDL", "52", "mg/dL", "[>40]"),
        ("LDL", "110", "mg/dL", "[<130]"),
    ],
]

PATIENT_INFO = [
    ("John Doe", "45", "M", "2024-01-15"),
    ("Jane Smith", "38", "F", "2024-01-16"),
    ("Robert Johnson", "52", "M", "2024-01-17"),
    ("Maria Garcia", "41", "F", "2024-01-18"),
    ("Ahmed Hassan", "48", "M", "2024-01-19"),
]


def create_sample_image(output_path: Path, sample_num: int) -> None:
    """Create a single placeholder lab report image."""
    # Image dimensions
    width, height = 896, 896

    # Create white background
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to default if unavailable
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
        header_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
        )
    except (OSError, IOError):
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw header bar (gray background)
    header_height = 50
    draw.rectangle([(0, 0), (width, header_height)], fill=(200, 200, 200))
    draw.text(
        (20, 15), "SAMPLE CLINICAL LABORATORY REPORT", fill="black", font=title_font
    )

    # Patient info section
    patient_idx = (sample_num - 1) % len(PATIENT_INFO)
    patient_name, age, gender, date = PATIENT_INFO[patient_idx]

    y_pos = 70
    patient_text = (
        f"Patient: {patient_name} | Age: {age} | Gender: {gender} | Date: {date}"
    )
    draw.text((20, y_pos), patient_text, fill="black", font=header_font)

    # Lab values section
    y_pos = 120
    draw.text((20, y_pos), "Laboratory Results:", fill="black", font=header_font)

    y_pos = 150
    lab_idx = (sample_num - 1) % len(LAB_VALUES)
    lab_tests = LAB_VALUES[lab_idx]

    # Draw table header
    draw.text((20, y_pos), "Test Name", fill="black", font=text_font)
    draw.text((250, y_pos), "Value", fill="black", font=text_font)
    draw.text((350, y_pos), "Unit", fill="black", font=text_font)
    draw.text((450, y_pos), "Reference Range", fill="black", font=text_font)

    # Draw separator line
    y_pos += 25
    draw.line([(20, y_pos), (850, y_pos)], fill=(150, 150, 150), width=1)

    # Draw lab test rows
    y_pos += 15
    for test_name, value, unit, ref_range in lab_tests:
        draw.text((20, y_pos), test_name, fill="black", font=text_font)
        draw.text((250, y_pos), value, fill="black", font=text_font)
        draw.text((350, y_pos), unit, fill="black", font=text_font)
        draw.text((450, y_pos), ref_range, fill="black", font=text_font)
        y_pos += 30

    # Footer
    footer_y = height - 40
    draw.line(
        [(20, footer_y - 10), (850, footer_y - 10)], fill=(150, 150, 150), width=1
    )
    footer_text = f"Lab: Cairo Medical Center | Report #{sample_num:03d}"
    draw.text((20, footer_y), footer_text, fill="black", font=small_font)

    # Save image
    img.save(output_path, "PNG")


def main() -> None:
    """Generate 5 sample images."""
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        output_path = samples_dir / f"sample_{i:03d}.png"
        create_sample_image(output_path, i)
        print(f"Created {output_path}")

    print(f"\nSuccessfully created 5 sample images in {samples_dir}")


if __name__ == "__main__":
    main()
