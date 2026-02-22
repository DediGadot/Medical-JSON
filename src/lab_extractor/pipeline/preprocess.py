from __future__ import annotations

import logging
import time

from PIL import Image, ImageEnhance, ImageOps

from lab_extractor.schemas.pipeline import StageResult

logger = logging.getLogger(__name__)

CONTRAST_BY_TYPE = {
    "cbc": 1.4,
    "metabolic": 1.4,
    "lipid": 1.3,
    "thyroid": 1.4,
    "urinalysis": 1.5,
    "liver": 1.4,
    "other": 1.4,
}


def preprocess_report(
    image: Image.Image, report_type: str
) -> tuple[Image.Image, StageResult]:
    start = time.time()

    contrast_factor = CONTRAST_BY_TYPE.get(report_type, 1.4)
    target_size = 896

    img = ImageOps.exif_transpose(image)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
    canvas.paste(img, offset)
    img = canvas

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    preprocessing_applied = [
        f"resize_{target_size}",
        f"contrast_{contrast_factor}",
        "exif_orient",
        "rgb_convert",
    ]

    reasoning = (
        f"Applied {report_type}-specific preprocessing: "
        f"resized to {target_size}x{target_size}, "
        f"contrast enhanced by {contrast_factor}x (type-specific), "
        "EXIF orientation corrected, converted to RGB."
    )

    logger.info(
        "preprocess: %s preprocessing applied for %s report",
        preprocessing_applied,
        report_type,
    )

    stage_result = StageResult(
        stage_name="preprocess",
        input_summary=f"Raw image ({image.size[0]}x{image.size[1]} px), report_type={report_type}",
        output={
            "preprocessing_applied": preprocessing_applied,
            "image_quality": "enhanced",
            "output_size": f"{target_size}x{target_size}",
            "contrast_factor": contrast_factor,
        },
        reasoning=reasoning,
        confidence=0.95,
        timing_seconds=time.time() - start,
    )

    return img, stage_result
