from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str | Path, target_size: int = 896) -> Image.Image:
    """Load and preprocess an image for MedGemma inference.

    Steps:
    1. Load image from path
    2. Auto-orient using EXIF data (handles rotated mobile photos)
    3. Convert to RGB if grayscale or RGBA
    4. Resize to target_size x target_size maintaining aspect ratio (pad with white)
    5. Apply contrast enhancement (factor 1.4 for scanned docs)

    Returns: PIL Image, RGB mode, target_size x target_size pixels
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path)

    # Auto-orient using EXIF
    img = ImageOps.exif_transpose(img)

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize maintaining aspect ratio, pad with white
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    # Create white canvas and paste centered
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
    canvas.paste(img, offset)
    img = canvas

    # Contrast enhancement for scanned documents
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)

    return img


def convert_pdf_to_images(pdf_path: str | Path) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page).

    Uses pdf2image.convert_from_path.
    Handles multi-page PDFs (lab reports may be 1-3 pages).
    """
    from pdf2image import convert_from_path

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    images = convert_from_path(str(path))
    return [img.convert("RGB") for img in images]


def load_image(path: str | Path) -> Image.Image:
    """Smart loader: detects file type, handles PDF vs image.

    For PDFs: returns first page as PIL Image.
    For images: returns PIL Image directly.
    Raises FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = convert_pdf_to_images(path)
        if not pages:
            raise ValueError(f"PDF has no pages: {path}")
        return pages[0]
    else:
        img = Image.open(path)
        return img.convert("RGB")
