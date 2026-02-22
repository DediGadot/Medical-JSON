"""CORD receipt dataset adapter for lab-extractor pipeline.

Maps CORD-v2 ground truth (receipt line items) to the LabReport/LabTest schema,
enabling real precision/recall/F1 evaluation against ground truth JSON.

CORD GT schema (gt_parse):
  menu: list of {nm, cnt, price, sub?: {nm}}
  sub_total: {subtotal_price, tax_price, service_price, etc?}
  total: {total_price}

Mapping to LabTest:
  menu[].nm    → test_name
  menu[].price → value
  menu[].cnt   → unit  (count serves as the "unit" field)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# Receipt-specific extraction prompt — replaces EXTRACT_PROMPT for CORD images.
CORD_EXTRACT_PROMPT = (
    "Extract ALL line items from this receipt image. "
    "For each item extract: the item name, quantity/count, and price. "
    "Also extract subtotal, tax, service charge, and total amount if visible. "
    "Respond ONLY with valid JSON in this exact format: "
    '{"tests": [{"test_name": "<item name>", "value": "<price>", "unit": "<count or null>"}], '
    '"metadata": {"subtotal": "<value or null>", "tax": "<value or null>", '
    '"service": "<value or null>", "total": "<value or null>"}}. '
    "Include ALL visible line items. Do not include any text outside the JSON."
)


@dataclass
class CordItem:
    nm: str
    price: str
    cnt: str | None = None
    sub_nm: str | None = None  # sub-item / modifier (e.g. "WELL DONE")


@dataclass
class CordGroundTruth:
    image_id: int
    items: list[CordItem] = field(default_factory=list)
    subtotal: str | None = None
    tax: str | None = None
    service: str | None = None
    total: str | None = None


def parse_cord_ground_truth(annotation: dict[str, Any]) -> CordGroundTruth:
    """Parse a CORD annotation dict (as saved by download_cord.py) into CordGroundTruth.

    annotation = {"image_id": int, "split": str, "gt": {"gt_parse": {...}, ...}}
    """
    image_id: int = annotation.get("image_id", 0)
    gt_parse: dict[str, Any] = annotation.get("gt", {}).get("gt_parse", {})

    items: list[CordItem] = []
    for m in gt_parse.get("menu", []):
        nm = str(m.get("nm", "")).strip()
        price = str(m.get("price", "")).strip()
        cnt = str(m.get("cnt", "")).strip() or None
        sub = m.get("sub", {})
        sub_nm = (
            str(sub.get("nm", "")).strip() or None if isinstance(sub, dict) else None
        )
        if nm:
            items.append(CordItem(nm=nm, price=price, cnt=cnt, sub_nm=sub_nm))

    sub_total: dict[str, Any] = gt_parse.get("sub_total", {})
    total_block: dict[str, Any] = gt_parse.get("total", {})

    return CordGroundTruth(
        image_id=image_id,
        items=items,
        subtotal=_str_or_none(sub_total.get("subtotal_price")),
        tax=_str_or_none(sub_total.get("tax_price")),
        service=_str_or_none(sub_total.get("service_price")),
        total=_str_or_none(total_block.get("total_price")),
    )


def load_cord_annotation(ann_path: str) -> CordGroundTruth:
    """Load and parse a CORD annotation JSON file."""
    with open(ann_path) as f:
        data = json.load(f)
    return parse_cord_ground_truth(data)


def evaluate_cord(
    predicted_tests: list[dict[str, Any]],
    gt: CordGroundTruth,
) -> dict[str, Any]:
    """Evaluate extracted tests against CORD ground truth.

    predicted_tests: list of dicts with keys test_name, value, unit (from pipeline output)
    gt: parsed CORD ground truth

    Returns:
      item_name: {precision, recall, f1}
      price_accuracy: float (for matched items, within 5% tolerance)
      predicted_count: int
      ground_truth_count: int
      matched_count: int
      total_match: bool | None  (did we extract the correct total?)
    """
    pred_by_name = {_normalize_name(t.get("test_name", "")): t for t in predicted_tests}
    gt_by_name = {_normalize_name(item.nm): item for item in gt.items}

    pred_names = set(pred_by_name.keys()) - {""}
    gt_names = set(gt_by_name.keys()) - {""}

    tp = len(pred_names & gt_names)
    fp = len(pred_names - gt_names)
    fn = len(gt_names - pred_names)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Price accuracy for matched items
    price_matches = 0
    matched = pred_names & gt_names
    for name in matched:
        pred_val = pred_by_name[name].get("value", "")
        gt_val = gt_by_name[name].price
        if _prices_match(pred_val, gt_val):
            price_matches += 1
    price_accuracy = price_matches / len(matched) if matched else 0.0

    # Total match
    total_match: bool | None = None
    if gt.total:
        pred_total = next(
            (t.get("metadata", {}) for t in predicted_tests if isinstance(t, dict)), {}
        )
        # Also check metadata field from pipeline output directly
        total_match = False

    return {
        "item_name": {"precision": precision, "recall": recall, "f1": f1},
        "price_accuracy": price_accuracy,
        "predicted_count": len(pred_names),
        "ground_truth_count": len(gt_names),
        "matched_count": tp,
        "total_match": total_match,
    }


def evaluate_cord_from_pipeline_output(
    pipeline_output: dict[str, Any],
    gt: CordGroundTruth,
) -> dict[str, Any]:
    """Evaluate pipeline output dict against CORD ground truth.

    pipeline_output: the extract stage output dict (has "tests" key)
    """
    tests = pipeline_output.get("tests", [])
    result = evaluate_cord(tests, gt)

    # Check total from metadata
    metadata = pipeline_output.get("metadata", {}) or {}
    if gt.total and metadata.get("total"):
        result["total_match"] = _prices_match(metadata["total"], gt.total)

    return result


# ── internal helpers ──────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _parse_price(price_str: str) -> float | None:
    """Parse price string to float, handling commas and currency symbols."""
    if not price_str:
        return None
    cleaned = re.sub(r"[^\d.]", "", price_str.replace(",", "."))
    # Handle multiple dots — keep last
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _prices_match(pred: str, gt: str, tolerance: float = 0.05) -> bool:
    """Return True if prices are within tolerance (default 5%)."""
    pv = _parse_price(pred)
    gv = _parse_price(gt)
    if pv is None or gv is None:
        return pred.strip() == gt.strip()
    if gv == 0:
        return pv == 0
    return abs(pv - gv) / abs(gv) <= tolerance


def _str_or_none(val: object) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None
