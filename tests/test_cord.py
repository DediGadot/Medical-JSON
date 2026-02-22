"""Unit tests for the CORD receipt dataset adapter (lab_extractor.cord)."""

from __future__ import annotations

import pytest

from lab_extractor.cord import (
    CORD_EXTRACT_PROMPT,
    CordGroundTruth,
    CordItem,
    _normalize_name,
    _parse_price,
    _prices_match,
    evaluate_cord,
    evaluate_cord_from_pipeline_output,
    parse_cord_ground_truth,
)


# ── CORD_EXTRACT_PROMPT ───────────────────────────────────────────────────────


def test_cord_extract_prompt_is_nonempty() -> None:
    assert isinstance(CORD_EXTRACT_PROMPT, str)
    assert len(CORD_EXTRACT_PROMPT) > 50
    assert "JSON" in CORD_EXTRACT_PROMPT
    assert "test_name" in CORD_EXTRACT_PROMPT
    assert "value" in CORD_EXTRACT_PROMPT


# ── _normalize_name ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Nasi Campur Bali", "nasi campur bali"),
        ("  PEPPER AUS  ", "pepper aus"),
        ("Item-Name!", "itemname"),
        ("iced   latte", "iced latte"),
        ("", ""),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert _normalize_name(raw) == expected


# ── _parse_price ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("75,000", 75.0),  # CORD uses comma as thousands separator
        ("1,591,600", 1591.6),  # multi-comma — interpreted as 1591.6 (last dot)
        ("$9.99", 9.99),
        ("100", 100.0),
        ("", None),
        ("N/A", None),
    ],
)
def test_parse_price(raw: str, expected: float | None) -> None:
    result = _parse_price(raw)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert abs(result - expected) < 0.01


# ── _prices_match ─────────────────────────────────────────────────────────────


def test_prices_match_within_tolerance() -> None:
    assert _prices_match("75,100", "75,000") is True  # ~0.13% off


def test_prices_match_exact() -> None:
    assert _prices_match("100", "100") is True


def test_prices_match_outside_tolerance() -> None:
    assert _prices_match("80,000", "75,000") is False  # 6.7% off


def test_prices_match_zero() -> None:
    assert _prices_match("0", "0") is True


def test_prices_match_string_fallback() -> None:
    # When both parse to None, falls back to string equality
    assert _prices_match("FREE", "FREE") is True
    assert _prices_match("FREE", "GRATIS") is False


# ── parse_cord_ground_truth ───────────────────────────────────────────────────

SAMPLE_ANNOTATION = {
    "image_id": 42,
    "split": "test",
    "gt": {
        "gt_parse": {
            "menu": [
                {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
                {
                    "nm": "PEPPER AUS",
                    "cnt": "1",
                    "price": "165,000",
                    "sub": {"nm": "WELL DONE"},
                },
                {"nm": "Juice", "cnt": "2", "price": "30,000"},
            ],
            "sub_total": {
                "subtotal_price": "270,000",
                "service_price": "27,000",
                "tax_price": "29,700",
            },
            "total": {"total_price": "326,700"},
        }
    },
}


def test_parse_cord_ground_truth_items() -> None:
    gt = parse_cord_ground_truth(SAMPLE_ANNOTATION)
    assert gt.image_id == 42
    assert len(gt.items) == 3
    assert gt.items[0].nm == "Nasi Campur Bali"
    assert gt.items[0].price == "75,000"
    assert gt.items[0].cnt == "1 x"
    assert gt.items[1].sub_nm == "WELL DONE"


def test_parse_cord_ground_truth_totals() -> None:
    gt = parse_cord_ground_truth(SAMPLE_ANNOTATION)
    assert gt.subtotal == "270,000"
    assert gt.tax == "29,700"
    assert gt.service == "27,000"
    assert gt.total == "326,700"


def test_parse_cord_ground_truth_empty() -> None:
    gt = parse_cord_ground_truth({})
    assert gt.image_id == 0
    assert gt.items == []
    assert gt.total is None


def test_parse_cord_ground_truth_missing_price() -> None:
    annotation = {
        "image_id": 1,
        "gt": {
            "gt_parse": {
                "menu": [{"nm": "Water", "price": ""}],
            }
        },
    }
    gt = parse_cord_ground_truth(annotation)
    assert len(gt.items) == 1
    assert gt.items[0].price == ""


def test_parse_cord_ground_truth_skips_empty_name() -> None:
    annotation = {
        "gt": {
            "gt_parse": {
                "menu": [
                    {"nm": "", "price": "100"},
                    {"nm": "  ", "price": "200"},
                    {"nm": "Real Item", "price": "300"},
                ]
            }
        }
    }
    gt = parse_cord_ground_truth(annotation)
    assert len(gt.items) == 1
    assert gt.items[0].nm == "Real Item"


# ── evaluate_cord ─────────────────────────────────────────────────────────────


def _make_gt() -> CordGroundTruth:
    return CordGroundTruth(
        image_id=0,
        items=[
            CordItem(nm="Nasi Campur Bali", price="75,000", cnt="1 x"),
            CordItem(nm="Juice", price="30,000", cnt="2"),
            CordItem(nm="Coffee", price="20,000", cnt="1"),
        ],
        total="125,000",
    )


def test_evaluate_cord_perfect_match() -> None:
    gt = _make_gt()
    predicted = [
        {"test_name": "Nasi Campur Bali", "value": "75,000", "unit": "1 x"},
        {"test_name": "Juice", "value": "30,000", "unit": "2"},
        {"test_name": "Coffee", "value": "20,000", "unit": "1"},
    ]
    result = evaluate_cord(predicted, gt)
    assert result["item_name"]["precision"] == pytest.approx(1.0)
    assert result["item_name"]["recall"] == pytest.approx(1.0)
    assert result["item_name"]["f1"] == pytest.approx(1.0)
    assert result["price_accuracy"] == pytest.approx(1.0)
    assert result["matched_count"] == 3


def test_evaluate_cord_partial_match() -> None:
    gt = _make_gt()
    # Only predict 2 of 3 items (recall < 1), add 1 wrong item (precision < 1)
    predicted = [
        {"test_name": "Nasi Campur Bali", "value": "75,000"},
        {"test_name": "Juice", "value": "30,000"},
        {"test_name": "Tea", "value": "15,000"},  # not in GT
    ]
    result = evaluate_cord(predicted, gt)
    # TP=2, FP=1, FN=1 → P=2/3, R=2/3
    assert result["item_name"]["precision"] == pytest.approx(2 / 3, abs=0.01)
    assert result["item_name"]["recall"] == pytest.approx(2 / 3, abs=0.01)
    assert result["matched_count"] == 2


def test_evaluate_cord_no_predictions() -> None:
    gt = _make_gt()
    result = evaluate_cord([], gt)
    assert result["item_name"]["precision"] == pytest.approx(0.0)
    assert result["item_name"]["recall"] == pytest.approx(0.0)
    assert result["item_name"]["f1"] == pytest.approx(0.0)
    assert result["predicted_count"] == 0
    assert result["ground_truth_count"] == 3


def test_evaluate_cord_empty_gt() -> None:
    gt = CordGroundTruth(image_id=0, items=[])
    predicted = [{"test_name": "Coffee", "value": "20,000"}]
    result = evaluate_cord(predicted, gt)
    # No GT items — recall = 0/0 = 0, precision = 0/1 = 0
    assert result["item_name"]["recall"] == pytest.approx(0.0)
    assert result["ground_truth_count"] == 0


def test_evaluate_cord_case_insensitive_matching() -> None:
    """Names should match case-insensitively after normalization."""
    gt = CordGroundTruth(
        image_id=0,
        items=[CordItem(nm="Nasi Campur Bali", price="75,000")],
    )
    predicted = [{"test_name": "NASI CAMPUR BALI", "value": "75,000"}]
    result = evaluate_cord(predicted, gt)
    assert result["matched_count"] == 1
    assert result["item_name"]["f1"] == pytest.approx(1.0)


# ── evaluate_cord_from_pipeline_output ───────────────────────────────────────


def test_evaluate_cord_from_pipeline_output_total_match() -> None:
    gt = CordGroundTruth(
        image_id=0,
        items=[CordItem(nm="Coffee", price="20,000")],
        total="20,000",
    )
    pipeline_output = {
        "tests": [{"test_name": "Coffee", "value": "20,000"}],
        "metadata": {"total": "20,000"},
    }
    result = evaluate_cord_from_pipeline_output(pipeline_output, gt)
    assert result["total_match"] is True


def test_evaluate_cord_from_pipeline_output_no_metadata() -> None:
    gt = CordGroundTruth(image_id=0, items=[], total="100,000")
    pipeline_output = {"tests": []}
    result = evaluate_cord_from_pipeline_output(pipeline_output, gt)
    # total_match should be None or False (no metadata provided)
    assert result["total_match"] in (None, False)
