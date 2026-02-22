"""Tests for the CLI entry point."""

import json
import subprocess
import sys
import pytest


def run_cli(*args) -> subprocess.CompletedProcess:
    """Run the CLI with given args and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "lab_extractor"] + list(args),
        capture_output=True,
        text=True,
    )


def test_cli_help_exits_zero():
    """--help returns exit code 0."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--dry-run" in result.stdout


def test_cli_dry_run_produces_valid_json():
    """--input --dry-run produces valid JSON with success=True."""
    result = run_cli("--input", "data/samples/sample_001.png", "--dry-run")
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["success"] is True
    assert len(data["stages"]) == 5


def test_cli_invalid_args_exits_nonzero():
    """Missing required args returns non-zero exit code."""
    result = run_cli("--dry-run")  # Missing --input or --batch
    assert result.returncode != 0
