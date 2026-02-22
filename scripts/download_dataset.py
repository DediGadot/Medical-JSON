#!/usr/bin/env python3
"""Download Mendeley Clinical Lab Reports dataset."""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


# Mendeley dataset URL
DATASET_URL = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/bygfmk4rx9-2.zip"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def download_with_progress(url: str, output_path: Path, force: bool = False) -> bool:
    """Download file with progress reporting and retry logic."""
    if output_path.exists() and not force:
        print(f"Files already exist at {output_path}. Use --force to overwrite.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading dataset (attempt {attempt}/{MAX_RETRIES})...")

            def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
                """Progress hook for urlretrieve."""
                if totalsize > 0:
                    downloaded = blocknum * blocksize
                    percent = min(100, int(100 * downloaded / totalsize))
                    if percent % 10 == 0 or percent == 100:
                        print(f"Downloading... {percent}%")

            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
            print(f"Download complete: {output_path}")
            return True

        except urllib.error.URLError as e:
            print(f"Download failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("All download attempts failed.")
                return False

    return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file to directory."""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction complete.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        return False
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        return False


def keep_first_n_files(directory: Path, n: int) -> None:
    """Keep only first n files in directory, remove rest."""
    files = sorted([f for f in directory.rglob("*") if f.is_file()])
    for f in files[n:]:
        f.unlink()
        print(f"Removed {f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Mendeley Clinical Lab Reports dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save dataset (default: data/raw)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download only first 5 files for quick testing",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = output_dir / "dataset.zip"

    # Download dataset
    if not download_with_progress(DATASET_URL, zip_path, args.force):
        sys.exit(1)

    # Extract dataset
    if not extract_zip(zip_path, output_dir):
        sys.exit(1)

    # Keep only first 5 files if --sample
    if args.sample:
        print("Keeping only first 5 files (--sample mode)...")
        keep_first_n_files(output_dir, 5)

    # Count extracted files
    extracted_files = [
        f for f in output_dir.rglob("*") if f.is_file() and f.name != "dataset.zip"
    ]
    print(f"\nDownloaded {len(extracted_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
