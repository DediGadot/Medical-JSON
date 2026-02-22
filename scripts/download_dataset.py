#!/usr/bin/env python3
"""Download Mendeley Clinical Lab Reports dataset.

Dataset: Clinical Laboratory Test Reports
DOI: 10.17632/bygfmk4rx9.2
License: CC BY 4.0

Download URL discovered from Mendeley Data SPA bundle.js:
  /public-api/zip/{dataset_id}/download/{version}
which redirects to a time-limited presigned S3 URL.
"""

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


# Mendeley public API download endpoint (discovered from bundle.js analysis)
DATASET_ID = "bygfmk4rx9"
DATASET_VERSION = "2"
DOWNLOAD_URL = (
    f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{DATASET_VERSION}"
)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def download_with_curl(url: str, output_path: Path, force: bool = False) -> bool:
    """Download file using curl (handles redirects to presigned S3 URLs correctly)."""
    if output_path.exists() and not force:
        print(f"File already exists at {output_path}. Use --force to overwrite.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl") is None:
        print("Error: curl is required but not found in PATH.")
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Downloading dataset (attempt {attempt}/{MAX_RETRIES})...")
        print(f"  URL: {url}")

        result = subprocess.run(
            [
                "curl",
                "--location",  # follow redirects (critical for S3 presigned URLs)
                "--max-redirs",
                "10",
                "--progress-bar",
                "--output",
                str(output_path),
                url,
            ],
            check=False,
        )

        if (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 1024
        ):
            print(
                f"\nDownload complete: {output_path} ({output_path.stat().st_size // 1024 // 1024} MB)"
            )
            return True

        print(
            f"Download failed (attempt {attempt}/{MAX_RETRIES}): curl exit code {result.returncode}"
        )
        if output_path.exists():
            output_path.unlink()  # remove partial file
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    print("All download attempts failed.")
    return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file to directory."""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            names = zip_ref.namelist()
            print(f"  {len(names)} files in archive")
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        return False
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        return False


def flatten_to_benchmark_input(
    raw_dir: Path, benchmark_dir: Path, force: bool = False
) -> int:
    """Flatten all extracted images into benchmark_input/ with sequential names."""
    if benchmark_dir.exists() and not force:
        existing = list(benchmark_dir.glob("*.jpg")) + list(benchmark_dir.glob("*.png"))
        if existing:
            print(
                f"benchmark_input/ already has {len(existing)} files. Use --force to overwrite."
            )
            return len(existing)

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files recursively
    all_images = sorted(
        raw_dir.rglob("*.jpg"),
        key=lambda p: p.name,
    )
    all_images += sorted(raw_dir.rglob("*.png"), key=lambda p: p.name)
    all_images += sorted(raw_dir.rglob("*.pdf"), key=lambda p: p.name)

    print(f"Flattening {len(all_images)} images to {benchmark_dir}...")
    for i, src in enumerate(all_images):
        safe_name = src.name.replace(" ", "_")
        dst = benchmark_dir / f"{i + 1:03d}_{safe_name}"
        import shutil as _shutil

        _shutil.copy2(src, dst)

    count = len(list(benchmark_dir.iterdir()))
    print(f"Done: {count} files in {benchmark_dir}")
    return count


def keep_first_n_files(directory: Path, n: int) -> None:
    """Keep only first n files in directory, remove rest."""
    files = sorted([f for f in directory.rglob("*") if f.is_file()])
    for f in files[n:]:
        f.unlink()
        print(f"Removed {f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Mendeley Clinical Lab Reports dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full download + extraction + benchmark flattening
  uv run python scripts/download_dataset.py --output-dir data/raw --force

  # Download only first 5 files for quick testing
  uv run python scripts/download_dataset.py --output-dir data/raw --sample

Notes:
  - Uses curl to follow redirects to time-limited presigned S3 URLs
  - curl must be available in PATH
  - Dataset: 260 JPG lab report images, ~58MB
  - License: CC BY 4.0 (DOI: 10.17632/bygfmk4rx9.2)
""",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save dataset (default: data/raw)",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmark_input",
        help="Directory to flatten images for benchmarking (default: data/benchmark_input)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Keep only first 5 files for quick testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--skip-flatten",
        action="store_true",
        help="Skip flattening to benchmark_input/",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = output_dir / "dataset.zip"

    # Download dataset
    if not download_with_curl(DOWNLOAD_URL, zip_path, args.force):
        sys.exit(1)

    # Extract dataset
    if not extract_zip(zip_path, output_dir):
        sys.exit(1)

    # Keep only first 5 files if --sample
    if args.sample:
        print("Keeping only first 5 files (--sample mode)...")
        # Find the extracted folder
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        for d in extracted_dirs:
            keep_first_n_files(d, 5)

    # Flatten to benchmark_input/
    if not args.skip_flatten:
        # Find extracted source directory
        extracted_root = output_dir / f"{DATASET_ID}-{DATASET_VERSION}"
        if not extracted_root.exists():
            # Try to find any extracted directory
            candidates = [d for d in output_dir.iterdir() if d.is_dir()]
            if candidates:
                extracted_root = candidates[0]
                print(f"Using extracted directory: {extracted_root}")
            else:
                print("Warning: could not find extracted directory, skipping flatten.")
                extracted_root = None

        if extracted_root:
            count = flatten_to_benchmark_input(
                extracted_root,
                Path(args.benchmark_dir),
                args.force,
            )
            print(f"\nBenchmark input ready: {count} images in {args.benchmark_dir}/")

    # Summary
    extracted_files = [
        f
        for f in output_dir.rglob("*")
        if f.is_file() and f.suffix in {".jpg", ".png", ".pdf"}
    ]
    print(
        f"\nDownloaded and extracted {len(extracted_files)} image files to {output_dir}/"
    )
    print(f"Run the benchmark with:")
    print(
        f"  uv run python scripts/preflight_gpu.py --input-dir {args.benchmark_dir} --require-4060"
    )
    print(
        f"  uv run python scripts/run_benchmark.py --input-dir {args.benchmark_dir} --output-dir benchmark_runs/"
    )


if __name__ == "__main__":
    main()
