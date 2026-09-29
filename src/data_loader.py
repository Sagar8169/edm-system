"""
data_loader.py — Download and load the OULAD dataset.

Downloads the 7 OULAD CSV files from the official source (analyse.kmi.open.ac.uk)
and loads them into pandas DataFrames.
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

# OULAD download URL (official Figshare mirror)
OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
OULAD_FIGSHARE_URL = "https://ndownloader.figshare.com/articles/5081998/versions/1"

# Expected CSV files in the dataset
EXPECTED_FILES = [
    "studentInfo.csv",
    "studentVle.csv",
    "studentAssessment.csv",
    "studentRegistration.csv",
    "courses.csv",
    "assessments.csv",
    "vle.csv",
]


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_raw_data_dir() -> Path:
    """Return the raw data directory, creating it if needed."""
    raw_dir = get_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def get_processed_data_dir() -> Path:
    """Return the processed data directory, creating it if needed."""
    proc_dir = get_project_root() / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    return proc_dir


def download_oulad(force: bool = False) -> Path:
    """
    Download the OULAD dataset zip file and extract CSVs.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        Path to the raw data directory containing the CSVs.
    """
    raw_dir = get_raw_data_dir()
    zip_path = raw_dir / "oulad.zip"

    # Check if already downloaded
    existing_csvs = [f for f in EXPECTED_FILES if (raw_dir / f).exists()]
    if len(existing_csvs) == len(EXPECTED_FILES) and not force:
        print(f"✓ All {len(EXPECTED_FILES)} OULAD CSV files already present in {raw_dir}")
        return raw_dir

    # Download
    print("⬇ Downloading OULAD dataset...")
    urls_to_try = [OULAD_URL, OULAD_FIGSHARE_URL]

    downloaded = False
    for url in urls_to_try:
        try:
            print(f"  Trying: {url}")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        pct = (downloaded_size / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded_size // 1024}KB / {total_size // 1024}KB)", end="")

            print(f"\n  ✓ Downloaded {downloaded_size // 1024}KB")
            downloaded = True
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    if not downloaded:
        raise RuntimeError(
            "Could not download OULAD dataset. Please manually download from:\n"
            "  https://analyse.kmi.open.ac.uk/open_dataset\n"
            f"  and place the CSV files in: {raw_dir}"
        )

    # Extract
    print("📦 Extracting CSV files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if basename.endswith(".csv"):
                # Extract to raw_dir with flat structure
                target = raw_dir / basename
                with zf.open(member) as source, open(target, "wb") as dest:
                    dest.write(source.read())
                print(f"  ✓ Extracted: {basename}")

    # Clean up zip
    if zip_path.exists():
        zip_path.unlink()
        print("  ✓ Cleaned up zip file")

    return raw_dir


def load_oulad(download_if_missing: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load all OULAD CSV files into a dictionary of DataFrames.

    Args:
        download_if_missing: If True, download the dataset if not found.

    Returns:
        Dictionary mapping table names to DataFrames:
        {
            'studentInfo': DataFrame,
            'studentVle': DataFrame,
            'studentAssessment': DataFrame,
            'studentRegistration': DataFrame,
            'courses': DataFrame,
            'assessments': DataFrame,
            'vle': DataFrame
        }
    """
    raw_dir = get_raw_data_dir()

    # Check if we need to download
    missing = [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]
    if missing and download_if_missing:
        print(f"Missing {len(missing)} files. Downloading dataset...")
        download_oulad()
    elif missing:
        raise FileNotFoundError(
            f"Missing OULAD files: {missing}\n"
            f"Run download_oulad() first or set download_if_missing=True"
        )

    # Load all CSVs
    data = {}
    print("\n📊 Loading OULAD tables:")
    for filename in EXPECTED_FILES:
        table_name = filename.replace(".csv", "")
        filepath = raw_dir / filename
        df = pd.read_csv(filepath)
        data[table_name] = df
        print(f"  {table_name:25s} → {df.shape[0]:>7,} rows × {df.shape[1]:>2} cols")

    return data


def inspect_data(data: dict[str, pd.DataFrame]) -> None:
    """Print detailed inspection of each table: dtypes, nulls, sample rows."""
    for name, df in data.items():
        print(f"\n{'='*60}")
        print(f"TABLE: {name}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"\nColumn types:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {col:30s}  {str(df[col].dtype):10s}  nulls: {null_count:>6} ({null_pct:.1f}%)")
        print(f"\nSample (first 3 rows):")
        print(df.head(3).to_string())


if __name__ == "__main__":
    # Quick test: download and inspect
    data = load_oulad()
    inspect_data(data)
