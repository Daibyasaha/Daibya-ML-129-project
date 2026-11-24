"""Dataset acquisition, checksum validation, and archive extraction utilities."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Union
import urllib.request
import zipfile
import pandas as pd


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Calculates SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_checksum(file_path: Union[str, Path], expected_checksum: str) -> bool:
    """Verifies that a file matches the expected SHA256 checksum."""
    actual = compute_sha256(file_path)
    return actual.lower() == expected_checksum.lower()


def download_file(
    url: str,
    destination: Union[str, Path],
    expected_checksum: Optional[str] = None,
) -> Path:
    """Downloads a file from a URL with optional SHA256 checksum validation."""
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and expected_checksum:
        if verify_file_checksum(dest_path, expected_checksum):
            return dest_path

    # Download with standard urllib
    urllib.request.urlretrieve(url, str(dest_path))

    if expected_checksum and not verify_file_checksum(dest_path, expected_checksum):
        dest_path.unlink()
        raise ValueError(f"Checksum mismatch for downloaded file from {url}")

    return dest_path


def extract_zip(
    zip_path: Union[str, Path],
    extract_to: Union[str, Path],
) -> Path:
    """Extracts a ZIP archive to a destination directory."""
    dest_dir = Path(extract_to)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def prepare_raw_ham10000(
    source_dir: Union[str, Path],
    output_images_dir: Union[str, Path],
    output_metadata_path: Union[str, Path],
) -> pd.DataFrame:
    """Aggregates HAM10000 image parts and standardizes metadata."""
    src = Path(source_dir)
    out_img = Path(output_images_dir)
    out_img.mkdir(parents=True, exist_ok=True)

    # Locate metadata CSV
    meta_candidates = list(src.glob("*.csv"))
    if not meta_candidates:
        raise FileNotFoundError(f"No metadata CSV found in {src}")

    meta_file = meta_candidates[0]
    df = pd.read_csv(meta_file)

    # Copy / symlink images if in different folders (part_1, part_2)
    for img_file in src.rglob("*.jpg"):
        target_path = out_img / img_file.name
        if not target_path.exists():
            # Link or copy
            try:
                os.link(str(img_file), str(target_path))
            except (OSError, AttributeError):
                import shutil
                shutil.copy2(str(img_file), str(target_path))

    df.to_csv(output_metadata_path, index=False)
    return df
