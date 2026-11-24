"""Fast procedural generator for synthetic dermoscopy benchmark fixtures.

Enables 100% offline, deterministic pipeline verification and CI testing without
relying on external network downloads of massive multi-gigabyte archives.
"""

import math
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import SKIN_CLASSES, CLASS_TO_IDX


def _create_synthetic_lesion_image(
    cls_name: str,
    rng: np.random.Generator,
    size: int = 224,
) -> np.ndarray:
    """Procedurally renders a synthetic dermoscopic skin lesion image."""
    # 1. Base skin background (Fitzpatrick type II-IV skin tone + texture noise)
    skin_base = rng.integers(195, 235, size=(3,), dtype=np.uint8)
    img = np.ones((size, size, 3), dtype=np.uint8) * skin_base
    noise = rng.normal(0, 7, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Central lesion pigmentation
    center_x = size // 2 + int(rng.integers(-15, 16))
    center_y = size // 2 + int(rng.integers(-15, 16))
    radius_x = float(rng.integers(35, 75))
    radius_y = float(radius_x * rng.uniform(0.7, 1.3))
    angle = float(rng.uniform(0, math.pi))

    # Diagnosis-dependent coloring
    if cls_name == "vasc":
        lesion_color = np.array([160, 40, 50], dtype=np.float32)  # Reddish vascular
    elif cls_name == "mel":
        lesion_color = np.array([50, 35, 30], dtype=np.float32)   # Dark variegated melanoma
    elif cls_name == "bcc":
        lesion_color = np.array([170, 110, 110], dtype=np.float32) # Pearly pink
    elif cls_name == "akiec":
        lesion_color = np.array([180, 130, 90], dtype=np.float32)  # Keratotic scaly
    elif cls_name == "bkl":
        lesion_color = np.array([120, 90, 60], dtype=np.float32)   # Keratosis brown
    elif cls_name == "df":
        lesion_color = np.array([130, 80, 70], dtype=np.float32)   # Dermal fibrohistiocytic
    else:  # nv
        lesion_color = np.array([110, 75, 55], dtype=np.float32)   # Regular melanocytic nevus brown

    y_indices, x_indices = np.ogrid[:size, :size]
    dx = x_indices - center_x
    dy = y_indices - center_y

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rotated ellipse distance
    dist = (
        ((dx * cos_a + dy * sin_a) / radius_x) ** 2
        + ((-dx * sin_a + dy * cos_a) / radius_y) ** 2
    )

    # Sigmoid smooth falloff with border roughness
    roughness = rng.normal(0, 0.15, (size, size))
    mask = 1.0 / (1.0 + np.exp(6.0 * (dist - 1.0 + roughness)))
    mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]

    img = (img.astype(np.float32) * (1.0 - mask) + lesion_color * mask).astype(np.uint8)

    # 3. Add simulated dark circular dermatoscope lens vignette
    corner_dist = np.sqrt((x_indices - size / 2) ** 2 + (y_indices - size / 2) ** 2) / (size * 0.70)
    vignette = np.clip(1.0 - np.maximum(0.0, corner_dist - 0.75) * 3.5, 0.0, 1.0)[:, :, np.newaxis]
    img = (img.astype(np.float32) * vignette).astype(np.uint8)

    # 4. Add 1-3 simulated dark hair strands
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    num_hairs = int(rng.integers(1, 4))
    for _ in range(num_hairs):
        x1 = int(rng.integers(0, size))
        y1 = int(rng.integers(0, size))
        x2 = x1 + int(rng.integers(-80, 81))
        y2 = y1 + int(rng.integers(-80, 81))
        width = int(rng.integers(1, 3))
        draw.line([(x1, y1), (x2, y2)], fill=(30, 20, 15), width=width)

    return np.array(pil_img)


def generate_synthetic_dermoscopy_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 140,
    seed: int = 42,
    image_size: int = 224,
) -> pd.DataFrame:
    """Generates a complete synthetic dermoscopic dataset fixture offline.

    Features:
      - 7 canonical diagnostic classes matching HAM10000/ISIC proportions.
      - Realistic multi-image lesion grouping (approx 20% dual-image lesions).
      - Clinical metadata: dx, dx_type, age, sex, localization, lesion_id, image_id.
      - Executes in < 3 seconds.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Realistic class proportions reflecting HAM10000 distribution
    # akiec: ~3%, bcc: ~5%, bkl: ~11%, df: ~1.5%, mel: ~11%, nv: ~67%, vasc: ~1.5%
    class_probs = np.array([0.035, 0.055, 0.11, 0.02, 0.11, 0.65, 0.02], dtype=np.float64)
    class_probs = class_probs / class_probs.sum()

    # Ensure every class has at least 3 samples for 3-way train/val/test stratification
    base_counts = {c: 3 for c in SKIN_CLASSES}
    remaining = max(0, num_samples - sum(base_counts.values()))
    if remaining > 0:
        random_counts = rng.multinomial(remaining, class_probs)
        for c, cnt in zip(SKIN_CLASSES, random_counts):
            base_counts[c] += int(cnt)

    records = []
    lesion_counter = 1
    image_counter = 1

    dx_types = ["histo", "follow_up", "consensus", "confocal"]
    sexes = ["male", "female"]
    localizations = ["back", "lower extremity", "trunk", "face", "abdomen", "upper extremity", "chest"]

    for cls_name, count in base_counts.items():
        # Approx 80% single-image lesions, 20% multi-image lesions
        num_lesions = max(1, int(round(count * 0.8)))
        samples_per_lesion = [1] * num_lesions
        # Distribute remaining samples across lesions
        for i in range(count - num_lesions):
            samples_per_lesion[i % num_lesions] += 1

        for lesion_samples in samples_per_lesion:
            lesion_id = f"HAM_SYN_{lesion_counter:06d}"
            lesion_counter += 1

            lesion_dx_type = str(rng.choice(dx_types, p=[0.55, 0.35, 0.08, 0.02]))
            lesion_age = int(rng.integers(18, 85))
            lesion_sex = str(rng.choice(sexes))
            lesion_loc = str(rng.choice(localizations))

            for _ in range(lesion_samples):
                image_id = f"ISIC_SYN_{image_counter:07d}"
                image_counter += 1

                img_array = _create_synthetic_lesion_image(cls_name, rng, size=image_size)
                img = Image.fromarray(img_array)
                img_path = images_dir / f"{image_id}.jpg"
                img.save(img_path, format="JPEG", quality=90)

                records.append({
                    "lesion_id": lesion_id,
                    "image_id": image_id,
                    "dx": cls_name,
                    "dx_type": lesion_dx_type,
                    "age": lesion_age,
                    "sex": lesion_sex,
                    "localization": lesion_loc,
                    "label": CLASS_TO_IDX[cls_name],
                })

    df = pd.DataFrame(records)
    metadata_path = out_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    return df
