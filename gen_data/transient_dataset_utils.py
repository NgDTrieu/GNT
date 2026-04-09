from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'}


def stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


def make_rng(base_seed: int, *keys: object) -> np.random.RandomState:
    seed = int(base_seed)
    for key in keys:
        seed = (seed * 1000003 + stable_int_hash(str(key))) % (2**32 - 1)
    return np.random.RandomState(seed)


def create_random_mask(img_h: int, img_w: int, coverage_ratio: float = 0.1, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(0)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    total_pixels = img_h * img_w
    target_pixels = max(1, int(total_pixels * coverage_ratio))

    shape_type = rng.choice(['rect', 'circle'])

    if shape_type == 'rect':
        max_h = int(np.sqrt(target_pixels * 2))
        max_w = int(np.sqrt(target_pixels * 2))

        h_low = max(10, max_h // 3)
        h_high = min(img_h, max_h)
        w_low = max(10, max_w // 3)
        w_high = min(img_w, max_w)

        h_high = max(h_low + 1, h_high)
        w_high = max(w_low + 1, w_high)

        h = rng.randint(h_low, h_high)
        w = rng.randint(w_low, w_high)
        y = rng.randint(0, max(1, img_h - h + 1))
        x = rng.randint(0, max(1, img_w - w + 1))
        mask[y:y + h, x:x + w] = 1
    else:
        radius = int(np.sqrt(target_pixels / np.pi))
        radius = max(5, min(min(img_h, img_w) // 2, radius))

        cy_low = radius
        cy_high = max(radius + 1, img_h - radius + 1)
        cx_low = radius
        cx_high = max(radius + 1, img_w - radius + 1)

        cy = rng.randint(cy_low, cy_high)
        cx = rng.randint(cx_low, cx_high)
        cv2.circle(mask, (cx, cy), radius, 1, -1)

    return mask


def apply_transient_augmentation(
    img: np.ndarray,
    mask: np.ndarray,
    aug_type: str = 'random',
    return_mask: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.RandomState(0)

    if aug_type == 'random':
        aug_type = rng.choice(['noise', 'color', 'blur'])

    original_dtype = img.dtype
    if np.issubdtype(original_dtype, np.integer):
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

    augmented_img = img_float.copy()
    mask_bool = mask.astype(bool)

    if aug_type == 'noise':
        noise = rng.normal(0.5, 0.2, img_float.shape)
        noise = np.clip(noise, 0, 1)
        augmented_img[mask_bool] = noise[mask_bool]
    elif aug_type == 'color':
        random_color = rng.rand(3)
        augmented_img[mask_bool] = random_color
    elif aug_type == 'blur':
        blurred = cv2.GaussianBlur(img_float, (21, 21), 0)
        augmented_img[mask_bool] = blurred[mask_bool]
    else:
        raise ValueError(f'Unsupported aug_type: {aug_type}')

    augmented_img = np.clip(augmented_img, 0, 1)
    if np.issubdtype(original_dtype, np.integer):
        augmented_img_out = (augmented_img * 255.0).round().astype(original_dtype)
    else:
        augmented_img_out = augmented_img.astype(original_dtype)

    if return_mask:
        transient_mask = (1.0 - mask).astype(np.float32)  # 1=static, 0=transient
        return augmented_img_out, transient_mask
    return augmented_img_out


def is_image_file(path: Path) -> bool:
    return path.suffix in IMAGE_EXTS


def copy_tree(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {dst}')
    shutil.copytree(src, dst)


def process_image_dir(
    src_img_dir: Path,
    dst_img_dir: Path,
    base_seed: int,
    augment_prob: float,
    coverage_min: float,
    coverage_max: float,
    save_masks: bool = False,
    mask_dir_name: Optional[str] = None,
) -> dict:
    stats = {
        'processed_images': 0,
        'augmented_images': 0,
        'mask_dir': None,
    }

    mask_dir = None
    if save_masks:
        if mask_dir_name is None:
            mask_dir_name = f'{dst_img_dir.name}_masks'
        mask_dir = dst_img_dir.parent / mask_dir_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        stats['mask_dir'] = str(mask_dir)

    for src_path in sorted(src_img_dir.iterdir()):
        if not src_path.is_file() or not is_image_file(src_path):
            continue
        rel_key = str(src_path)
        rng = make_rng(base_seed, rel_key)
        stats['processed_images'] += 1

        should_augment = rng.rand() < augment_prob
        dst_path = dst_img_dir / src_path.name

        if not should_augment:
            shutil.copy2(src_path, dst_path)
            if mask_dir is not None:
                img = imageio.imread(src_path)
                h, w = img.shape[:2]
                static_mask = np.ones((h, w), dtype=np.uint8) * 255
                imageio.imwrite(mask_dir / src_path.name, static_mask)
            continue

        img = imageio.imread(src_path)
        coverage = rng.uniform(coverage_min, coverage_max)
        mask = create_random_mask(img.shape[0], img.shape[1], coverage_ratio=coverage, rng=rng)
        aug_type = rng.choice(['noise', 'color', 'blur'])
        aug_img, transient_mask = apply_transient_augmentation(
            img, mask, aug_type=aug_type, return_mask=True, rng=rng
        )
        imageio.imwrite(dst_path, aug_img)
        stats['augmented_images'] += 1

        if mask_dir is not None:
            # Save static/transient mask as 255=static, 0=transient for easy viewing.
            static_mask = (transient_mask * 255.0).round().astype(np.uint8)
            imageio.imwrite(mask_dir / src_path.name, static_mask)

    return stats


def write_metadata(dst_root: Path, metadata: dict) -> None:
    with open(dst_root / 'transient_generation_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
