from __future__ import annotations

import argparse
from pathlib import Path

from transient_dataset_utils import copy_tree, process_image_dir, write_metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Create an offline transient-augmented copy of real_iconic_noface.')
    parser.add_argument('--rootdir', type=str, default='.', help='Path to GNT repo root.')
    parser.add_argument('--input_name', type=str, default='real_iconic_noface', help='Source dataset directory inside <rootdir>/data.')
    parser.add_argument('--output_name', type=str, default='real_iconic_noface_new', help='Output dataset directory inside <rootdir>/data.')
    parser.add_argument('--image_dir_name', type=str, default='images_4', help='Image directory used by the current LLFF loader.')
    parser.add_argument('--seed', type=int, default=20260408, help='Base seed for deterministic augmentation.')
    parser.add_argument('--augment_prob', type=float, default=1.0, help='Probability of augmenting each image.')
    parser.add_argument('--coverage_min', type=float, default=0.10, help='Minimum transient coverage ratio.')
    parser.add_argument('--coverage_max', type=float, default=0.30, help='Maximum transient coverage ratio.')
    parser.add_argument('--save_masks', action='store_true', help='Also save oracle static/transient masks next to image dirs.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output dataset if it already exists.')
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    rootdir = Path(args.rootdir).resolve()
    src_root = rootdir / 'data' / args.input_name
    dst_root = rootdir / 'data' / args.output_name

    if not src_root.exists():
        raise FileNotFoundError(f'Input dataset not found: {src_root}')

    copy_tree(src_root, dst_root, overwrite=args.overwrite)

    all_stats = []
    scene_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    for scene_dir in scene_dirs:
        src_img_dir = scene_dir / args.image_dir_name
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_root / scene_dir.name / args.image_dir_name
        stats = process_image_dir(
            src_img_dir=src_img_dir,
            dst_img_dir=dst_img_dir,
            base_seed=args.seed,
            augment_prob=args.augment_prob,
            coverage_min=args.coverage_min,
            coverage_max=args.coverage_max,
            save_masks=args.save_masks,
            mask_dir_name=f'{args.image_dir_name}_masks',
        )
        stats['scene'] = scene_dir.name
        stats['image_dir'] = args.image_dir_name
        all_stats.append(stats)
        print(f'[{args.input_name}] {scene_dir.name}: processed={stats["processed_images"]}, augmented={stats["augmented_images"]}')

    metadata = {
        'source_dataset': args.input_name,
        'output_dataset': args.output_name,
        'image_dir_name': args.image_dir_name,
        'seed': args.seed,
        'augment_prob': args.augment_prob,
        'coverage_min': args.coverage_min,
        'coverage_max': args.coverage_max,
        'save_masks': args.save_masks,
        'scenes': all_stats,
    }
    write_metadata(dst_root, metadata)
    print(f'Wrote: {dst_root}')


if __name__ == '__main__':
    main()
