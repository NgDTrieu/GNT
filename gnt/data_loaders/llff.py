import os
import hashlib
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys

sys.path.append("../")
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

import cv2

def create_random_mask(img_h, img_w, coverage_ratio=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    total_pixels = img_h * img_w
    target_pixels = int(total_pixels * coverage_ratio)

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


def apply_transient_augmentation(img, mask, aug_type='random', return_mask=False, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)

    if aug_type == 'random':
        aug_type = rng.choice(['noise', 'color', 'blur'])

    augmented_img = img.copy()
    mask_bool = mask.astype(bool)

    if aug_type == 'noise':
        noise = rng.normal(0.5, 0.2, img.shape)
        noise = np.clip(noise, 0, 1)
        augmented_img[mask_bool] = noise[mask_bool]

    elif aug_type == 'color':
        random_color = rng.rand(3)
        augmented_img[mask_bool] = random_color

    elif aug_type == 'blur':
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        augmented_img[mask_bool] = blurred[mask_bool]

    if return_mask:
        transient_mask = (1.0 - mask).astype(np.float32)  # 1=static, 0=transient
        return augmented_img, transient_mask

    return augmented_img

def stable_int_hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def make_rng(base_seed, *keys):
    seed = int(base_seed)
    for key in keys:
        seed = (seed * 1000003 + stable_int_hash(str(key))) % (2**32 - 1)
    return np.random.RandomState(seed)


class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.rootdir, "data/real_iconic_noface/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.eval_seed = getattr(args, "eval_seed", 20260408)
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        scenes = os.listdir(base_dir)
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor=4
            )
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if mode == "train":
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
                i_train = np.array(
                    [
                        j
                        for j in np.arange(int(poses.shape[0]))
                        if (j not in i_test and j not in i_test)
                    ]
                )
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        sample_rng = None
        if self.mode != "train":
            sample_rng = make_rng(self.eval_seed, "target", rgb_file)

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 20),
            tar_id=id_render,
            angular_dist_method="dist",
        )

        num_select_eff = min(num_select, len(nearest_pose_ids))
        if self.mode == "train":
            nearest_pose_ids = np.random.choice(
                nearest_pose_ids, num_select_eff, replace=False
            )
        else:
            nearest_pose_ids = sample_rng.choice(
                nearest_pose_ids, num_select_eff, replace=False
            )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train":
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        if self.mode == "train" and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        src_transient_masks = None

        # áp transient giả cho source views khi eval/infer
        if self.mode != "train":
            src_masks = []
            for i, src_id in enumerate(nearest_pose_ids):
                src_rgb_file = train_rgb_files[src_id]
                src_rng = make_rng(self.eval_seed, "target", rgb_file, "source", src_rgb_file)

                coverage = src_rng.uniform(0.10, 0.30)
                mask = create_random_mask(
                    src_rgbs.shape[1],
                    src_rgbs.shape[2],
                    coverage_ratio=coverage,
                    rng=src_rng,
                )
                aug_type = src_rng.choice(['noise', 'color', 'blur'])

                src_rgbs[i], transient_mask = apply_transient_augmentation(
                    src_rgbs[i],
                    mask,
                    aug_type=aug_type,
                    return_mask=True,
                    rng=src_rng,
                )
                src_masks.append(transient_mask)

            src_transient_masks = np.stack(src_masks, axis=0).astype(np.float32)  # [num_src, H, W]

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return_dict = {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }

        if src_transient_masks is not None:
            return_dict["src_transient_masks"] = torch.from_numpy(src_transient_masks)

        return return_dict
