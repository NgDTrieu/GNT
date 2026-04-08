import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys

sys.path.append("../")
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

import cv2

def create_random_mask(img_h, img_w, coverage_ratio=0.1):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    total_pixels = img_h * img_w
    target_pixels = int(total_pixels * coverage_ratio)

    shape_type = np.random.choice(['rect', 'circle'])

    if shape_type == 'rect':
        max_h = int(np.sqrt(target_pixels * 2))
        max_w = int(np.sqrt(target_pixels * 2))
        h = np.random.randint(max(10, max_h // 3), min(img_h, max_h))
        w = np.random.randint(max(10, max_w // 3), min(img_w, max_w))
        y = np.random.randint(0, max(1, img_h - h))
        x = np.random.randint(0, max(1, img_w - w))
        mask[y:y+h, x:x+w] = 1
    else:
        radius = int(np.sqrt(target_pixels / np.pi))
        radius = max(5, min(min(img_h, img_w) // 2, radius))
        cy = np.random.randint(radius, max(radius + 1, img_h - radius))
        cx = np.random.randint(radius, max(radius + 1, img_w - radius))
        cv2.circle(mask, (cx, cy), radius, 1, -1)

    return mask


def apply_transient_augmentation(img, mask, aug_type='random', return_mask=False):
    if aug_type == 'random':
        aug_type = np.random.choice(['noise', 'color', 'blur'])

    augmented_img = img.copy()
    mask_bool = mask.astype(bool)

    if aug_type == 'noise':
        noise = np.random.normal(0.5, 0.2, img.shape)
        noise = np.clip(noise, 0, 1)
        augmented_img[mask_bool] = noise[mask_bool]

    elif aug_type == 'color':
        random_color = np.random.rand(3)
        augmented_img[mask_bool] = random_color

    elif aug_type == 'blur':
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        augmented_img[mask_bool] = blurred[mask_bool]

    if return_mask:
        transient_mask = (1.0 - mask).astype(np.float32)  # 1=static, 0=transient
        return augmented_img, transient_mask

    return augmented_img


class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.rootdir, "data/real_iconic_noface/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
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
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
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
            for i in range(src_rgbs.shape[0]):
                coverage = np.random.uniform(0.10, 0.30)
                mask = create_random_mask(src_rgbs.shape[1], src_rgbs.shape[2], coverage)
                aug_type = np.random.choice(['noise', 'color', 'blur'])
                src_rgbs[i], transient_mask = apply_transient_augmentation(
                    src_rgbs[i], mask, aug_type, return_mask=True
                )
                src_masks.append(transient_mask)

            src_transient_masks = np.stack(src_masks, axis=0)  # [num_src, H, W]

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
