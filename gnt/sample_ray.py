import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2

    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data["rgb"] if "rgb" in data.keys() else None
        self.camera = data["camera"]
        self.rgb_path = data["rgb_path"]
        self.depth_range = data["depth_range"]
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(
                    self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor
                ).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_single_image(
            self.H, self.W, self.intrinsics, self.c2w_mat
        )
        # Store full rgb image before flattening for transient encoder
        self.rgb_full = self.rgb.clone() if self.rgb is not None else None
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if "src_rgbs" in data.keys():
            self.src_rgbs = data["src_rgbs"]
        else:
            self.src_rgbs = None
        if "src_cameras" in data.keys():
            self.src_cameras = data["src_cameras"]
        else:
            self.src_cameras = None
        
        # # Support for transient masks (scenario 2)
        # if "src_transient_masks" in data.keys():
        #     self.src_transient_masks = data["src_transient_masks"]
        # else:
        #     self.src_transient_masks = None

        # if "src_transient_masks" in data.keys():
        #     self.src_transient_masks = data["src_transient_masks"]
        #     if self.src_transient_masks.ndim == 4 and self.src_transient_masks.shape[0] == 1:
        #         self.src_transient_masks = self.src_transient_masks.squeeze(0)
        # else:
        #     self.src_transient_masks = None

        if "src_transient_masks" in data.keys():
            self.src_transient_masks = data["src_transient_masks"]

            # DataLoader may add batch dim: [1, n_views, H, W]
            if self.src_transient_masks.ndim == 4:
                assert self.src_transient_masks.shape[0] == 1, \
                    f"Unexpected src_transient_masks shape: {self.src_transient_masks.shape}"
                self.src_transient_masks = self.src_transient_masks.squeeze(0)  # -> [n_views, H, W]
        else:
            self.src_transient_masks = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        u, v = np.meshgrid(
            np.arange(W)[:: self.render_stride], np.arange(H)[:: self.render_stride]
        )
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x HW x 3
        return rays_o, rays_d

    def get_all(self):
        src_transient_masks = None
        if self.src_transient_masks is not None:
            masks = self.src_transient_masks
            assert masks.ndim == 3, f"Expected [n_views, H, W], got {masks.shape}"
            n_views, H, W = masks.shape
            src_transient_masks = masks.reshape(n_views, H * W)

        ret = {
            "ray_o": self.rays_o.cuda(),
            "ray_d": self.rays_d.cuda(),
            "depth_range": self.depth_range.cuda(),
            "camera": self.camera.cuda(),
            "rgb": self.rgb.cuda() if self.rgb is not None else None,
            "rgb_full": self.rgb_full.cuda() if self.rgb_full is not None else None,
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "src_transient_masks": src_transient_masks.cuda() if src_transient_masks is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == "center":
            border_H = int(self.H * (1 - center_ratio) / 2.0)
            border_W = int(self.W * (1 - center_ratio) / 2.0)

            # pixel coordinates
            u, v = np.meshgrid(
                np.arange(border_H, self.H - border_H), np.arange(border_W, self.W - border_W)
            )
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == "uniform":
            # Random from one image
            select_inds = rng.choice(self.H * self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        """
        :param N_rand: number of rays to be casted
        :return:
        """

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None
        
        # # Flatten transient masks if available (scenario 2) - lỗi
        # src_transient_masks = None
        # if self.src_transient_masks is not None:
        #     # src_transient_masks shape: [num_src, H, W]
        #     # Flatten to [num_src, H*W] for consistency
        #     num_src = self.src_transient_masks.shape[0]
        #     H, W = self.src_transient_masks.shape[1], self.src_transient_masks.shape[2]
        #     src_transient_masks = self.src_transient_masks.reshape(num_src, -1)

        # Flatten transient masks if available (scenario 2)
        src_transient_masks = None
        if self.src_transient_masks is not None:
            masks = self.src_transient_masks

            # # DataLoader adds batch dim: [1, num_src, H, W]
            # if masks.ndim == 4:
            #     assert masks.shape[0] == 1, f"Unexpected batch size for src_transient_masks: {masks.shape}"
            #     masks = masks.squeeze(0)   # -> [num_src, H, W]

            # assert masks.ndim == 3, f"Expected [num_src, H, W], got {masks.shape}"

            num_src, H, W = masks.shape
            src_transient_masks = masks.reshape(num_src, H * W)

        ret = {
            "ray_o": rays_o.cuda(),
            "ray_d": rays_d.cuda(),
            "camera": self.camera.cuda(),
            "depth_range": self.depth_range.cuda(),
            "rgb": rgb.cuda() if rgb is not None else None,
            "rgb_full": self.rgb_full.cuda() if self.rgb_full is not None else None,
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "selected_inds": select_inds,
        }
        
        # Add transient masks for source views if available (scenario 2)
        if src_transient_masks is not None:
            ret["src_transient_masks"] = src_transient_masks.cuda()
        
        return ret
