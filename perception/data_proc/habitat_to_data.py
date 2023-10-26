"""
Adapted from 2022 Ruilong Li, UC Berkeley.
"""

import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

sys.path.append("perception/models")
from datasets.utils import Rays
from utils import (
    render_image_with_occgrid_test,
    render_probablistic_image_with_occgrid_test,
)


from ipdb import set_trace as st

from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt

import cv2

from skimage import io
from skimage import color


class Dataset(torch.utils.data.Dataset):
    """Gathered Dataset"""

    def __init__(
        self,
        training: bool,
        save_fp: str,
        num_rays: int = None,
        batch_over_images: bool = True,
        num_models: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_rays = num_rays
        self.batch_over_images = batch_over_images
        self.num_models = num_models
        self.bootstrap_indices = [
            np.array([]).astype(int) for _ in range(self.num_models - 1)
        ]
        self.images = None
        self.depths = None
        self.semantics = None
        self.camtoworlds = None
        self.training = training
        self.device = device
        self.save_fp = save_fp
        self.boot_scale = 0.7

        self.saved_batch = 0
        self.downsampled_end = None

        self.size = 0
        self.save_batch_size = 5000

        if not os.path.exists(self.save_fp):
            os.makedirs(self.save_fp)

    def resample_data(self):
        indices = np.random.choice(
            len(self.images), size=(int(len(self.images) * 0.7),), replace=False
        )
        self.images = self.images[indices]
        self.depths = self.depths[indices]
        self.semantics = self.semantics[indices]
        self.camtoworlds = self.camtoworlds[indices]

        # bootstrap indices
        self.bootstrap_indices = [
            np.array([]).astype(int) for _ in range(self.num_models - 1)
        ]
        for i, arr in enumerate(self.bootstrap_indices):
            ids = np.random.choice(
                len(self.images),
                size=(int(len(self.images) * self.boot_scale),),
                replace=True,
            )
            self.bootstrap_indices[i] = np.concatenate([arr, ids], axis=0)

    def update_data(self, images, depths, semantics, camtoworlds):
        """
        update_data
            when current self.images is greater than self.save_batch_size, save it to disk and clear it
        fetch_data
            when we want to retrieve data from a certain batch, we load it from disk and get it
        """
        if self.images is None:
            self.camtoworlds = (
                torch.from_numpy(camtoworlds).to(torch.float32).to(self.device)
            )
            self.images = torch.from_numpy(images).to(torch.uint8).to(self.device)
            self.depths = torch.from_numpy(depths).to(torch.float32).to(self.device)
            self.semantics = (
                torch.from_numpy(semantics.astype(np.int64))
                .to(torch.int64)
                .to(self.device)
            )

            # bootstrap indices
            for i, arr in enumerate(self.bootstrap_indices):
                ids = np.random.choice(
                    len(images),
                    size=(int(len(images) * self.boot_scale),),
                    replace=True,
                )
                self.bootstrap_indices[i] = np.concatenate(
                    [arr, self.size + ids], axis=0
                )

            self.height, self.width = self.images.shape[1:3]
            hfov = np.pi / 2
            focal = 0.5 * self.width / np.tan(hfov / 2)
            self.K = np.array(
                [
                    [focal, 0.0000, self.width / 2],
                    [0.0000, focal, self.height / 2],
                    [0.0000, 0.0000, 1.0000],
                ]
            )
            self.K = torch.tensor(self.K).to(torch.float32).to(self.device)
        else:
            for i, arr in enumerate(self.bootstrap_indices):
                ids = np.random.choice(
                    len(images),
                    size=(int(len(images) * self.boot_scale),),
                    replace=True,
                )
                ids = ids + self.size
                self.bootstrap_indices[i] = np.concatenate([arr, ids], axis=0)

            images = torch.from_numpy(images).to(torch.uint8).to(self.device)
            camtoworlds = (
                torch.from_numpy(np.array(camtoworlds))
                .to(torch.float32)
                .to(self.device)
            )
            self.images = torch.cat([self.images, images], dim=0)

            depths = torch.from_numpy(depths).to(torch.float32).to(self.device)
            self.depths = torch.cat([self.depths, depths], dim=0)
            semantics = (
                torch.from_numpy(semantics.astype(np.int64))
                .to(torch.int64)
                .to(self.device)
            )
            self.semantics = torch.cat([self.semantics, semantics], dim=0)

            self.camtoworlds = torch.cat([self.camtoworlds, camtoworlds], dim=0)

        self.size = self.size + len(images)

    def __len__(self):
        return self.size

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def save(self):
        np.savez(
            self.save_fp + "/data" + str(self.saved_batch) + ".npz",
            images=self.images.cpu().numpy(),
            depths=self.depths.cpu().numpy(),
            semantics=self.semantics.cpu().numpy(),
            camtoworlds=self.camtoworlds.cpu().numpy(),
            K=self.K.cpu().numpy(),
            bootstrap_indices=np.array(self.bootstrap_indices),
        )

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def bootstrap(self, model_idx):
        if model_idx == 0:
            return np.arange(self.size)
        else:
            return self.bootstrap_indices[model_idx - 1]

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]
        dep, sem = data["dep"], data["sem"]

        if self.training:
            # random during training to aid learning correct radiance field
            color_bkgd = torch.rand(3, device=self.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "dep": dep,  # [n_rays,] or [h, w]
            "sem": sem,  # [n_rays,] or [h, w]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays", "dep", "sem"]},
        }

    def fetch_data(self, index):
        """Images to Rays"""
        num_rays = self.num_rays

        if self.training:
            image_id = torch.randint(
                0,
                self.size,
                size=(1,),
                device=self.device,
            )

            x = torch.randint(0, self.width, size=(num_rays,), device=self.device)
            y = torch.randint(0, self.height, size=(num_rays,), device=self.device)
        else:
            image_id = torch.tensor([index], device=self.device)
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.device),
                torch.arange(self.height, device=self.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        dep = self.depths[image_id, y, x]  # (num_rays,)
        sem = self.semantics[image_id, y, x]  # (num_rays,)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)

        # -1.0 for OPENGL_CAMERA
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1] * -1.0,
                ],
                dim=1,
            ),
            (0, 1),
            value=-1.0,
        )  # [num_rays, 3]

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
            dep = torch.reshape(dep, (num_rays,))
            sem = torch.reshape(sem, (num_rays,))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))
            dep = torch.reshape(dep, (self.height, self.width))
            sem = torch.reshape(sem, (self.height, self.width))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "dep": dep,  # [h, w] or [num_rays,]
            "sem": sem,  # [h, w] or [num_rays,]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def generate_image_rays(pose, width, height, K, device):
        x, y = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )
        x = x.flatten().to(device)
        y = y.flatten().to(device)

        # generate rays
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5) / K[1, 1] * -1.0,
                ],
                dim=1,
            ),
            (0, 1),
            value=-1.0,
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * pose[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(pose[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        return Rays(origins=origins, viewdirs=viewdirs)

    # Render Image
    def render_image_from_pose(
        radiance_field,
        estimator,
        poses,
        width,
        height,
        focal,
        near_plane,
        render_step_size,
        scale,
        cone_angle,
        alpha_thre,
        downsample,
        device="cuda:0",
    ):
        images = np.zeros((poses.shape[0], int(height * scale), int(width * scale), 3))
        depths = np.zeros((poses.shape[0], int(height * scale), int(width * scale)))
        accs = np.zeros((poses.shape[0], int(height * scale), int(width * scale)))
        sems = np.zeros(
            (
                poses.shape[0],
                int(height * scale),
                int(width * scale),
                radiance_field.num_semantic_classes,
            )
        )

        for i, p in enumerate(poses):
            v = p[:3]
            so = R.from_quat(p[3:])

            pose = np.eye(4)
            pose[:3, :3] = so.as_matrix()
            pose[:3, 3] = v
            pose = torch.from_numpy(pose).unsqueeze(0).float().to(device)

            K = np.array(
                [
                    [focal, 0.0000, width / 2],
                    [0.0000, focal, height / 2],
                    [0.0000, 0.0000, 1.0000],
                ]
            )

            rs = Dataset.generate_image_rays(pose, width, height, K, device)
            idx = np.round(
                np.linspace(
                    0, len(rs.origins) - 1, int(height * scale) * int(width * scale)
                )
            ).astype(int)
            rays = Rays(origins=rs.origins[idx], viewdirs=rs.viewdirs[idx])

            render_bkgd = torch.zeros(3, device=device)

            if radiance_field.num_semantic_classes > 0:
                rgb, acc, depth, sem, _ = render_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                sems[i] = (
                    sem.cpu()
                    .numpy()
                    .reshape(
                        (
                            int(height * scale),
                            int(width * scale),
                            radiance_field.num_semantic_classes,
                        )
                    )
                )
            else:
                rgb, acc, depth, _ = render_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )

            images[i] = (
                rgb.cpu().numpy().reshape((int(height * scale), int(width * scale), 3))
            )
            depths[i] = (
                depth.cpu().numpy().reshape((int(height * scale), int(width * scale)))
            )
            accs[i] = (
                acc.cpu().numpy().reshape((int(height * scale), int(width * scale)))
            )

        if radiance_field.num_semantic_classes > 0:
            return images, depths, accs, sems
        else:
            return images, depths, accs

    def render_probablistic_image_from_pose(
        radiance_field,
        estimator,
        poses,
        width,
        height,
        focal,
        near_plane,
        render_step_size,
        scale,
        cone_angle,
        alpha_thre,
        downsample,
        device="cuda:0",
    ):
        images = np.zeros((poses.shape[0], int(height * scale), int(width * scale), 3))
        depths = np.zeros((poses.shape[0], int(height * scale), int(width * scale)))
        images_var = np.zeros(
            (poses.shape[0], int(height * scale), int(width * scale), 3)
        )
        depths_var = np.zeros((poses.shape[0], int(height * scale), int(width * scale)))
        accs = np.zeros((poses.shape[0], int(height * scale), int(width * scale)))
        sems = np.zeros(
            (
                poses.shape[0],
                int(height * scale),
                int(width * scale),
                radiance_field.num_semantic_classes,
            )
        )

        for i, p in enumerate(poses):
            v = p[:3]
            so = R.from_quat(p[3:])

            pose = np.eye(4)
            pose[:3, :3] = so.as_matrix()
            pose[:3, 3] = v
            pose = torch.from_numpy(pose).unsqueeze(0).float().to(device)

            K = np.array(
                [
                    [focal, 0.0000, width / 2],
                    [0.0000, focal, height / 2],
                    [0.0000, 0.0000, 1.0000],
                ]
            )

            rs = Dataset.generate_image_rays(pose, width, height, K, device)
            idx = np.round(
                np.linspace(
                    0, len(rs.origins) - 1, int(height * scale) * int(width * scale)
                )
            ).astype(int)
            rays = Rays(origins=rs.origins[idx], viewdirs=rs.viewdirs[idx])

            render_bkgd = torch.zeros(3, device=device)

            if radiance_field.num_semantic_classes > 0:
                (
                    rgb,
                    rgb_var,
                    acc,
                    depth,
                    depth_var,
                    sem,
                    _,
                ) = render_probablistic_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                sems[i] = (
                    sem.cpu()
                    .numpy()
                    .reshape(
                        (
                            int(height * scale),
                            int(width * scale),
                            radiance_field.num_semantic_classes,
                        )
                    )
                )
            else:
                (
                    rgb,
                    rgb_var,
                    acc,
                    depth,
                    depth_var,
                    _,
                ) = render_probablistic_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )

            images[i] = (
                rgb.cpu().numpy().reshape((int(height * scale), int(width * scale), 3))
            )
            depths[i] = (
                depth.cpu().numpy().reshape((int(height * scale), int(width * scale)))
            )
            images_var[i] = (
                rgb_var.cpu()
                .numpy()
                .reshape((int(height * scale), int(width * scale), 3))
            )
            depths_var[i] = (
                depth_var.cpu()
                .numpy()
                .reshape((int(height * scale), int(width * scale)))
            )
            accs[i] = (
                acc.cpu().numpy().reshape((int(height * scale), int(width * scale)))
            )

        if radiance_field.num_semantic_classes > 0:
            return images, images_var, depths, depths_var, accs, sems
        else:
            return images, images_var, depths, depths_var, accs
