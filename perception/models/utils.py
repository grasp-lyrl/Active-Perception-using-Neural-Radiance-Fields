"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence, Callable, Dict, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from torch import Tensor

from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map


import sys

sys.path.append("perception/nerfacc")
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
)

sys.path.append("perception/nerfacc/nerfacc")
from volrend import accumulate_along_rays, render_weight_from_alpha

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid_with_depth_guide(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    def rgb_sigma_sem_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas, sems = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas, sems = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1), sems

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            depth=depth,
        )

        # device = "cuda:" + str(ray_indices.get_device())

        # for i in range(20):
        #     t_p = torch.normal(mean=depth, std=0.3).to(device)
        #     idx = torch.arange(depth.shape[0]).to(device)
        #     # print(t_p.shape)
        #     # print(idx.shape)
        #     ray_indices = torch.hstack((ray_indices, idx))
        #     t_starts = torch.hstack((t_starts, t_p))
        #     t_ends = torch.hstack((t_ends, t_p))

        #     # print(ray_indices.shape)
        #     # print(t_starts.shape)
        #     # print(t_ends.shape)

        if radiance_field.num_semantic_classes > 0:
            rgb, opacity, depth, semantics, extras = sem_rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_sem_fn=rgb_sigma_sem_fn,
                render_bkgd=render_bkgd,
                num_sumantic_classes=radiance_field.num_semantic_classes,
            )
            chunk_results = [rgb, opacity, depth, semantics, len(t_starts)]
            results.append(chunk_results)
        else:
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [rgb, opacity, depth, len(t_starts)]
            results.append(chunk_results)

    if radiance_field.num_semantic_classes > 0:
        colors, opacities, depths, semantics, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            semantics.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )
    else:
        colors, opacities, depths, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    def rgb_sigma_sem_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas, sems = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas, sems = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1), sems

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if radiance_field.num_semantic_classes > 0:
            rgb, opacity, depth, semantics, extras = sem_rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_sem_fn=rgb_sigma_sem_fn,
                render_bkgd=render_bkgd,
                num_sumantic_classes=radiance_field.num_semantic_classes,
            )
            chunk_results = [rgb, opacity, depth, semantics, len(t_starts)]
            results.append(chunk_results)
        else:
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [rgb, opacity, depth, len(t_starts)]
            results.append(chunk_results)

    if radiance_field.num_semantic_classes > 0:
        colors, opacities, depths, semantics, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            semantics.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )
    else:
        colors, opacities, depths, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )


def sem_rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_sem_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    # semantics info
    num_sumantic_classes: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """
    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_sem_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density and semantics
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1),
        semantics (n_rays, num_semantics classes), and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")
    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_sem_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma, color, and semantics with gradients
    if rgb_sigma_sem_fn is not None:
        if t_starts.shape[0] != 0:
            rgbs, sigmas, sems = rgb_sigma_sem_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
            sems = torch.empty((0, num_sumantic_classes), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        assert (
            sems.shape[-1] == num_sumantic_classes
        ), "sems must have shape of (N, num_sumantic_classes)! Got {}".format(
            sems.shape
        )
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }

    # Rendering: accumulate rgbs, opacities, depths, and semantics along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)
    semantics = accumulate_along_rays(
        weights,
        values=sems,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, semantics, extras


def render_image_with_propnet(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        extras,
    )


@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0

        if positions.shape[0] == 0:
            # print("position is empty")
            sigma = torch.zeros(0, device=positions.device)
            color = torch.zeros(0, 3, device=positions.device)
            # print(color.shape, sigma.shape)
            return color, sigma

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    def rgb_sigma_sem_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0

        if positions.shape[0] == 0:
            # print("position is empty")
            sigma = torch.zeros(0, device=positions.device)
            color = torch.zeros(0, 3, device=positions.device)
            sem = torch.zeros(
                0, radiance_field.num_semantic_classes, device=positions.device
            )
            # print(color.shape, sigma.shape)
            return color, sigma, sem

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas, sems = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas, sems = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1), sems

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)
    sem = torch.zeros(num_rays, radiance_field.num_semantic_classes, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:
        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        if radiance_field.num_semantic_classes > 0:
            rgbs, sigmas, sems = rgb_sigma_sem_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )
            if radiance_field.num_semantic_classes > 0:
                sems = sems[vis_mask]

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        if radiance_field.num_semantic_classes > 0:
            accumulate_along_rays_(
                weights,
                values=sems,
                ray_indices=ray_indices,
                outputs=sem,
            )
        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)
    if radiance_field.num_semantic_classes > 0:
        return (
            rgb.view((*rays_shape[:-1], -1)),
            opacity.view((*rays_shape[:-1], -1)),
            depth.view((*rays_shape[:-1], -1)),
            sem.view((*rays_shape[:-1], -1)),
            total_samples,
        )
    else:
        return (
            rgb.view((*rays_shape[:-1], -1)),
            opacity.view((*rays_shape[:-1], -1)),
            depth.view((*rays_shape[:-1], -1)),
            total_samples,
        )


@torch.no_grad()
def render_probablistic_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0

        if positions.shape[0] == 0:
            # print("position is empty")
            sigma = torch.zeros(0, device=positions.device)
            color = torch.zeros(0, 3, device=positions.device)
            # print(color.shape, sigma.shape)
            return color, sigma

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    def rgb_sigma_sem_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0

        if positions.shape[0] == 0:
            # print("position is empty")
            sigma = torch.zeros(0, device=positions.device)
            color = torch.zeros(0, 3, device=positions.device)
            sem = torch.zeros(
                0, radiance_field.num_semantic_classes, device=positions.device
            )
            # print(color.shape, sigma.shape)
            return color, sigma, sem

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas, sems = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas, sems = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1), sems

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)
    sem = torch.zeros(num_rays, radiance_field.num_semantic_classes, device=device)

    depth_var = torch.zeros(num_rays, 1, device=device)
    rgb_var = torch.zeros(num_rays, 3, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:
        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        if radiance_field.num_semantic_classes > 0:
            rgbs, sigmas, sems = rgb_sigma_sem_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )
            if radiance_field.num_semantic_classes > 0:
                sems = sems[vis_mask]

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        if radiance_field.num_semantic_classes > 0:
            accumulate_along_rays_(
                weights,
                values=sems,
                ray_indices=ray_indices,
                outputs=sem,
            )

        # calculate variance
        accumulate_along_rays_(
            weights,
            values=torch.pow(rgbs - rgb[ray_indices], 2),
            ray_indices=ray_indices,
            outputs=rgb_var,
        )

        
        accumulate_along_rays_(
            weights,
            values=torch.pow(
                (t_starts + t_ends)[..., None] / 2.0 - depth[ray_indices], 2
            ),
            ray_indices=ray_indices,
            outputs=depth_var,
        )

        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)
    if radiance_field.num_semantic_classes > 0:
        return (
            rgb.view((*rays_shape[:-1], -1)),
            rgb_var.view((*rays_shape[:-1], -1)),
            opacity.view((*rays_shape[:-1], -1)),
            depth.view((*rays_shape[:-1], -1)),
            depth_var.view((*rays_shape[:-1], -1)),
            sem.view((*rays_shape[:-1], -1)),
            total_samples,
        )
    else:
        return (
            rgb.view((*rays_shape[:-1], -1)),
            rgb_var.view((*rays_shape[:-1], -1)),
            opacity.view((*rays_shape[:-1], -1)),
            depth.view((*rays_shape[:-1], -1)),
            depth_var.view((*rays_shape[:-1], -1)),
            total_samples,
        )
