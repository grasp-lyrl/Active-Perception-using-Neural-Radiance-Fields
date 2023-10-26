"""
Imports
"""
# Vehicles. Currently there is only one.
# There must also be a corresponding parameter file.
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

import tqdm
import datetime

# from rotorpy.vehicles.hummingbird_params import quad_params  # There's also the Hummingbird

# You will also need a controller (currently there is only one) that works for your vehicle.
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.minsnap import MinSnap

from rotorpy.simulate import (
    time_exit,
    merge_dicts,
    sanitize_trajectory_dic,
    sanitize_control_dic,
)

from scipy.spatial.transform import Rotation as R

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from dijkstra import Dijkstra

import sys

sys.path.append("perception/data_proc")
from depth_to_grid import generate_ray_casting_grid_map, bresenham

# Reference the files above for more documentation.

# Other useful imports
import numpy as np  # For array creation/manipulation
import matplotlib.pyplot as plt  # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import (
    Rotation,
)  # For doing conversions between different rotation descriptions, applying rotations, etc.
import os
import copy


def sample_waypoints_from_free_space(
    voxel_grid, current_state, aabb, voxel_grid_size, N=10
):
    """
    In: voxel_map, N sample waypoints,
    Out: Free space waypoints points

    ToDo: Sobol sampling
    """
    ## use channel 0
    voxel_ch = voxel_grid[0]
    shape = voxel_ch.shape
    ## get indices where voxels == 0
    free_indices = np.argwhere(voxel_ch == 0)

    current_state = current_state - aabb[:3]

    current_vox = world2voxels(current_state, voxel_grid_size)

    vertical_voxels = (aabb[5] - aabb[2]) // voxel_grid_size

    surrounding = (
        (free_indices[:, 2] >= int(vertical_voxels / 3))
        & (free_indices[:, 2] <= int(vertical_voxels * 2 / 3))
        & (
            (free_indices[:, 0] >= np.clip(current_vox[0] + 2, 0, shape[0]))
            | (free_indices[:, 0] <= np.clip(current_vox[0] - 2, 0, shape[0]))
            | (free_indices[:, 1] >= np.clip(current_vox[1] + 2, 0, shape[1]))
            | (free_indices[:, 1] <= np.clip(current_vox[1] - 2, 0, shape[1]))
        )
    )

    free_indices = free_indices[surrounding]
    ## N samples out of len(free_indices)

    free_samples = np.random.choice(len(free_indices), N, replace=False)
    ## get sample indices
    sample_indices = free_indices[free_samples]
    sample_indices = sample_indices
    x_samples = voxels2world(sample_indices, voxel_grid_size) + aabb[:3]
    return x_samples


def get_voxels_between_points(
    start_pos, end_pose, current_voxel, end_voxel, voxel_size
):
    # start_time = time.time()

    vx_ls = []

    current_voxel = current_voxel.astype(np.int32)
    view_point = np.copy(current_voxel)
    start_pos = start_pos.astype(np.float64)
    last_voxel = end_voxel.astype(np.int32)
    end_pos = end_pose.astype(np.float64)

    ray = (end_pos - start_pos).astype(np.float64)

    step = np.full(3, -1, dtype=np.float64)
    step[ray >= 0] = 1

    next_intersection = ((current_voxel + step) * voxel_size).astype(np.float64)

    t_max = np.array(
        [
            (next_intersection[i] - start_pos[i]) / ray[i] if ray[i] != 0 else np.inf
            for i in range(0, 3)
        ],
        dtype=np.float64,
    )

    t_delta = np.array(
        [voxel_size / ray[i] * step[i] if ray[i] != 0 else np.inf for i in range(0, 3)],
        dtype=np.float64,
    )

    neg_ray = False
    diff = np.zeros(3, dtype="int32")
    for i in range(0, 3):
        if current_voxel[i] != last_voxel[i] and ray[i] < 0:
            diff[i] -= 1
            neg_ray = True

    dist = np.sum(np.power((current_voxel - view_point) * voxel_size, 2))
    range_sq = np.sum(np.power((last_voxel - view_point) * voxel_size, 2))

    while dist <= range_sq:
        if t_max[0] < t_max[1]:
            if t_max[0] < t_max[2]:
                current_voxel[0] += step[0]
                t_max[0] += t_delta[0]
            else:
                current_voxel[2] += step[2]
                t_max[2] += t_delta[2]
        else:
            if t_max[1] < t_max[2]:
                current_voxel[1] += step[1]
                t_max[1] += t_delta[1]
            else:
                current_voxel[2] += step[2]
                t_max[2] += t_delta[2]

        vx_ls.append(np.copy(current_voxel))
        dist = np.sum(np.power((current_voxel - view_point) * voxel_size, 2))

    return vx_ls


def collision_checker(voxel_grid, flat, voxel_grid_size, aabb):
    x = flat["x"]
    voxel_x_idx = world2voxels(x - aabb[:3], voxel_grid_size)
    intersected_voxels = np.array(
        get_voxels_between_points(
            x[0], x[-1], voxel_x_idx[0], voxel_x_idx[-1], voxel_grid_size
        )
    )
    voxel_ch = voxel_grid[0]
    print(np.max(intersected_voxels[:, 0]))
    print(np.max(intersected_voxels[:, 1]))
    print(np.max(intersected_voxels[:, 2]))
    in_collision = voxel_ch[
        np.clip(intersected_voxels[:, 0], 0, voxel_ch.shape[0] - 1),
        np.clip(intersected_voxels[:, 1], 0, voxel_ch.shape[1] - 1),
        np.clip(intersected_voxels[:, 2], 0, voxel_ch.shape[2] - 1),
    ].any()
    return in_collision


def voxels2world(voxel_x_idx, voxel_grid_size=0.1):
    x = voxel_x_idx * voxel_grid_size
    return x


def world2voxels(x, voxel_grid_size=0.1):
    voxel_x_idx = np.array(x // voxel_grid_size, dtype=int)
    return voxel_x_idx


def update_cost_map(cost_map, depth, angle, g_loc, w_loc, aabb, resolution):
    ox = np.sin(-angle) * depth + w_loc[0]
    oy = -np.cos(-angle) * depth + w_loc[2]
    (
        occupancy_map,
        min_x,
        max_x,
        min_y,
        max_y,
        xy_resolution,
    ) = generate_ray_casting_grid_map(
        ox,
        oy,
        cost_map.shape[0],
        cost_map.shape[1],
        g_loc[0],
        g_loc[2],
        aabb,
        resolution,
    )

    cost_map[occupancy_map > 0.9] = 1
    cost_map[occupancy_map < 0.1] = 0

    visiting_map = np.zeros(cost_map.shape)
    visiting_map[occupancy_map < 0.1] = 1

    return cost_map, visiting_map


def sample_traj(
    voxel_grid,
    current_state,
    N_traj,
    aabb,
    sim,
    cost_map,
    save_path,
    visiting_map,
    N_sample_disc=20,
    voxel_grid_size=0.1,
):
    """
    In: free space points
    Out: MinSnap trajectory
    """
    voxel_grid = np.squeeze(voxel_grid)
    v_idx = world2voxels(current_state - aabb[:3], voxel_grid_size)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    v_merge = voxel_grid[0, :, :, 8].astype(np.int32) + voxel_grid[1, :, :, 8].astype(
        np.int32
    )

    path_finding_map = (v_merge > 1e-4).astype(np.int32)

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    path_finding_map = np.array(
        np.array(
            signal.convolve2d(
                path_finding_map,
                kernel,
                boundary="symm",
                mode="same",
            )
        )
        > 1e-4
    ).astype(np.int32)
    path_finding_map[v_idx[1], v_idx[0]] = False
    path_finding_map[v_idx[1] + 1, v_idx[0]] = False
    path_finding_map[v_idx[1] - 1, v_idx[0]] = False
    path_finding_map[v_idx[1], v_idx[0] + 1] = False
    path_finding_map[v_idx[1], v_idx[0] - 1] = False

    vm = np.copy(visiting_map)
    vm[(path_finding_map > 1e-4) == 1] = -1
    vm[(path_finding_map > 1e-4) == 0] = np.exp(
        -(
            vm[(path_finding_map > 1e-4) == 0]
            - np.min(vm[(path_finding_map > 1e-4) == 0])
        )
        / 5
    )

    print("saving map")
    if not os.path.exists(save_path + "/maps"):
        os.makedirs(save_path + "/maps")
    plt.imshow(vm, vmin=-1, vmax=1)
    plt.plot(v_idx[1], v_idx[0], "r*")
    plt.colorbar()
    plt.savefig(save_path + "/maps/vmap_" + str(current_time) + ".png")
    plt.clf()
    np.save(save_path + "/maps/vmap_" + str(current_time) + ".npy", vm)

    dijkstra = Dijkstra(aabb, path_finding_map, voxel_grid_size, 0.05)

    controller = SE3Control(quad_params)

    N_sample_traj_pose = []
    for _ in tqdm.tqdm(range(N_traj)):
        in_collision = True

        while in_collision:
            # free_indices = np.argwhere(path_finding_map == 0)
            free_indices = np.argwhere(vm >= 0)
            # st()
            vst_times = visiting_map[free_indices[:, 0], free_indices[:, 1]]
            exponent = np.exp(-(vst_times - np.min(vst_times)) * 0)
            distribution = exponent / np.sum(exponent)
            # st()
            free_samples = np.random.choice(
                len(free_indices), 1, replace=False, p=distribution
            )
            ## get sample indices
            sample_indices = free_indices[free_samples]
            sample_indices = sample_indices
            sample_indices = np.array([np.append(sample_indices[0], 0)])
            x_samples = voxels2world(sample_indices, voxel_grid_size) + aabb[:3]

            x_samples[0, 2] = 1.5
            end_idx = world2voxels(x_samples - aabb[:3], voxel_grid_size)
            # st()
            crr_world = current_state - aabb[:3]
            end_world = x_samples - aabb[:3]

            path = dijkstra.planning(
                crr_world[0], crr_world[1], end_world[0, 0], end_world[0, 1]
            )

            if path is None:
                # print("no path")
                # print(sample_indices)
                continue

            path[0].reverse()
            path[1].reverse()

            dij_waypoints = (
                np.array(
                    [
                        path[0],
                        path[1],
                        np.linspace(1.7, 1.7, len(path[0])),
                    ]
                ).T
                + aabb[:3]
            )
            yaw = np.linspace(2 * np.pi, 0, len(dij_waypoints))

            trajectory = MinSnap(points=dij_waypoints, yaw_angles=yaw, v_avg=0.5)
            if not trajectory.initialize():
                print("trajectory init" + str(np.random.uniform(1.3, 1.7, 1).item()))
                continue

            if trajectory.null:
                print("trajectory null " + str(np.random.uniform(1.3, 1.7, 1).item()))
                continue

            t_final = np.sum(trajectory.delta_t)
            N_sample_disc = max(int(t_final * 20), 20)
            t_step = t_final / N_sample_disc
            time = [0]
            flat = [sanitize_trajectory_dic(trajectory.update(time[-1]))]
            control_ref = [
                sanitize_control_dic(controller.update_ref(time[-1], flat[-1]))
            ]
            exit_status = None
            while True:
                exit_status = exit_status or time_exit(time[-1], t_final)
                if exit_status:
                    break

                time.append(time[-1] + t_step)
                flat.append(sanitize_trajectory_dic(trajectory.update(time[-1])))
                control_ref.append(
                    sanitize_control_dic(controller.update_ref(time[-1], flat[-1]))
                )

            time = np.array(time, dtype=float)
            flat = merge_dicts(flat)
            control_ref = merge_dicts(control_ref)
            in_collision = False

        xzy_x = np.copy(flat["x"])
        xzy_x[:, 1] = flat["x"][:, 2]
        xzy_x[:, 2] = flat["x"][:, 1]

        for i in range(control_ref["cmd_q"].shape[0]):
            rot = Rotation.from_quat(control_ref["cmd_q"][i])
            rot = rot.as_rotvec()
            rot = np.array([-rot[0], rot[2], -rot[1]])
            rot = Rotation.from_rotvec(rot)
            control_ref["cmd_q"][i] = rot.as_quat()

        traj_x_quat = np.hstack((xzy_x, control_ref["cmd_q"]))

        ## (N_traj, len_traj, 7)
        for ang in np.linspace(0, 360, 20):
            quat = R.from_euler("y", ang, degrees=True).as_quat().tolist()
            end_pos = (traj_x_quat[-1, :3]).tolist()
            pos = np.array(end_pos + quat)
            traj_x_quat = np.vstack((traj_x_quat, pos))

        N_sample_traj_pose.append(traj_x_quat)

    return N_sample_traj_pose
