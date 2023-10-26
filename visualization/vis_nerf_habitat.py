# general
import argparse
import pathlib
import time
import datetime
import copy
import imageio
import tqdm
import pdb
import curses
import sys
import os
import random

import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib

import torch

# matplotlib.use("Agg")
matplotlib.use("TkAgg")
from skimage import color, io


import threading

# habitat simulator
sys.path.append("simulator")
from sim import HabitatSim

# data processing
sys.path.append("perception/data_proc")
from habitat_to_data import Dataset

# nerfacc
sys.path.append("perception/models")
from utils import render_image_with_occgrid, render_image_with_occgrid_test
from radiance_fields.ngp import NGPRadianceField

sys.path.append("perception/nerfacc")
from nerfacc.estimators.occ_grid import OccGridEstimator

# define global variable
pose = np.array([[1, 1.5, 3, 0.0, 0.0, 0.0, 1.0]])
flag = False
continue_program = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/10stepExploration/model_0.pth",
        help="the path of the model to load",
    )
    parser.add_argument(
        "--observation_path",
        type=str,
        default="data/10stepExploration/data.npz",
        help="the path of the observation poses to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda device",
    )
    return parser.parse_args()


def viz_thread(args):
    global flag, pose, continue_program

    width = 640
    height = 480

    sim = HabitatSim(
        "102344280",
        str(
            pathlib.Path.cwd()
            / "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
        ),
        img_w=width,
        img_h=height,
    )

    # load poses
    observations = np.load(args.observation_path)
    locs = observations["camtoworlds"][:, :3, 3]
    # # print(camtoworlds.shape)
    sim.add_visited_location(locs, 0.1)

    # load model
    aabb = torch.tensor([-11, -0.2, -9, 10, 3.2, 14], device=args.device)
    main_grid_nlvl = 1
    main_grid_size = 0.2
    main_neurons = 64
    main_layer = 6
    # main_neurons = 128
    # main_layer = 3
    main_grid_resolution = (
        ((aabb.cpu().numpy()[3:] - aabb.cpu().numpy()[:3]) / main_grid_size)
        .astype(int)
        .tolist()
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb,
        resolution=main_grid_resolution,
        levels=main_grid_nlvl,
    ).to(args.device)

    radiance_field = NGPRadianceField(
        aabb=estimator.aabbs[-1],
        neurons=main_neurons,
        layers=main_layer,
    ).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    estimator.binaries = checkpoint["occ_grid"]
    radiance_field.load_state_dict(checkpoint["model"])

    hfov = np.pi / 2
    focal = 0.5 * width / np.tan(hfov / 2)

    if radiance_field.num_semantic_classes > 0:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle("Habitat and NeRF Visualization")
        (
            sampled_images,
            sampled_depth_images,
            sampled_sem_images,
        ) = sim.sample_images_from_poses(pose)

        viz_obj1 = axes[0, 0].imshow(sampled_images[0, :, :, :3])

        rgb, depth, _, sem = Dataset.render_image_from_pose(
            radiance_field,
            estimator,
            pose,
            width,
            height,
            focal,
            0.1,  # near_plane
            1e-3,  # render_step_size=self.render_step_size,
            1,  # scale DONT CHANGE
            0.004,  # cone_angle
            1e-2,  # alpha_thre
            1,  # downsample
            args.device,  # cuda device
        )
        # print(rgb.shape)
        viz_obj2 = axes[0, 1].imshow(rgb[0, :, :, :3])

        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(sampled_depth_images[0] / 10, 0, 1) * 255),
            cv2.COLORMAP_VIRIDIS,
        )

        viz_obj3 = axes[1, 0].imshow(depth_image)

        nerf_depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth[0] / 10, 0, 1) * 255),
            cv2.COLORMAP_VIRIDIS,
        )

        viz_obj4 = axes[1, 1].imshow(nerf_depth_image)

        sem_image_gt = ((color.label2rgb(sampled_sem_images[0])) * 255).astype(np.uint8)
        viz_obj5 = axes[2, 0].imshow(sem_image_gt)

        sem_collapse = np.argmax(sem[0], axis=-1)
        sem_image = ((color.label2rgb(sem_collapse)) * 255).astype(np.uint8)
        viz_obj6 = axes[2, 1].imshow(sem_image)

        while continue_program:
            if flag:
                print("viz")
                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = sim.sample_images_from_poses(pose)

                print(np.min(sampled_sem_images))

                rgb, depth, _ = Dataset.render_image_from_pose(
                    radiance_field,
                    estimator,
                    pose,
                    width,
                    height,
                    focal,
                    0.1,  # near_plane
                    1e-3,  # render_step_size=self.render_step_size,
                    1,  # scale DONT CHANGE
                    0.004,  # cone_angle
                    1e-2,  # alpha_thre
                    1,  # downsample
                    args.device,  # cuda device
                )

                # img = sim.sample_images_from_poses(pose)[0][0, :, :, :3]
                viz_obj1.set_data(sampled_images[0, :, :, :3])
                viz_obj2.set_data(rgb[0, :, :, :3])

                depth_image = cv2.applyColorMap(
                    cv2.convertScaleAbs(
                        np.clip(sampled_depth_images[0] / 10, 0, 1) * 255
                    ),
                    cv2.COLORMAP_VIRIDIS,
                )
                nerf_depth_image = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.clip(depth[0] / 10, 0, 1) * 255),
                    cv2.COLORMAP_VIRIDIS,
                )

                viz_obj3.set_data(depth_image)
                viz_obj4.set_data(nerf_depth_image)

                plt.pause(0.1)
                plt.draw()
                # plt.show()
                flag = False
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle("Habitat and NeRF Visualization")
        (
            sampled_images,
            sampled_depth_images,
            sampled_sem_images,
        ) = sim.sample_images_from_poses(pose)

        viz_obj1 = axes[0, 0].imshow(sampled_images[0, :, :, :3])

        rgb, depth, _ = Dataset.render_image_from_pose(
            radiance_field,
            estimator,
            pose,
            width,
            height,
            focal,
            0.1,  # near_plane
            1e-3,  # render_step_size=self.render_step_size,
            1,  # scale DONT CHANGE
            0.004,  # cone_angle
            1e-2,  # alpha_thre
            1,  # downsample
            args.device,  # cuda device
        )
        # print(rgb.shape)
        viz_obj2 = axes[0, 1].imshow(rgb[0, :, :, :3])

        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(sampled_depth_images[0] / 10, 0, 1) * 255),
            cv2.COLORMAP_VIRIDIS,
        )

        viz_obj3 = axes[1, 0].imshow(depth_image)

        nerf_depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth[0] / 10, 0, 1) * 255),
            cv2.COLORMAP_VIRIDIS,
        )

        viz_obj4 = axes[1, 1].imshow(nerf_depth_image)

        while continue_program:
            if flag:
                print("viz")
                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = sim.sample_images_from_poses(pose)

                print(np.min(sampled_sem_images))

                rgb, depth, _ = Dataset.render_image_from_pose(
                    radiance_field,
                    estimator,
                    pose,
                    width,
                    height,
                    focal,
                    0.1,  # near_plane
                    1e-3,  # render_step_size=self.render_step_size,
                    1,  # scale DONT CHANGE
                    0.004,  # cone_angle
                    1e-2,  # alpha_thre
                    1,  # downsample
                    args.device,  # cuda device
                )

                # img = sim.sample_images_from_poses(pose)[0][0, :, :, :3]
                viz_obj1.set_data(sampled_images[0, :, :, :3])
                viz_obj2.set_data(rgb[0, :, :, :3])

                depth_image = cv2.applyColorMap(
                    cv2.convertScaleAbs(
                        np.clip(sampled_depth_images[0] / 10, 0, 1) * 255
                    ),
                    cv2.COLORMAP_VIRIDIS,
                )
                nerf_depth_image = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.clip(depth[0] / 10, 0, 1) * 255),
                    cv2.COLORMAP_VIRIDIS,
                )

                viz_obj3.set_data(depth_image)
                viz_obj4.set_data(nerf_depth_image)

                plt.pause(0.1)
                plt.draw()
                # plt.show()
                flag = False


def move_thread():
    global flag, pose, continue_program
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)

    angle = np.pi / 20

    try:
        while True:
            ch = screen.getch()

            SE = np.eye(4)
            SE[:3, 3] = pose[0][:3]
            SE[:3, :3] = R.from_quat(pose[0][3:]).as_matrix()

            if ch == ord("w"):
                update = np.eye(4)
                update[1, 3] += 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("up to ", pose)
                flag = True
            elif ch == ord("s"):
                update = np.eye(4)
                update[1, 3] -= 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("down to ", pose)
                flag = True
            elif ch == ord("a"):
                update = np.eye(4)
                update[0, 0] = np.cos(angle)
                update[0, 2] = np.sin(angle)
                update[2, 0] = -np.sin(angle)
                update[2, 2] = np.cos(angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("yaw_left to ", pose)
                flag = True
            elif ch == ord("d"):
                update = np.eye(4)
                update[0, 0] = np.cos(-angle)
                update[0, 2] = np.sin(-angle)
                update[2, 0] = -np.sin(-angle)
                update[2, 2] = np.cos(-angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("yaw right to", pose)
                flag = True
            elif ch == ord("i"):
                update = np.eye(4)
                update[1, 1] = np.cos(angle)
                update[1, 2] = -np.sin(angle)
                update[2, 1] = np.sin(angle)
                update[2, 2] = np.cos(angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("pitch up to ", pose)
                flag = True
            elif ch == ord("k"):
                update = np.eye(4)
                update[1, 1] = np.cos(-angle)
                update[1, 2] = -np.sin(-angle)
                update[2, 1] = np.sin(-angle)
                update[2, 2] = np.cos(-angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("pitch down to", pose)
                flag = True
            elif ch == ord("j"):
                update = np.eye(4)
                update[0, 0] = np.cos(angle)
                update[0, 1] = -np.sin(angle)
                update[1, 0] = np.sin(angle)
                update[1, 1] = np.cos(angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("pitch up to ", pose)
                flag = True
            elif ch == ord("l"):
                update = np.eye(4)
                update[0, 0] = np.cos(-angle)
                update[0, 1] = -np.sin(-angle)
                update[1, 0] = np.sin(-angle)
                update[1, 1] = np.cos(-angle)

                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()

                print("pitch down to", pose)
                flag = True
            elif ch == curses.KEY_UP:
                update = np.eye(4)
                update[2, 3] -= 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("forward to ", pose)
                flag = True
            elif ch == curses.KEY_DOWN:
                update = np.eye(4)
                update[2, 3] += 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("backward to ", pose)
                flag = True
            elif ch == curses.KEY_RIGHT:
                update = np.eye(4)
                update[0, 3] += 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("right to ", pose)
                flag = True
            elif ch == curses.KEY_LEFT:
                update = np.eye(4)
                update[0, 3] -= 0.5
                SE = SE @ update
                pose[0][:3] = SE[:3, 3]
                pose[0][3:] = R.from_matrix(SE[:3, :3]).as_quat()
                print("left to ", pose)
                flag = True
            elif ch == ord("q"):
                continue_program = False
                break
            # print(flag)

    finally:
        curses.nocbreak()
        screen.keypad(0)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
    args = parse_args()

    viz_td = threading.Thread(target=viz_thread, args=(args,))
    move_td = threading.Thread(target=move_thread)

    viz_td.start()
    move_td.start()

    viz_td.join()
    move_td.join()
