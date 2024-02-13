import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import magnum as mn
import cv2
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation

sys.path.append("simulator")
from sim import HabitatSim
from occupancy_grid import VoxelGrid
from bresenhan import bresenhamline

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from typing import TYPE_CHECKING, Union, cast

import json
import copy

def display_sample(rgb_obs, depth_obs, semantic_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    # plt.savefig('sample.png')
    
def find_frontiers(grid):
    frontiers = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0:  # check if cell is free
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < grid.shape[0] and
                            0 <= ny < grid.shape[1] and
                            grid[nx, ny] == -1  # check if neighbor is unknown
                        ):
                            frontiers.append((x, y))
                            break
    return np.array(frontiers)

def main(scene, run_idx):
    scene_dataset_config_file = (
        "./data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
    )
    sim = HabitatSim(scene, scene_dataset_config_file, 640, 480)
    top_down_map = maps.get_topdown_map_from_sim(
                cast("HabitatSim", sim._sim), map_resolution=1024
            )
    recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )

    ### SEMANTIC GRID ###
    num_classes = 28
    # read gt objects file
    gt_obj_filepath = './simulator/objects_' + scene + '.json'
    gt_obj_json = json.load(open(gt_obj_filepath))

    # get num gt objs and locations
    gt_obj_locs = {i:[] for i in range(num_classes)}
    gt_objs_num = {i:0 for i in range(num_classes)}
    for tid, obj in gt_obj_json.items():
        gt_objs_num[obj['label']] += 1
        gt_obj_locs[obj['label']].append(obj['location'])
        
    det_sem_objs = [] # number of detected objects per class, per step

    sem_grids = []
    for i in range(num_classes):
        sem_grid = VoxelGrid(grid_size=500, grid_resolution=0.1, occupancy=False)
        sem_grids.append(sem_grid)
    ### SEMANTIC GRID ###

    hfov = np.pi / 2
    width = 640
    height = 480
    focal = 0.5 * width / np.tan(hfov / 2)
    fx = fy = focal
    cx, cy = width/2, height/2
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    if scene == "102344280":
        start_pose = np.array([-2, 1.5, -3.0, 0.0, 0.0, 0.0, 1.0])
    elif scene == "102344529":
        start_pose = np.array([-9.30928599, 1.5, -0.85245934, 0.0, 0.0, 0.0, 1.0])
    elif scene == "102344250":
        start_pose = np.array([-14.41991868, 1.5, -11.17515239])
        
    sample_poses = [start_pose]
    curr_pose = start_pose
    terminate = False

    voxel_grid_size = 100
    voxel_resolution = 0.5
    voxel_grid = None

    rotations = np.array([[0,0,0,1],
                            [0, 0.5, 0, 0.866,],
                            [0, 0.866, 0, 0.5],
                            [0, 1, 0, 0],
                            [0, 0.866, 0, -0.5],
                            [0, 0.5, 0, -0.866]])

    poses_to_explore = [start_pose]
    prev_pose = start_pose
    occ_grid = VoxelGrid(grid_size=voxel_grid_size, grid_resolution=voxel_resolution, occupancy=True)

    init = True
    visited_frontiers = []
    num_steps = 20

    det_dist_thresh = 0.5

    for step_idx in range(num_steps):
        curr_pose = poses_to_explore.pop(0)
        curr_pose = sim._sim.pathfinder.get_random_navigable_point_near(curr_pose[:3].astype(np.float32), 2.0)
        curr_pose = np.array([curr_pose[0], 1.5, curr_pose[2], 0.0, 0.0, 0.0, 1.0])
        
        if not init:
            sampled_path = sim.sample_path_2p(prev_pose[:3], curr_pose[:3])
            print(sampled_path)
            for i in range(sampled_path.shape[0]):
                sampled_path[i, 1] = 1.5
        else:
            init = False
            sampled_path = np.array([curr_pose])
        
        for m in range(sampled_path.shape[0]):
            if m == sampled_path.shape[0] - 1:
                num_obs = 6
            else:
                num_obs = 1
            for i in range(num_obs):
                # rotate on the spot in quaternion
                curr_pose = np.concatenate((sampled_path[m, :3], rotations[i]))
                # get observations
                rgbs, depths, sems = sim.sample_images_from_poses([curr_pose])
                occ_grid.insert_depth_image(depths[0], curr_pose)
                
                ### SEMANTIC GRID ###
                # update semantic grids
                for s in range(len(sem_grids)):
                    sem_id = s+1
                    # set all depth pixels to nan where the semantic class is not sem_id
                    sem_img = sems[0]
                    sem_depth_img = copy.deepcopy(depths[0])
                    sem_depth_img[sem_img != sem_id] = np.nan
                    has_pts = sem_grids[s].insert_depth_image(sem_depth_img, curr_pose)
                    if has_pts:
                        if gt_objs_num[s] == 0:
                            print('shouldnt have points')
                ### SEMANTIC GRID ###

        prev_pose = curr_pose
        # compute frontier points on occupancy grid
        gridtd = occ_grid.get_occupancy_grid()
        frontiers = find_frontiers(gridtd)

        # DBSCAN
        eps = 1  # max dist
        min_samples = 3  # min points
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(frontiers)
        
        poses_to_explore = []
        curr_pose_idx = np.array([(curr_pose[0] + voxel_grid_size/2) / voxel_resolution, (curr_pose[2] + voxel_grid_size/2) / voxel_resolution])
        # for each cluster, pick the point closest to the current pose
        clustered_pts = []
        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_points = frontiers[cluster_indices]
        
            # pick the point closest to the current pose
            closest_point = cluster_points[np.argmin(np.linalg.norm(cluster_points - curr_pose_idx, axis=1))]
            cp = [closest_point[0], closest_point[1]]
            if cp in visited_frontiers:
                continue
            # convert to pose
            clustered_pts.append(closest_point)
            poses_to_explore.append(np.array([(closest_point[0] -voxel_grid_size/2) * voxel_resolution, 1.5, (closest_point[1] - voxel_grid_size/2) * voxel_resolution]))

        # sort poses_to_explore based on distance to current pose, smallest first
        if (len(poses_to_explore) == 0):
            break
        poses_to_explore = sorted(poses_to_explore, key=lambda x: np.linalg.norm(x[:3] - curr_pose[:3]))
        # poses_to_explore.reverse()
        next_pose = poses_to_explore[0]
        next_pose = np.array([[next_pose[0]/voxel_resolution + voxel_grid_size/2, next_pose[2]/voxel_resolution + voxel_grid_size/2]])
        visited_frontiers.append([next_pose[0][0], next_pose[0][1]])
        
        ### SEMANTIC GRID ###
        # get objects from semantic grids
        sem_objs = []
        for i in range(len(sem_grids)):
            objs = []
            if sem_grids[i].initialized == False:
                sem_objs.append(objs)
                continue
            ptcloud = sem_grids[i].get_pointcloud()
            # dbscan to cluster points
            eps = 0.2 # max dist
            min_samples = 1 # min points
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(ptcloud)
            
            # number of clusters and their centroids
            n_clusters = len(np.unique(cluster_labels))
            cluster_centroids = []
            for label in np.unique(cluster_labels):
                if label == -1:
                    continue
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_points = ptcloud[cluster_indices]
                cluster_centroids.append(np.mean(cluster_points, axis=0))
            sem_objs.append(cluster_centroids)

        # match detected objects to gt objects
        valid_sem_objs = [[] for i in range(len(sem_objs))]
        gt_obj_loc_cnt = copy.deepcopy(gt_obj_locs)
        for i in range(len(sem_objs)):
            for j in range(len(sem_objs[i])):
                min_dist = 10.0
                best_idx = -1
                for k in range(len(gt_obj_loc_cnt[i])):
                    dist = np.linalg.norm(gt_obj_loc_cnt[i][k] - sem_objs[i][j])
                    if dist < det_dist_thresh and dist < min_dist:
                        min_dist = dist
                        best_idx = k
                if best_idx != -1:
                    gt_obj_loc_cnt[i].pop(best_idx)
                    valid_sem_objs[i].append(sem_objs[i][j])
        
        # # number of det objects per class for this step
        sem_objs_num = []          
        for i in range(len(sem_objs)):
            sem_objs_num.append(len(valid_sem_objs[i]))
        # add to full list of steps
        det_sem_objs.append(sem_objs_num)
        ### SEMANTIC GRID ###
        
        print('=====================================')
        print('Iteration', step_idx+1)  
        for i in range(num_classes):
            print('class {}: {} detected, {} gt'.format(i+1, sem_objs_num[i], gt_objs_num[i]))
        print('=====================================')

    sim._sim.close()

    for i in range(len(sem_grids)):
        print('Semantic class {}: {} detected, {} gt'.format(i+1, sem_objs_num[i], gt_objs_num[i]))

    # plot number of objects detected (y-axis) vs planning steps (x-axis)
    total_num_det_step = []
    total_num_det_gt = sum(gt_objs_num.values())

    for i in range(len(det_sem_objs)):
        total_num_det_step.append(sum(det_sem_objs[i]))
    # dotted line is the number of gt objects
    plt.figure(figsize=(8, 4))
    plt_range = [i for i in range(1, num_steps + 1)]
    plt.plot(plt_range, [total_num_det_gt for i in range(num_steps)], 'b--')
    plt.plot(plt_range, total_num_det_step, 'r')
    plt.xticks(plt_range)
    # yticks on 0 and last step
    plt.xlabel('Planning Steps')
    plt.ylabel('Number of Objects Detected')
    plt.savefig('obj_detection_' + scene + '.png')
    
    for i in range(len(total_num_det_step)-1):
        total_num_det_step[i+1] = max(total_num_det_step[i], total_num_det_step[i+1])
    # frontier_det = np.clip(frontier_det, 0, frontier_det[-1])
    total_num_det_step = np.insert(total_num_det_step, 0, 0)


    # save data
    np.save('frontier_det_' + scene + '_dist0.5_cluster0.2_3_' + str(run_idx) + '.npy', total_num_det_step)

    # for i in range(28):
    #     print('Semantic class {}: {} detected, {} gt'.format(i+1, sem_objs_num[i], gt_objs_num[i]))

if __name__ == '__main__':
    scenes = ["102344280", "102344529", "102344250"]
    for scene in scenes:
        for i in range (1,4,1):
            main(scene, i)
