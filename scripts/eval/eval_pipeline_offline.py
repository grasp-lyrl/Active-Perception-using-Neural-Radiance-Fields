import numpy as np
import json
import copy
import sys

sys.path.append("simulator")
from occupancy_grid import VoxelGrid
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation

from matplotlib import pyplot as plt
import open3d as o3d

scenes = ['102344280', '102344529', '102344250']
# scene = '102344280'
scene = "102344529"

def update_sem_step(sem_grids, gt_obj_locs, det_dist_thresh=1.0):
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
        # sem_objs_num.append(len(cluster_centroids))

    # match detected objects to gt objects
    valid_sem_objs = [[] for i in range(len(sem_objs))]
    gt_obj_loc_cnt = copy.deepcopy(gt_obj_locs)
    for i in range(len(sem_objs)):
        for j in range(len(sem_objs[i])):
            min_dist = 10.0
            best_idx = -1
            for k in range(len(gt_obj_loc_cnt[i])):
                # only check first and third coords
                # dist = np.linalg.norm(gt_obj_loc_cnt[i][k][:3:2] - sem_objs[i][j][:3:2])
                dist = np.linalg.norm(gt_obj_loc_cnt[i][k] - sem_objs[i][j])
                if dist < det_dist_thresh and dist < min_dist:
                    min_dist = dist
                    best_idx = k
            if best_idx != -1:
                gt_obj_loc_cnt[i].pop(best_idx)
                valid_sem_objs[i].append(sem_objs[i][j])
            # else:
            #     print('no match found for class', i+1)
            #     print(sem_objs[i][j])
            #     print(gt_obj_loc_cnt[i])
    
    # # number of det objects per class for this step
    sem_objs_num = []          
    for i in range(len(sem_objs)):
        sem_objs_num.append(len(valid_sem_objs[i]))
    
    return sem_objs_num

def run_eval(scene, run_idx):
    
    if scene == '102344280':
        datapath = './data/scene1_' + str(run_idx) + '_data0.npz'
    elif scene == '102344529':
        datapath = './data/scene2_' + str(run_idx) + '_data0.npz'
    elif scene == '102344250':
        datapath = './data/scene3_' + str(run_idx) + '_data0.npz'
    
    data = np.load(datapath)
    
    num_steps = 20
    
    images = data['images']
    depths = data['depths']
    semantics = data['semantics']
    cam_poses = data['camtoworlds']
    K = data['K']

    K = np.hstack((K, np.zeros((3, 1))))
    K = np.vstack((K, np.array([0, 0, 0, 1])))

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
        
    det_dist_thresh = 1.0
    det_sem_objs = []
    steps_taken = 0

    for it, (depth_img, sem_img, pose) in enumerate(zip(depths, semantics, cam_poses)):
        # simulate steps
        if it > 39 and (it-39) % 30 == 0:
            det_sem_objs.append(update_sem_step(sem_grids, gt_obj_locs, det_dist_thresh=det_dist_thresh))
            steps_taken += 1
        if steps_taken == num_steps:
            break
            
        # convert transformation matrix to xyz and qxqyqzqw
        rotation_matrix = Rotation.from_matrix(pose[:3, :3]).as_quat()
        translation = pose[:3, 3]
        pose = np.concatenate((translation, rotation_matrix))
        
        # update semantic grids
        for s in range(len(sem_grids)):
            sem_id = s+1
            # set all depth pixels to nan where the semantic class is not sem_id
            sem_depth_img = copy.deepcopy(depth_img)
            sem_depth_img[sem_img != sem_id] = np.nan
            has_pts = sem_grids[s].insert_depth_image(sem_depth_img, pose)
            if has_pts:
                if gt_objs_num[s] == 0:
                    print('shouldnt have points')

    for i in range(len(sem_grids)):
        print('Semantic class {}: {} detected, {} gt'.format(i+1, det_sem_objs[-1][i], gt_objs_num[i]))

    # plot number of objects detected (y-axis) vs planning steps (x-axis)
    total_num_det_step = []
    total_num_det_gt = sum(gt_objs_num.values())

    for i in range(len(det_sem_objs)):
        total_num_det_step.append(sum(det_sem_objs[i]))
    for i in range(len(total_num_det_step)-1):
        total_num_det_step[i+1] = max(total_num_det_step[i], total_num_det_step[i+1])
    total_num_det_step = np.insert(total_num_det_step, 0, 0)

    np.save('results/eval_' + scene + '_dist0.5_cluster0.2_' + str(run_idx) + '.npy', total_num_det_step)
    
for scene in scenes:
    for i in range(1,4,1):
        run_eval(scene, i)