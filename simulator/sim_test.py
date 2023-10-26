import numpy as np
import matplotlib.pyplot as plt
from sim import HabitatSim
import os
import magnum as mn
import cv2
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

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
    plt.savefig('sample.png')
    
# scene = "./data/scene_datasets/habitat-test-scenes/apartment_1.glb"
# scene_dataset_config_file = ""
# scene = "Baked_sc1_staging_00"
# scene_dataset_config_file = (
#     "data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"
# )
scene = "102344280"
scene_dataset_config_file = (
    "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
)
# scene_filepath = "data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
sim = HabitatSim(scene, scene_dataset_config_file, 640, 480)

# # load object template
# rigid_obj_mgr = sim.sim.get_rigid_object_manager()
# obj_templates_mgr = sim.sim.get_object_template_manager()
# quad_template_id = obj_templates_mgr.load_configs("./simulator/assets/quad")[0]
# quad_template = obj_templates_mgr.get_template_by_id(quad_template_id)
# quad_template.scale = np.array([0.2, 0.2, 0.2])
# obj_templates_mgr.register_template(quad_template)
# quad_obj = rigid_obj_mgr.add_object_by_template_id(quad_template_id, sim.sim.agents[0].scene_node)

# sample images along fake trajectory
radius = 0.5
num_points = 100  # Number of points in the circle
start_position = np.array([-3, 1.5, -3.0, 0.0, 0.0, 0.0, 1.0])

translation_list = []
quaternion_list = []

for i in range(num_points):
    theta = i * (2 * np.pi / num_points)
    x = start_position[0] + radius * np.cos(theta)
    z = start_position[2] - radius * np.sin(theta)
    translation = np.array([x, start_position[1], z])
    quaternion = np.array([0.0, np.sin(theta / 2), 0.0, np.cos(theta / 2)])
    translation_list.append(translation)
    quaternion_list.append(quaternion)
translation_array = np.array(translation_list)
quaternion_array = np.array(quaternion_list)
sample_poses = np.concatenate((translation_array, quaternion_array), axis=1)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
# out = cv2.VideoWriter('out.mp4', fourcc, 10.0, (640, 480))
# sampled_images = sim.render_tpv(sample_poses)
rgbs, depths, sems = sim.sample_images_from_poses(sample_poses)
for rgb, depth, sem in zip(rgbs, depths, sems):
    display_sample(rgb, depth, sem)
    # cv2.imshow("img", img)
    # cv2.waitKey(100)
#     out.write(img)
# out.release()

# run trajectory
# for pose in sample_poses:
#     sim.set_state(pose)
#     rgb, _ = sim.get_observations()
#     plt.imshow(rgb)
#     plt.show()

# sim.sim.add_trajectory_object("viz", sample_traj)
