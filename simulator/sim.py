import math
import os
import random
import sys

import imageio
import numpy as np
import magnum as mn
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
import cv2

class HabitatSim:
    def __init__(self, scene, scene_dataset_config_file, img_w, img_h):
        self._sim_settings = {
            "scene": scene,  # Scene
            "scene_dataset_config_file": scene_dataset_config_file,  # Scene dataset config file
            "quad_agent_idx": 0,  # Index of the agent
            "sample_agent_idx": 1,  # Index of the agent
            "sensor_height": 0,  # Height of sensors in meters, relative to the agent
            "width": img_w,  # Spatial resolution of the observations
            "height": img_h,
        }
        self.cfg = self.make_simple_cfg(self._sim_settings)
        self._sim = habitat_sim.Simulator(self.cfg)
        self.quad_agent = self._sim.initialize_agent(
            self._sim_settings["quad_agent_idx"]
        )
        self.sample_agent = self._sim.initialize_agent(
            self._sim_settings["sample_agent_idx"]
        )

        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.0, 0.0])  # in world space
        self.quad_agent.set_state(agent_state)

        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        navmesh_settings.agent_radius = 0.1
        self._sim.recompute_navmesh(self._sim.pathfinder, habitat_sim.NavMeshSettings())

        # load quad object
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        obj_templates_mgr = self._sim.get_object_template_manager()
        quad_template_id = obj_templates_mgr.load_configs("./simulator/assets/quad")[0]
        quad_template = obj_templates_mgr.get_template_by_id(quad_template_id)
        quad_template.scale = np.array([0.1, 0.1, 0.1])
        obj_templates_mgr.register_template(quad_template)
        self.quad_obj = rigid_obj_mgr.add_object_by_template_id(
            quad_template_id, self._sim.agents[0].scene_node
        )

        self.ex_poses = []

    def add_visited_location(self, locations, r=0.001):
        self._sim.add_trajectory_object("final1", locations, radius=r)

    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        if settings["scene_dataset_config_file"] != "":
            sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
        sim_cfg.pbr_image_based_lighting = True

        # agent
        quad_agent_cfg = habitat_sim.agent.AgentConfiguration()
        sample_agent_cfg = habitat_sim.agent.AgentConfiguration()

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]
        # rgb_sensor_spec.hfov = 100

        tpv_sensor_spec = habitat_sim.CameraSensorSpec()
        tpv_sensor_spec.uuid = "third_person_view"
        tpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        tpv_sensor_spec.resolution = [settings["height"], settings["width"]]
        tpv_sensor_spec.position = [0.0, 0.5, 1.0]
        tpv_sensor_spec.orientation = [-0.5, 0.0, 0.0]
        # tpv_sensor_spec.hfov = 100

        sample_rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        sample_rgb_sensor_spec.uuid = "sample_rgb_sensor"
        sample_rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        sample_rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        sample_rgb_sensor_spec.position = [0.0, 0.0, 0.0]
        # sample_rgb_sensor_spec.hfov = 100

        sample_depth_sensor_spec = habitat_sim.CameraSensorSpec()
        sample_depth_sensor_spec.uuid = "sample_depth_sensor"
        sample_depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        sample_depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        sample_depth_sensor_spec.position = [0.0, 0.0, 0.0]
        # sample_depth_sensor_spec.hfov = 100

        sample_sem_sensor_spec = habitat_sim.CameraSensorSpec()
        sample_sem_sensor_spec.uuid = "sample_sem_sensor"
        sample_sem_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sample_sem_sensor_spec.resolution = [settings["height"], settings["width"]]
        sample_sem_sensor_spec.position = [0.0, 0.0, 0.0]
        # sample_sem_sensor_spec.hfov = 100

        quad_agent_cfg.sensor_specifications = [rgb_sensor_spec, tpv_sensor_spec]
        sample_agent_cfg.sensor_specifications = [
            sample_rgb_sensor_spec,
            sample_depth_sensor_spec,
            sample_sem_sensor_spec,
        ]

        return habitat_sim.Configuration(sim_cfg, [quad_agent_cfg, sample_agent_cfg])

    def reset(self):
        self.set_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    # def set_state(self, pose):
    #     # set state by scene node so that it can be off the nav mesh
    #     translation = mn.Vector3(pose[:3])
    #     rotation = mn.Quaternion(((pose[3:6]), pose[6]))
    #     self._sim.get_agent(0).scene_node.translation = translation
    #     self._sim.get_agent(0).scene_node.rotation = rotation

    # def get_state(self):
    #     translation = self._sim.get_agent(0).scene_node.translation
    #     rotation = self._sim.get_agent(0).scene_node.rotation
    #     translation = np.array([translation.x, translation.y, translation.z])
    #     rotation = np.array([rotation.vector.x, rotation.vector.y, rotation.vector.z, rotation.scalar])
    #     return np.concatenate([translation, rotation])

    def set_quad_state(self, pose):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pose[:3]
        agent_state.rotation = np.normalized(
            np.quaternion(pose[6], pose[3], pose[4], pose[5])
        )
        self.quad_agent.set_state(agent_state)

    def set_sample_state(self, pose):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pose[:3]
        agent_state.rotation = np.normalized(
            np.quaternion(pose[6], pose[3], pose[4], pose[5])
        )
        self.sample_agent.set_state(agent_state)

    def get_quad_state(self):
        translation = self.quad_agent.get_state().position
        rotation = self.quad_agent.get_state().rotation
        translation = np.array(translation)
        rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        return np.concatenate([translation, rotation])

    def get_observations(self):
        observations = self._sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        tpv = observations["third_person_view"]
        depth = observations["depth_sensor"]
        sample = observations["sample_sensor"]

        return rgb, tpv, depth

    def sample_images_from_poses(self, poses):
        """
        sample images from list of poses

        Args:
            poses: list of numpy arrays of pose (x, y, z, qx, qy, qz, qw)

        Returns:
            list of images
        """
        # move quad out of scene so it doesn't show up in the images
        quad_state = self.get_quad_state()
        self.set_quad_state(np.array([999.0, 999.0, 999.0, 0.0, 0.0, 0.0, 1.0]))

        rgbs = []
        depths = []
        sems = []
        for pose in poses:
            self.set_sample_state(pose)
            rgb = self._sim.get_sensor_observations(1)["sample_rgb_sensor"]
            depth = self._sim.get_sensor_observations(1)["sample_depth_sensor"]
            sem = self._sim.get_sensor_observations(1)["sample_sem_sensor"]
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgbs.append(rgb)
            depths.append(depth)
            sems.append(sem)
        return np.array(rgbs), np.array(depths), np.array(sems)

        # move quad back to original position
        self.set_quad_state(quad_state)

        return images

    def render_tpv_with_trail(self, poses, draw_traj=True):
        """
        move the quad and retrieve third-person view images from the chase cam behind the quad.
        this would be used for creating visualizations of the quad's trajectory in the scene

        Args:
            poses: list of numpy arrays of pose (x, y, z, qx, qy, qz, qw)

        Returns:
            list of images
        """
        images = []
        for pose in poses:
            # move quad
            self.set_quad_state(pose)
            # adjust third person view camera
            quad_state = self.quad_agent.get_state()
            camera_position = quad_state.sensor_states["third_person_view"].position
            camera_position[1] = quad_state.position[1] + 0.5
            camera_look_at = quad_state.position
            rot = mn.Quaternion.from_matrix(
                mn.Matrix4.look_at(
                    camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
                ).rotation()
            )
            quad_state.sensor_states["third_person_view"].position = camera_position
            quad_state.sensor_states["third_person_view"].rotation = np.quaternion(
                rot.scalar, rot.vector[0], rot.vector[1], rot.vector[2]
            )
            self.quad_agent.set_state(quad_state, infer_sensor_states=False)
            # get image
            tpv = self._sim.get_sensor_observations(0)["third_person_view"]
            if draw_traj:
                self.ex_poses.append(pose[:3])
                if len(self.ex_poses) > 50:
                    self.ex_poses.pop(0)
                ex_poses_plot = self.ex_poses[:-1]
                for i, ex_pose in enumerate(reversed(ex_poses_plot)):
                    point2d = self.get_2d_point(ex_pose, "third_person_view")
                    color_value = i / 20
                    color = self.interpolate_color(color_value)
                    tpv = cv2.circle(tpv, (point2d[0], point2d[1]), 5, color, -1)
            images.append(tpv)
        return images

    def render_tpv(self, poses, draw_traj=True):
        """
        move the quad and retrieve third-person view images from the chase cam behind the quad.
        this would be used for creating visualizations of the quad's trajectory in the scene

        Args:
            poses: list of numpy arrays of pose (x, y, z, qx, qy, qz, qw)

        Returns:
            list of images
        """
        images = []
        traj = poses[:, :3]
        traj_len = len(traj)
        for pose in poses:
            # move quad
            self.set_quad_state(pose)
            # adjust third person view camera
            quad_state = self.quad_agent.get_state()
            camera_position = quad_state.sensor_states["third_person_view"].position
            camera_position[1] = quad_state.position[1] + 0.5
            camera_look_at = quad_state.position
            rot = mn.Quaternion.from_matrix(
                mn.Matrix4.look_at(
                    camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
                ).rotation()
            )
            quad_state.sensor_states["third_person_view"].position = camera_position
            quad_state.sensor_states["third_person_view"].rotation = np.quaternion(
                rot.scalar, rot.vector[0], rot.vector[1], rot.vector[2]
            )
            self.quad_agent.set_state(quad_state, infer_sensor_states=False)
            # get image
            tpv = self._sim.get_sensor_observations(0)["third_person_view"]
            if draw_traj:
                traj = traj[1:]
                for i, tp in enumerate(reversed(traj)):
                    point2d = self.get_2d_point(tp, "third_person_view")
                    color_value = i / traj_len
                    color = self.interpolate_color(color_value)
                    # tpv = cv2.circle(
                    #     tpv, (int(point2d[0]), int(point2d[1])), 3, color, -1
                    # )
                    # point2d[0] = min(point2d[0], tpv.shape[1])
                    # point2d[0] = max(0, point2d[0])
                    # point2d[1] = min(point2d[1], tpv.shape[0])
                    # point2d[1] = max(0, point2d[1])
                    if (
                        point2d[0] < 0
                        or point2d[0] >= tpv.shape[1]
                        or point2d[1] < 0
                        or point2d[1] >= tpv.shape[0]
                    ):
                        continue

                    try:
                        tpv = cv2.circle(
                            tpv, (int(point2d[0]), int(point2d[1])), 5, color, -1
                        )
                    except cv2.error as error:
                        print("[Error]: {}".format(error))
            tpv = cv2.cvtColor(tpv, cv2.COLOR_BGR2RGB)
            images.append(tpv)
        return images

    def render_top_tpv(self, poses, draw_traj=True):
        """
        move the quad and retrieve third-person view images from the chase cam behind the quad.
        this would be used for creating visualizations of the quad's trajectory in the scene

        Args:
            poses: list of numpy arrays of pose (x, y, z, qx, qy, qz, qw)

        Returns:
            list of images
        """
        images = []
        traj = poses[:, :3]
        traj_len = len(traj)
        for pose in poses:
            # move quad
            self.set_quad_state(pose)
            # adjust third person view camera
            quad_state = self.quad_agent.get_state()
            # camera_position = quad_state.sensor_states["third_person_view"].position
            # camera_position[1] = quad_state.position[1] + 0.5
            camera_position = np.copy(quad_state.position)
            camera_position[1] = camera_position[1] + 3

            camera_look_at = quad_state.position
            # rot = mn.Quaternion.from_matrix(
            #     mn.Matrix4.look_at(
            #         camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
            #     ).rotation()
            # )
            quad_state.sensor_states["third_person_view"].position = camera_position
            quad_state.sensor_states["third_person_view"].rotation = np.quaternion(
                -7.07106781e-01, 7.07106781e-01, 0, 0
            )
            # 7.07106781e-01 -1.14270070e-18 -6.74106633e-18 -7.07106781e-01
            self.quad_agent.set_state(quad_state, infer_sensor_states=False)
            # get image
            tpv = self._sim.get_sensor_observations(0)["third_person_view"]
            if draw_traj:
                traj = traj[1:]
                for i, tp in enumerate(reversed(traj)):
                    point2d = self.get_2d_point(tp, "third_person_view")
                    color_value = i / traj_len
                    color = self.interpolate_color(color_value)
                    # point2d[0] = min(point2d[0], tpv.shape[1])
                    # point2d[0] = max(0, point2d[0])
                    # point2d[1] = min(point2d[1], tpv.shape[0])
                    # point2d[1] = max(0, point2d[1])
                    if (
                        point2d[0] < 0
                        or point2d[0] >= tpv.shape[1]
                        or point2d[1] < 0
                        or point2d[1] >= tpv.shape[0]
                    ):
                        continue
                    try:
                        tpv = cv2.circle(
                            tpv, (int(point2d[0]), int(point2d[1])), 5, color, -1
                        )
                    except cv2.error as error:
                        print("[Error]: {}".format(error))
            tpv = cv2.cvtColor(tpv, cv2.COLOR_BGR2RGB)
            images.append(tpv)
        return images

    def interpolate_color(self, value):
        blue = (1 - value) * 255
        red = value * 255
        return (int(blue), 0, int(red))

    def check_navigability(self, location):
        return self._sim.pathfinder.is_navigable(location[0])

    def sample_path(self, curr_loc):
        found_path = False
        cl = np.copy(curr_loc)
        cl[2] = cl[1]
        while not found_path:
            sample2 = self._sim.pathfinder.get_random_navigable_point()
            path = habitat_sim.ShortestPath()
            cl[1] = sample2[1]
            path.requested_start = cl
            path.requested_end = sample2
            found_path = self._sim.pathfinder.find_path(path)
            # geodesic_distance = path.geodesic_distance
            path_points = path.points
            # print(cl)
            # print(sample2)
            # print("not found path " + str(np.random.uniform(1.3, 1.7, 1).item()))
        return np.array(path_points)

    def get_2d_point(self, point_3d, sensor_name):
        # get the scene render camera and sensor object
        render_camera = self._sim._sensors[sensor_name]._sensor_object.render_camera

        # use the camera and projection matrices to transform the point onto the near plane
        projected_point_3d = render_camera.projection_matrix.transform_point(
            render_camera.camera_matrix.transform_point(point_3d)
        )
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / render_camera.projection_size()[0]
        point_2d += mn.Vector2(0.5)
        point_2d *= render_camera.viewport

        # convert to numpy array
        point2d = np.array([point_2d[0], point_2d[1]]).astype(int)

        return point2d
