import open3d as o3d
import numpy as np
from ipdb import set_trace as st
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voxel_path",
        type=str,
        default="/home/siminghe/Code/ActiveNeRFMapping/data/habitat_collection+20230908-085240/prediction/voxel/20230908-102309_step_3998_0.npy",
        help="the path of the voxel grid to load",
    )


if __name__ == "__main__":
    args = parse_args()

    grids = np.load(
        args.voxel_path,
    )

    # print(grids[2].shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array(np.where(grids[0, :, :12, :] == True)).T
    )
    # o3d.visualization.draw_geometries([pcd])

    alpha = 2
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    # o3d.visualization.draw_geometries([voxel_grid])


# def generate_image_rays(width, height, focal, pose, device):
#     x, y = torch.meshgrid(
#         torch.arange(width),
#         torch.arange(height),
#         indexing="xy",
#     )
#     x = x.flatten().to(device)
#     y = y.flatten().to(device)

#     K = torch.tensor(
#         [
#             [focal, 0, width / 2.0],
#             [0, focal, height / 2.0],
#             [0, 0, 1],
#         ],
#         dtype=torch.float32,
#     ).to(device)

#     OPENGL_CAMERA = True

#     # generate rays
#     camera_dirs = F.pad(
#         torch.stack(
#             [
#                 (x - K[0, 2] + 0.5) / K[0, 0],
#                 (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
#             ],
#             dim=-1,
#         ),
#         (0, 1),
#         value=(-1.0 if OPENGL_CAMERA else 1.0),
#     )  # [num_rays, 3]

#     # [n_cams, height, width, 3]
#     directions = (camera_dirs[:, None, :] * pose[:, :3, :3]).sum(dim=-1)
#     origins = torch.broadcast_to(pose[:, :3, -1], directions.shape)
#     viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

#     return Rays(origins=origins, viewdirs=viewdirs)


# # Render Image
# def render_image(
#     radiance_field,
#     estimator,
#     rotation,
#     location,
#     width,
#     height,
#     focal,
#     near_plane,
#     render_step_size,
#     cone_angle,
#     alpha_thre,
#     device="cuda:0",
# ):
#     images = np.zeros((len(rotation), height, width, 3))
#     depths = np.zeros((len(rotation), height, width))

#     for i, (r, v) in enumerate(zip(rotation, location)):
#         pose = np.eye(4)
#         pose[:3, :3] = r
#         pose[:3, 3] = v
#         pose = torch.from_numpy(pose).unsqueeze(0).float().to(device)

#         rays = generate_image_rays(width, height, focal, pose, device)
#         render_bkgd = torch.zeros(3, device=device)

#         rgb, _, depth, _ = render_image_with_occgrid_test(
#             1024,
#             # scene
#             radiance_field,
#             estimator,
#             rays,
#             # rendering options
#             near_plane=near_plane,
#             render_step_size=render_step_size,
#             render_bkgd=render_bkgd,
#             cone_angle=cone_angle,
#             alpha_thre=alpha_thre,
#         )

#         images[i] = rgb.cpu().numpy().reshape((height, width, 3))
#         depths[i] = depth.cpu().numpy().reshape((height, width))

#     return images, depths
