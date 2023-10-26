import cv2
import os
import glob
import numpy as np
from ipdb import set_trace as st
import tqdm


num_frames = 4600

path = "/home/siminghe/code/ActiveNeRFMapping/data/nerfacc/viz_scene3/viz"
path_top = path + "/top"
path_rgb_gt = path + "/fpv/gt_rgb"
path_rgb_pd = path + "/fpv/pd_rgb"
path_dep_gt = path + "/fpv/gt_dep"
path_dep_pd = path + "/fpv/pd_dep"
path_sem_gt = path + "/fpv/gt_sem"
path_sem_pd = path + "/fpv/pd_sem"

img_folder_path = path
video_path = path + "/fpv/video_output.mov"

font = cv2.FONT_HERSHEY_SIMPLEX

images = [img for img in sorted(glob.glob(f"{img_folder_path}/*.png"))]
print(len(images))

frame = cv2.imread(os.path.join(img_folder_path, images[0]))

# setting the frame width, height width
# the width, height of an image (assuming all images are the same size)
height, width, layers = frame.shape
print(frame.shape)

# write a video with 20 fps
video = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*"DIVX"), 20, (int(width * 5 / 3), height)
)

assert len(images) > num_frames

for i in tqdm.tqdm(range(num_frames)):
    if i % 2 == 1:
        continue
    img = np.zeros((height, int(width * 5 / 3), 3), dtype=np.uint8)

    tpv = cv2.imread(os.path.join(img_folder_path, str(i) + ".png"))
    top = cv2.imread(os.path.join(path_top, str(i) + ".png"))
    rgb_gt = cv2.imread(os.path.join(path_rgb_gt, str(i) + ".png"))
    rgb_pd = cv2.imread(os.path.join(path_rgb_pd, str(i) + ".png"))
    dep_gt = cv2.imread(os.path.join(path_dep_gt, str(i) + ".png"))
    dep_pd = cv2.imread(os.path.join(path_dep_pd, str(i) + ".png"))
    sem_gt = cv2.imread(os.path.join(path_sem_gt, str(i) + ".png"))
    sem_pd = cv2.imread(os.path.join(path_sem_pd, str(i) + ".png"))

    top_size = (int(width / 3), int(height / 3))
    top = cv2.resize(top, top_size)
    tpv[0 : top_size[0], width - top_size[1] : width] = top

    img[0:height, 0:width] = tpv

    rgb_gt = cv2.resize(rgb_gt, (int(width / 3), int(height / 3)))
    rgb_pd = cv2.resize(rgb_pd, (int(width / 3), int(height / 3)))
    dep_gt = cv2.resize(dep_gt, (int(width / 3), int(height / 3)))
    dep_pd = cv2.resize(dep_pd, (int(width / 3), int(height / 3)))
    sem_gt = cv2.resize(sem_gt, (int(width / 3), int(height / 3)))
    sem_pd = cv2.resize(sem_pd, (int(width / 3), int(height / 3)))

    img[0 : int(height / 3), width : width + int(width / 3)] = rgb_gt
    img[
        0 : int(height / 3), width + int(width / 3) : width + int(width * 2 / 3)
    ] = rgb_pd

    img[int(height / 3) : int(height * 2 / 3), width : width + int(width / 3)] = dep_gt
    img[
        int(height / 3) : int(height / 3) * 2,
        width + int(width / 3) : width + int(width / 3) * 2,
    ] = dep_pd

    img[
        int(height / 3) * 2 : int(height / 3) * 3, width : width + int(width / 3)
    ] = sem_gt
    img[
        int(height / 3) * 2 : int(height / 3) * 3,
        width + int(width / 3) : width + int(width / 3) * 2,
    ] = sem_pd

    cv2.putText(img, "2X", (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    video.write(img)
    # video.write(cv2.imread(os.path.join(img_folder_path, str(i) + ".png")))

cv2.destroyAllWindows()
video.release()
