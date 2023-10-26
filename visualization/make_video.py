import cv2
import os
import glob

num_frames = 5000

path = "/home/siminghe/code/ActiveNeRFMapping/data/nerfacc/habitat_collection+20231001-130040/viz"
# path_top = "/home/siminghe/code/ActiveNeRFMapping/data/nerfacc/habitat_collection+20230929-225344/viz/top"
# path_fpv =
img_folder_path = path
video_path = path + "/video_output.mov"

images = [img for img in sorted(glob.glob(f"{img_folder_path}/*.png"))]
print(len(images))

frame = cv2.imread(os.path.join(img_folder_path, images[0]))

# setting the frame width, height width
# the width, height of an image (assuming all images are the same size)
height, width, layers = frame.shape


# write a video with 20 fps
video = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*"DIVX"), 25, (width, height)
)

assert len(images) > num_frames

for i in range(num_frames):
    video.write(cv2.imread(os.path.join(img_folder_path, str(i) + ".png")))

cv2.destroyAllWindows()
video.release()
