# This script generates a video from a sequence of pictures.
import cv2
import os
from tqdm import tqdm
from videolib import generateVideoFromList, imageHiveOverview

# Path to the folder containing the pictures
rootpath = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/24.09_observation_OH/Images/'
first_dt = "240918-180000Z"
last_dt = "240918-190000Z"

frame_drop = 5 # We keep 1 frame every frame_drop frames. Put one to keep all frames.

# Get the list of folders in the rootpath
paths = [os.path.join(rootpath, f) for f in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, f))]
# Order the paths alphabetically
paths.sort() # Now this contains the path to all four RPis images

# Generate first and last picture name
hives = [path.split("/")[-1][1] for path in paths]
rpis = [path.split("/")[-1][3] for path in paths]
first_pics = ["hive" + hive + "_rpi" + rpi + "_" + first_dt + ".jpg" for hive, rpi in zip(hives, rpis)]
last_pics = ["hive" + hive + "_rpi" + rpi + "_" + last_dt + ".jpg" for hive, rpi in zip(hives, rpis)]

# Get the list of files in the folders
files = [os.listdir(path) for path in paths]
# Sort the files by name
for i in range(len(files)):
    files[i].sort()
    # Keep only the files that are between the first and last picture
    files[i] = [f for f in files[i] if first_pics[i] <= f <= last_pics[i]]
    # Drop some frames
    files[i] = files[i][::frame_drop]

final_imgs = []
print("Generating frames...")
for i in tqdm(range(len(files[0]))):
    imgs = [cv2.imread(os.path.join(paths[j], files[j][i])) for j in range(len(files))]

    assembled_img = imageHiveOverview(imgs, [files[j][i] for j in range(len(files))])
    # Add the image to the list of images
    final_imgs.append(assembled_img)

print("Number of frames: ", len(final_imgs))
# Show the first frame (numbers, not colors)

generateVideoFromList(final_imgs, dest = "outputVideos/", name = "video_" + first_dt + "_" + last_dt, fps=10, grayscale=False)