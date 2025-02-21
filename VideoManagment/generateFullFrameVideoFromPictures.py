# This script generates a video from a sequence of pictures.
import cv2
import os
from tqdm import tqdm
from videolib import generateVideoFromList, imageHiveOverview
import multiprocessing

# Path to the folder containing the pictures
rootpath = '/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.11_aSensing_OH/Images/'
hive = "1"

first_dt = "241028-093000Z"
last_dt = "241114-190000Z"

frame_drop = 10 # We keep 1 frame every frame_drop frames. Put one to keep all frames.

# Get the list of folders in the rootpath
paths = [os.path.join(rootpath, f) for f in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, f))]
paths = [path for path in paths if "h"+hive in path]
# Order the paths alphabetically
paths.sort() # Now this contains the path to all RPis images

# Generate first and last picture name
rpis = [path.split("/")[-1][3] for path in paths]
first_pics = ["hive" + hive + "_rpi" + rpi + "_" + first_dt + ".jpg" for rpi in rpis]
last_pics = ["hive" + hive + "_rpi" + rpi + "_" + last_dt + ".jpg" for rpi in rpis]

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

dest = "outputVideos/"
# Make this a global path
generateVideoFromList(final_imgs, dest = "outputVideos/", name = "hive"+ hive + "_" + first_dt + "_" + last_dt, fps=10, grayscale=False)

# Cleanup step to release resources
multiprocessing.active_children()