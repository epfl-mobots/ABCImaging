# This script generates a video from a sequence of pictures.
import cv2
import os
from tqdm import tqdm
from videolib import initVideoWriter, imageHiveOverview
import multiprocessing
import pandas as pd

# ====================================== CONGIGURATION ======================================
# Path to the folder containing the pictures
rootpath = '/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.09-24.10_observation_OH/Images/'
rootpath_img_period = 1 # Period between every image in the folder in minutes
hive = "2"

first_dt_str = "241011-150000Z"
last_dt_str = "241028-100000Z"

frame_drop = 20 # We keep 1 frame every frame_drop frames. Put one to keep all frames.
fps_video = 30 # Frames per second for the video

# ===================================== MAIN SCRIPT =====================================


if __name__ == "__main__":
    multiprocessing.set_start_method("fork") # Use fork to avoid issues with cv2 and multiprocessing

    first_dt = pd.to_datetime(first_dt_str, format="%y%m%d-%H%M%SZ").tz_localize('UTC')
    last_dt = pd.to_datetime(last_dt_str, format="%y%m%d-%H%M%SZ").tz_localize('UTC')
    print("First date: ", first_dt)
    print("Last date: ", last_dt)

    time_range = pd.date_range(start=first_dt, end=last_dt, freq=f"{rootpath_img_period*frame_drop}min")
    time_range_str = [dt.strftime("%y%m%d-%H%M%SZ") for dt in time_range]
    # Remove seconds from the time range strings
    time_range_str = [ts[:-3] for ts in time_range_str]

    # Get the list of folders in the rootpath
    paths = [os.path.join(rootpath, f) for f in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, f))]
    paths = [path for path in paths if "h"+hive in path]
    # Order the paths alphabetically
    paths.sort() # Now this contains the path to all RPis images

    # Generate first and last picture name
    rpis = [path.split("/")[-1][3] for path in paths]
    first_pics = ["hive" + hive + "_rpi" + rpi + "_" + first_dt_str + ".jpg" for rpi in rpis]
    last_pics = ["hive" + hive + "_rpi" + rpi + "_" + last_dt_str + ".jpg" for rpi in rpis]

    # Get the list of files in the folders
    files = [os.listdir(path) for path in paths]
    # Sort the files by name
    for i in range(len(files)):
        files[i].sort()
    
    print(len(files[0]),len(files[1]),len(files[2]),len(files[3]))

    example_frame = imageHiveOverview([cv2.imread(os.path.join(paths[j], files[j][0])) for j in range(len(files))],
                                      ["hive" + hive + "_rpi" + str(j+1) + "_" + files[j][0] for j in range(len(files))])
    video = initVideoWriter(dest="outputVideos/", frame=example_frame,name="hive" + hive + "_" + first_dt_str + "_" + last_dt_str,
                            fps=fps_video, grayscale=False)
    print("Writing video...")
    for i,ts in enumerate(tqdm(time_range_str)):
        imgs = []
        names = []
        for j in range(len(files)):
            # Check if the file exists
            if any(ts in f for f in files[j]):
                # Find the file that has ts in its name
                file = [f for f in files[j] if ts in f][0]
                names.append(file)
                # Read the image
                imgs.append(cv2.imread(os.path.join(paths[j], file)))
            else:
                # If the file does not exist, append a black image
                names.append("hive" + hive + "_rpi" + str(j+1) + "_" + ts + ".jpg")
                imgs.append(cv2.imread(os.path.join(paths[j], files[j][0])) * 0)
        #imgs = [cv2.imread(os.path.join(paths[j], files[j][i])) for j in range(len(files))]

        assembled_img = imageHiveOverview(imgs, names)
        # Add the image to the list of images
        video.write(assembled_img)

    # Release the VideoWriter object
    video.release()
    print("Video written successfully.")