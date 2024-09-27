# This script plays a video from a sequence of pictures in a new window.

import cv2
import os

# Path to the folder containing the pictures
rootpath = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/24.09_observation_OH/Images/'
first_dt = "240913-120000Z"
last_dt = "240913-180100Z"

# Get the list of folders in the rootpath
paths = [os.path.join(rootpath, f) for f in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, f))]
# Order the paths alphabetically
paths.sort()

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

paused = False
for i in range(len(files[0])):
    imgs = [cv2.imread(os.path.join(paths[j], files[j][i])) for j in range(len(files))]
    # put the texts on each image
    for j in range(len(imgs)):
        cv2.putText(imgs[j], files[j][i], (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Concatenate the images horizontally
    img_top = cv2.hconcat([imgs[2], imgs[0]]) # Frame 3 and 1 on top
    img_bottom = cv2.hconcat([imgs[3], imgs[1]]) # Frame 4 and 2 on bottom
    img = cv2.vconcat([img_top, img_bottom])
    # Show image and its name in the top right corner
    cv2.imshow('image', img)
    
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

    if paused:
        # Allow to skip a single frame
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('j'), ord('p')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('j'):
            continue
        elif key == ord('p'):
            paused = not paused

cv2.destroyAllWindows()