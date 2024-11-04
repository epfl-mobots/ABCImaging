import glob
import cv2 as cv
import numpy as np
import dask.array as da
from dask import delayed
from skimage.io import imread
import timeit
import re
from datetime import datetime


# Define a function to read an image
@delayed
def read_image(path):
    return imread(path)


image_folder = "/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/24.09-24.10_observation_OH/Images/h1r1_1minute/"

p_out = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/Imaging/ForegroundRemoval/MedianFilter/Outputs/Cyril/'

files = sorted(glob.glob(image_folder + '*.jpg'))
# Only keep the first 1000 images
files = files[:1000]


n = 100  # median of 100 images = 100 * 1 min = 100 min = 1.6667 hours
image_interval = 12  # Use one image every = 12 * 1 min = 12 minutes

n_files = range(0, len(files)-n, image_interval)  # Number of generated images

for i in n_files:
    start_time = timeit.default_timer()

    # List of image paths
    image_paths = files[i:i+n]

    # Use Dask's delayed function to create a lazy evaluation of the function
    delayed_images = [da.from_delayed(read_image(path), shape=(cv.imread(image_paths[0]).shape), dtype=np.uint8) for path in image_paths]

    # Stack all the images into a Dask array
    images = da.stack(delayed_images, axis=0)
    first_image = np.array(images[0].compute(), dtype=np.uint8)

    # Calculate the median along the new dimension
    median_image = np.median(images, axis=0)

    # convert to numpy array so it can be displayed by CV2
    median_image = median_image.compute()
    median_image = np.array(median_image, dtype=np.uint8)

    end_time = timeit.default_timer()

    print(f"{i}: {round(end_time - start_time, 1)} seconds.")

    ## Ex: 'median_5568_front_rpi2_201008-200000.jpg'
    cv.imwrite(p_out + 'median_%04d_%s.jpg' % (i, files[i].split('/')[-1][10:28]), median_image)

print('DONE!')
