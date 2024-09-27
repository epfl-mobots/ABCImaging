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

pos = 'back'  # 'back' or 'front'

p_back = '/Users/barmak/Desktop/sample_h4r4_200729--201009/imgs/imgs_h4r4/'
p_front = '/Users/barmak/Desktop/sample_h4r2_200729--201009/imgs/imgs_h4r2/'

p_out = '/Users/barmak/Desktop/sample_median/%s/' % pos

p = p_front if pos == 'front' else p_back
files = sorted(glob.glob(p + '*.jpg'))


n = 100  # median of 100 images = 100 * 15 min = 1500 min = 25 hours
image_interval = 12  # Use one image every = 12 * 15 min = 3 hours

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
    cv.imwrite(p_out + 'median_%04d_%s_%s.jpg' % (i, pos, files[i].split('/')[-1][10:28]), median_image)

print('DONE!')
