import cv2 as cv
import numpy as np
import dask.array as da
from dask import delayed
from skimage.io import imread
import timeit

# Define a function to read an image
@delayed
def read_image(path):
    return imread(path)


def median_filter(image_list,paths = False):
    '''
    This function takes a list of images and returns the median image.
    params:
    image_list: list of images (images or paths)
    paths: if True, image_list is a list of paths, else a list of images directly
    '''
    start_time = timeit.default_timer()
    if paths:
        delayed_images = [da.from_delayed(read_image(path), shape=(cv.imread(image_list[0]).shape), dtype=np.uint8) for path in image_list]
    else:
        delayed_images = image_list

    # Stack all the images into a Dask array
    images = da.stack(delayed_images, axis=0)

    # Calculate the median along the new dimension
    median_image = np.median(images, axis=0)

    median_image = median_image.compute()
    # convert to numpy array so it can be displayed by CV2
    median_image = np.array(median_image, dtype=np.uint8)   

    end_time = timeit.default_timer()


    return median_image