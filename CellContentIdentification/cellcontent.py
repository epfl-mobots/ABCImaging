import cv2 as cv
import numpy as np

def thresholding(img, threshold):
    '''
    Applies a binary threshold to the image.
    Parameters:
    img: The input image as np.uint8.
    threshold: The threshold value (0-255).
    Returns:
    The thresholded image as a binary image.
    '''
    # Apply a threshold to the image
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img

def morph(img, kernel_size = 7, close_first=True,iterations=2):
    # Apply a morphological transformation to the image to remove small islands of noise
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    if close_first:
        # Apply a closing operation to fill small holes
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations)
    if not close_first:
        # Apply a closing operation to fill small holes
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return img

def remove_small_patches(img, min_size):
    # Find the connected components in the image
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
    # Remove the small patches
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img