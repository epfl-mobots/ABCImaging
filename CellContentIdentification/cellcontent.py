import cv2 as cv
import numpy as np

def thresholding(img, threshold):
    # Apply a threshold to the image
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img

def morph(img, kernel_size = 7):
    # Apply a morphological transformation to the image to remove small islands of noise
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    return img

def remove_small_patches(img, min_size):
    # Find the connected components in the image
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
    # Remove the small patches
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img