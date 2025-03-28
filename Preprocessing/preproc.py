'''Based on Daniel's code'''
import cv2 as cv
import numpy as np

def unsharp_mask(
        image,
        kernel_size=(5, 5),
        sigma=1.0,
        amount=1.0,
        threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.

    https://en.wikipedia.org/wiki/Unsharp_masking
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm"""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        # OpenCV4 function copyTo
        np.copyTo(sharpened, image, where=low_contrast_mask)
    return sharpened

def beautify_frame(img):
    """Undistort, sharpen, hist-equalize and label image."""
    img = unsharp_mask(img, amount=1.5)

    # Histogram equalization
    img = cv.equalizeHist(img)
    #img = cv.GaussianBlur(img, (3, 3), 0)

    return img

