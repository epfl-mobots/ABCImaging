''' Useful stuff to deal with images in the context of the 2020 winter season
    Contributors: Daniel H., Martin S., Rafael B., Rob M.
'''

import cv2 as cv
import numpy as np

def findVerticalBoardLimits(img):
    # Perform edge detection
    edges = cv.Canny(img, 150, 300)
    # Detect lines using Hough Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=2600, maxLineGap=1500)
    h_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 80] # Horizontal lines

    if len(h_lines) == 0:
        raise ValueError("No horizontal lines detected")

    lowest_h_line = h_lines[np.argmax([line[0][1] for line in h_lines])]

    bottom_border = np.mean([lowest_h_line[0][1], lowest_h_line[0][3]])
    top_border = bottom_border - 1900
    return int(top_border), int(bottom_border)

def findHorizontalBoardLimits(img):
    # Perform edge detection
    edges = cv.Canny(img, 150, 300)
    # Detect lines using Hough Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=1200, maxLineGap=600)
    v_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 80] # Vertical lines

    if len(v_lines) == 0:
        raise ValueError("No vertical lines detected")
    
    leftmost_v_line = v_lines[np.argmin([line[0][0] for line in v_lines])]
    rightmost_v_line = v_lines[np.argmax([line[0][0] for line in v_lines])]
    left_border = np.mean([leftmost_v_line[0][0], leftmost_v_line[0][2]])
    right_border = np.mean([rightmost_v_line[0][0], rightmost_v_line[0][2]])
    return int(left_border), int(right_border)


def _undistort(img):
    """Remove distortions from the image using calibration data.
    From
    * https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    """
    # Shape of the image
    h, w = img.shape[:2]

    # Matrices are specific for the camera setup (use these for the 2020 season)
    _cam_matrix = np.array([
        [3.88774532e+03, 0.00000000e+00, 2.01016942e+03],
        [0.00000000e+00, 3.90460835e+03, 1.63295552e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    _distortion_coeffs = np.array([[
        -5.04380784e-01, 3.32158862e-01, -4.75798455e-03, -3.43884435e-04, -1.47282285e-01]])

    # Refine cam matrix and compute ROI
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(_cam_matrix, _distortion_coeffs, (w, h), 1, (w, h))

    # Undistort
    dst = cv.undistort(
        img,
        _cam_matrix,
        _distortion_coeffs,
        None,
        newcameramtx,
    )

    # Crop the image
    x, y, w, h = roi
    # Ensure even numbers of pixels in rows and cols
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1
    dst = dst[y: y + h, x: x + w]

    return dst


def _undistort2(img, rpi):
    """ Remove perspective distortion from Rpi4 (back)
        Ref:
        https://pysource.com/2018/02/14/perspective-transformation-opencv-3-4-with-python-3-tutorial-13/
    """

    if rpi == 'rpi2':
        # Points we want to transform
        pa1 = (78, 95)
        pa2 = (3382, 95)
        pa3 = (132, 1950)
        pa4 = (3263, 1950)

        # cv.circle(img, pa1, 5, (0, 0, 255), -1)
        # cv.circle(img, pa2, 5, (0, 0, 255), -1)
        # cv.circle(img, pa3, 5, (0, 0, 255), -1)
        # cv.circle(img, pa4, 5, (0, 0, 255), -1)

        # Target locations for the points to move to
        pb1 = (0, 0)
        pb2 = (3200, 0)
        pb3 = (0, 1950)
        pb4 = (3200, 1950)

        # cv.circle(img, pb1, 15, (0, 200, 255), -1)
        # cv.circle(img, pb2, 15, (0, 200, 255), -1)
        # cv.circle(img, pb3, 15, (0, 200, 255), -1)
        # cv.circle(img, pb4, 15, (0, 200, 255), -1)

        # result = img
        pts_src = np.float32([list(pa1), list(pa2), list(pa3), list(pa4)])
        pts_tgt = np.float32([list(pb1), list(pb2), list(pb3), list(pb4)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        result = cv.warpPerspective(img, matrix, (3200, img.shape[0]))

    elif rpi == 'rpi4':
        # Points we want to transform
        pa1 = (0, 220)
        pa2 = (3200, 250)
        pa3 = (32, 1900)
        pa4 = (3130, 1850)

        # cv.circle(img, pa1, 5, (0, 0, 255), -1)
        # cv.circle(img, pa2, 5, (0, 0, 255), -1)
        # cv.circle(img, pa3, 5, (0, 0, 255), -1)
        # cv.circle(img, pa4, 5, (0, 0, 255), -1)

        # Target locations for the points to move to
        pb1 = (0, 220)
        pb2 = (3200, 220)
        pb3 = (0, 1900)
        pb4 = (3200, 1900)

        # cv.circle(img, pb1, 15, (0, 200, 255), -1)
        # cv.circle(img, pb2, 15, (0, 200, 255), -1)
        # cv.circle(img, pb3, 15, (0, 200, 255), -1)
        # cv.circle(img, pb4, 15, (0, 200, 255), -1)

        pts_src = np.float32([list(pa1), list(pa2), list(pa3), list(pa4)])
        pts_tgt = np.float32([list(pb1), list(pb2), list(pb3), list(pb4)])

        matrix = cv.getPerspectiveTransform(pts_src, pts_tgt)
        result = cv.warpPerspective(img, matrix, (3200, img.shape[0]))

    return result


def _unsharp_mask(
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


def beautify_frame(img, rpi):
    """Undistort, sharpen, hist-equalize and label image."""

    img = _undistort(img)
    img = _undistort2(img, rpi)
    img = _unsharp_mask(img, amount=1.5)

    # Histogram equalization
    img = cv.equalizeHist(img)
    # img = cv.GaussianBlur(img, (3, 3), 0)

    return img

def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    # prev_image_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    # current_image_gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    prev_image_gray = prev_image
    current_image_gray = current_image

    # assert current_image.shape == old_shape
    # hsv = np.zeros_like(prev_image)
    # hsv[..., 1] = 255

    _flow = cv.calcOpticalFlowFarneback(prev=prev_image_gray,
                                     next=current_image_gray, flow=None,
                                     pyr_scale=0.5, # PyrScale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one. default 0.5.
                                     levels=3,      # Number of pyramid layers including the initial image. Levels=1 means that no extra layers are created and only the original images are used. default 5.
                                     winsize=15,    # Averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. default 13.
                                     iterations=3,  # Number of iterations the algorithm does at each pyramid level. default 10.
                                     poly_n=5,      # Size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field. Typically, PolyN is 5 or 7. default 5.
                                     poly_sigma=1.2,# Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. For PolyN=5, you can set PolySigma = 1.1. For PolyN=7, a good value would be PolySigma = 1.5. default 1.1.
                                     flags=0)


    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # return flow, cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    _of_mag, _of_ang = cv.cartToPolar(_flow[..., 0], _flow[..., 1], angleInDegrees=True)

    return _flow, _of_mag, _of_ang

def find_biggest_active_area( of_magintude ):
    '''
    Find the biggest active area of an optical flow scalar (magnitude) field.
    of_magnitude can be computed using <compute_dense_optical_flow()>
    '''
    # higher values select areas with higher activities
    # So, it is more selective (smaller contour)
    thresh_val = 7.5
    ret, thresh = cv.threshold(of_magintude, thresh_val, 255, cv.THRESH_BINARY)

    ## Morphological opening and closing to improve mask
    # kernel1 = np.ones((5,5), np.uint8)
    # kernel2 = np.ones((100,100), np.uint8)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (140, 140))

    mask_morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1)
    mask_morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel2)

    ## Find contours
    # Convert binary image from float to int
    mask_morph = cv.convertScaleAbs(mask_morph)

    contours, hierarchy = cv.findContours(mask_morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours((contours, hierarchy))

    ## find biggest contour (ie, biggest active area)
    contour_areas = np.array([cv.contourArea(c) for c in contours])
    biggest_contour_idx = np.argmax(contour_areas)
    biggest_contour = contours[biggest_contour_idx]

    # find centroid
    mom = cv.moments(biggest_contour)
    cx = int(mom['m10'] / mom['m00'])
    cy = int(mom['m01'] / mom['m00'])

    # mask_cnt = np.zeros(mask_morph.shape, np.uint8)
    # cv.drawContours(mask_cnt, [biggest_contour], 0, 127, cv.LINE_4)
    # cv.drawContours(img_to_draw_contour, [biggest_contour], 0, 127, cv.LINE_4)

    # pixel_points = np.transpose(np.nonzero(mask_cnt))
    # pixel_points = cv2.findNonZero(mask)

    # return the centroid of the cluster (cx, cy)
    # and an image (np.array) with the contour
    # and the biggest area
    return (cx, cy), biggest_contour, contour_areas[biggest_contour_idx]

def plot_of_quiver(ax, flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.
    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    Ref: https://stackoverflow.com/questions/61943240/quiver-plot-with-optical-flow
    """
    h, w, *_ = flow.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    _flow = flow[np.ix_(y, x)]
    u = _flow[:, :, 0]
    v = _flow[:, :, 1]

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}

    xx, yy = np.meshgrid(x, y)
    M = np.sqrt(u*u + v*v) #magnitude vector for colors
    ax.quiver(xx, yy, u, v, M, cmap='RdYlBu_r', **kwargs)
    # ax.quiver(x, y, u, v, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")

def load_bg_img( bg_dir, scale_factor=1.0 ):
    _bg = cv.imread(str(bg_dir) + 'final_background_rpi2.jpg')
    _bg = cv.resize(_bg, (0, 0), fx=scale_factor, fy=scale_factor)
    _bg = cv.GaussianBlur(_bg, (55, 55), 0)
    _bg = cv.equalizeHist(cv.cvtColor(_bg, cv.COLOR_BGR2GRAY))

    return _bg

def find_cluster_contour( cluster_img, bg_img ):
    '''Method developed by Martin S.'''
    # use load_bg_img()
    bg = bg_img

    _img = cv.GaussianBlur(cluster_img, (55, 55), 0)
    _img = cv.equalizeHist(_img)

    thresh = np.zeros_like(_img)

    threshold1 = 180
    threshold2 = 70

    img_thresh1 = np.full_like(_img, 255)
    img_thresh2 = np.full_like(_img, 255)
    img_thresh3 = np.full_like(_img, 255)

    img_thresh1[(bg > threshold1)] = _img[(bg > threshold1)]
    img_thresh2[(bg > threshold2) & (bg < threshold1)] = _img[(bg > threshold2) & (bg < threshold1)]
    img_thresh3[(bg < threshold2)] = _img[(bg < threshold2)]

    ret, thresh1 = cv.threshold(img_thresh1, 175, 255, 0)
    ret, thresh2 = cv.threshold(img_thresh2, 140, 255, 0)
    ret, thresh3 = cv.threshold(img_thresh3, 80, 255, 0)

    thresh[(bg > threshold1)] = thresh1[(bg > threshold1)]
    thresh[(bg > threshold2) & (bg < threshold1)] = thresh2[(bg > threshold2) & (bg < threshold1)]
    thresh[(bg < threshold2)] = thresh3[(bg < threshold2)]

    # ret, thresh = cv2.threshold(img, 115, 255, 0)
    #thresh = cv2.GaussianBlur(thresh,(35,35),0)

    # Morphological opening and closing to improve mask
    mask_morph = cv.morphologyEx(
            thresh,
            cv.MORPH_OPEN,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (16, 16))
    )

    mask_morph = cv.morphologyEx(
            mask_morph,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (155, 155))
    )

    mask_morph[0, :] = 255
    mask_morph[:, 0] = 255
    mask_morph[-1, :] = 255
    mask_morph[:, -1] = 255

    ## Find contours
    _contours, hierarchy = cv.findContours(mask_morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ## find biggest contour
    _contour_areas = np.array([cv.contourArea(c) for c in _contours[1:]])
    _biggest_contour_idx = np.argmax(_contour_areas) + 1
    _biggest_contour = _contours[_biggest_contour_idx]

    # find centroid
    mom = cv.moments(_biggest_contour)
    cx = int(mom['m10'] / mom['m00'])
    cy = int(mom['m01'] / mom['m00'])

    # some_img = cv.drawContours(cluster_img, contours, -1, 255)
    _some_img = mask_morph

    return (cx, cy), _biggest_contour, _contour_areas[_biggest_contour_idx-1], _some_img

def px_to_mm(_img, _val, _rpi, _axis):
    ''' This function is tuned to undistorted images with the specific script (???.py)
    ex: cenX = px_to_mm(img2, cX, rpi, _axis='x')
        cenY = px_to_mm(img2, cY, rpi, _axis='y')
    '''
    px_min = None
    px_max = None
    py_min = None
    py_max = None

    w = _img.shape[1]
    h = _img.shape[0]

    if _rpi == 'rpi2':
        # Pixel number that corresponds to board x=0 mm
        px_min = int(w * 0.0)
        # Pixel number that corresponds to board x=410 mm
        px_max = int(w * 1)
        # Pixel number that corresponds to board y=0 mm
        py_min = int(h * 0.79) # 79% of the height
        # Pixel number that corresponds to board y=180 mm
        py_max = int(h * 0.18) # 22% of the height

    elif _rpi == 'rpi4':
        px_min = int(w * 1)
        px_max = 0 #int(w * 0.0)
        py_min = int(h * 0.77)
        py_max = int(h * 0.18)

    if _axis == 'x':
        x_mm = np.round((_val - px_min) / (px_max - px_min) * (410 - 0) + 0, 1)
        x_norm = np.round((_val - px_min) / (px_max - px_min) * (1 - 0) + 0, 3) # Normalized
        return (x_mm, x_norm)

    elif _axis == 'y':
        # y = (px - 0) / (410 - 0) * (px_max-px_min) + px_min
        y_mm = np.round((_val - py_min) / (py_max - py_min) * (180 - 0) + 0, 1)
        y_norm = np.round((_val - py_min) / (py_max - py_min) * (1 - 0) + 0, 3)
        return (y_mm, y_norm)

    elif _axis == 'area':
        dx = np.abs(px_max - px_min)
        dy = np.abs(py_min - py_max)
        tot_px = dx * dy
        tot_area = 410 * 180 # mm x mm
        area_per_px = tot_area / tot_px # [mm^2/px]
        area = _val * area_per_px # [mm^2]
        return area


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt

    rpi = None
    img_sf = 0.6  # resize by 60% (faster than 100%)

    ## Test image
    flist = glob.glob("./data/images/optical_flow/rpi2/day-201207/*.jpg")
    f_img = flist[0]

    # Detect which RPi the file belongs to
    if "rpi2" in f_img:
        rpi = 'rpi2'
    elif "rpi4" in f_img:
        rpi = 'rpi4'

    bg_img = load_bg_img('./outputs/2_zigzag/background/%s/' % rpi)

    img = cv.imread(f_img, cv.IMREAD_GRAYSCALE)
    img_beauty = beautify_frame(img, rpi)
    img_small = cv.resize(img_beauty, (0, 0), fx=img_sf, fy=img_sf)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 10))
    ax0.imshow(bg_img, cmap='gray')
    ax1.imshow(img, cmap='gray')
    ax2.imshow(img_beauty, cmap='gray')
    # ax1.set_title("Sequence image sample")

    plt.show()