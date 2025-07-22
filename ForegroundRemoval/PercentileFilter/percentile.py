# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, cv2, sys
import numpy as np
import pandas as pd
from math import ceil, floor
import dask_image.imread
from dask import delayed
import dask.array as da
from Preprocessing.preproc import beautify_frame


# Converts an image to grayscale
def convert_gray_old(bgr): # 20min33
    result = ((bgr[..., 0] * 0.114) +
              (bgr[..., 1] * 0.587) +
              (bgr[..., 2] * 0.299))
    return result

@delayed
def convert_gray(img): # 13min14
    np_img = np.array(img) # Convert from Dask array to numpy array
    return cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

# Function to apply filtering on a substack of images
@delayed
def percentile_custom(substack, percentile=75):
    # Return the percentile across the specified axis
    return np.percentile(substack, percentile, axis=0).astype(np.uint8)

@delayed
def annotate_name(img, filename):
    # Pre-calculate parameters for annotation
    font_scale = 2
    font_color = (255, 0, 0)
    thickness = 3
    position = (10, 70)  # Fixed position for text
    return cv2.putText(img, filename, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv2.LINE_AA)

@delayed
def beautify_frame_delayed(img):
    #img = np.array(img) # Convert from Dask array to numpy array
    return beautify_frame(img)

def __filter_substack(images,i,filter_length,percentile,frame_skip=1):
    # Get the substack
    substack = images[max(0,i-frame_skip*floor(filter_length/2)):i + frame_skip*ceil(filter_length/2):frame_skip]
    # Convert the substack to grayscale
    substack_gray = [convert_gray(img) for img in substack]
    # Preprocess the substack
    substack_gray = [beautify_frame_delayed(img) for img in substack_gray]
    # Calculate the percentile
    percentile_img = percentile_custom(substack_gray, percentile)
    return percentile_img

def __filter(images_folder,idxs:list,frame_skip=1,filter_length=40,percentile=75, annotate_names=False, verbose=False):
    if verbose:
        print("Indexes: ", idxs)
    all_files = os.path.join(images_folder, '*.jpg')
    images = dask_image.imread.imread(all_files)
    img_names = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    img_names.sort()
    img_names = [img_names[i] for i in idxs]
    if verbose:
        print("Dask images: ", images)

    # Prepare all filenames
    filenames = os.listdir(images_folder)
    filenames.sort()
    # Remove hidden files
    filenames = [f for f in filenames if not f.startswith('.')]

    # Read the first image to get the dimensions
    first_image = cv2.imread(os.path.join(images_folder, filenames[0]))
    height, width, _ = first_image.shape
    if verbose:
        print("Image dimensions: ", height, width)

    filtered_imgs = [__filter_substack(images, i,filter_length,percentile,frame_skip) for i in idxs]
    # Annotate all images with their name
    if annotate_names:
        filtered_imgs = [annotate_name(img, filenames[idx]) for idx, img in zip(idxs,filtered_imgs)]
    filtered_imgs = da.stack([da.from_delayed(d, shape=(height,width), dtype=np.uint8) for d in filtered_imgs], axis=0)
    return filtered_imgs, img_names


def percentile_filter_df(img_paths:pd.DataFrame,frame_skip=1,filter_length=40,percentile=75, annotate_names=False, verbose=False):
    '''
    This function makes a percentile filter of images with paths contained in a dataframe. It is preprocessing images.
    Returns a dataframe with the filtered images with the same structure as the input dataframe.
    '''
    filtered_imgs = pd.DataFrame(columns=img_paths.columns)
    imgs_names = pd.DataFrame(columns=img_paths.columns)
    # For names, just put the os.path.basename of img_paths
    for col in img_paths.columns:
        imgs_names[col] = img_paths[col].apply(lambda x: os.path.basename(x).split('.')[0])
        filtered_imgs[col] = img_paths[col].apply(lambda x: None)

    for col in img_paths.columns:
        first_path = img_paths[col].iloc[0]
        folder = os.path.dirname(first_path)
        files = sorted(os.listdir(folder))

        # Find the idx in files corresponding to all the images in img_paths[col]
        idxs = []
        for path in img_paths[col]:
            idxs.append(files.index(os.path.basename(path)))
        rpi_imgs, _ = __filter(folder, idxs, frame_skip=frame_skip, filter_length=filter_length, percentile=percentile, annotate_names=annotate_names, verbose=verbose)
        rpi_imgs = np.array(rpi_imgs) # This computes the result
        filtered_imgs[col] = list(rpi_imgs)

    return filtered_imgs, imgs_names


def percentile_filter(images_folder,start_idx, stop_idx=None,step=1,frame_skip=1,filter_length=40,percentile=75, annotate_names=False, verbose=False):
    '''
    This function makes a percentile filter of images between start and stop indexes.  It is preprocessing images.
    '''
    if stop_idx is None:
        stop_idx = start_idx + 1
    idxs = range(start_idx, stop_idx, step) # Images that need to be filtered
    return __filter(images_folder, idxs, frame_skip=frame_skip, filter_length=filter_length, percentile=percentile, annotate_names=annotate_names, verbose=verbose)

def percentile_filter_single(img_path,frame_skip=1,filter_length=40,percentile=75, annotate_names=False, verbose=False):
    '''
    This function makes a percentile filter of a single image rather than a substack. It is preprocessing images.
    '''
    parent_folder = os.path.dirname(img_path)
    image_file = os.path.basename(img_path)
    all_files = [f for f in os.listdir(parent_folder) if f.endswith('.jpg')]
    index = sorted(all_files).index(image_file)
    img, img_name = __filter(parent_folder, [index], frame_skip=frame_skip, filter_length=filter_length, percentile=percentile, annotate_names=annotate_names, verbose=verbose)
    img = img[0]
    img_name = img_name[0]

    return img, img_name