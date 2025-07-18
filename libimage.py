'''
A general library used in all imaging scripts.

Author: Cyril Monette
Initial date: 18/07/2025
'''


import pandas as pd
import os
from tqdm import tqdm
from dask import delayed, compute

@delayed
def _fetch_single_datetime(dt, paths, hive_nb):
    dt_result = {}
    for path in paths:
        rpi_name = os.path.basename(path)[:4]
        filename = f"hive{hive_nb}_rpi{path.split('/')[-1][3]}_{dt.strftime('%y%m%d-%H%M')}"
        files = os.listdir(path)
        img_path = next((os.path.join(path, f) for f in files if filename in f), None)
        dt_result[rpi_name] = img_path if img_path else None
    return dt, dt_result


def fetchImagesPaths_dask(rootpath_imgs:str, datetimes:list, hive_nb:str, images_fill_limit=30, rpis:list[int]=[1,2,3,4]):
    '''
    Fetches the images' paths for a specific hive at specific datetimes using Dask for parallel processing.
    Parameters:
    - rootpath_imgs: str, root path to the images
    - datetimes: list of pd.DatetimeIndex, datetimes for which we want the images. Precision at minute level.
    - hive: int, hive number
    - images_fill_limit: int, maximum number of images to fill the gaps with the previous images. Default is 30 (5 hours at 1 img/min).
    Returns:
    - imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi.
    '''
    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [p for p in paths if f"h{hive_nb}" in p and int(os.path.basename(p)[3]) in rpis]
    paths.sort()

    columns = [os.path.basename(p)[:4] for p in paths]

    # Delayed processing
    delayed_results = [_fetch_single_datetime(dt, paths, hive_nb) for dt in datetimes]
    results = compute(*delayed_results)

    # Build final DataFrame
    imgs_paths = pd.DataFrame(index=datetimes, columns=columns)
    for dt, dt_result in results:
        for rpi in columns:
            imgs_paths.loc[dt, rpi] = dt_result.get(rpi)

    print(f"Missing images before filtering: {imgs_paths.isnull().sum().sum()} out of {imgs_paths.size}")

    imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit, axis=0) if images_fill_limit > 0 else imgs_paths

    if imgs_paths_filtered.isnull().sum().sum() > 0:
        raise ValueError(f"Still missing images despite filling gaps up to {images_fill_limit} images.")

    return imgs_paths_filtered 

def fetchImagesPaths(rootpath_imgs:str, datetimes:list, hive_nb:str, images_fill_limit = 30, rpis:list[int]=[1,2,3,4]):
    '''
    Fetches the images' paths for a specific hive at specific datetimes.
    Parameters:
    - rootpath_imgs: str, root path to the images
    - datetimes: list of pd.DatetimeIndex, datetimes for which we want the images. Precision at minute level.
    - hive: int, hive number
    - images_fill_limit: int, maximum number of images to fill the gaps with the previous images. Default is 30 (5 hours at 1 img/min).
    Returns:
    - imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi.
    '''

    # Get the list of folders in the rootpath
    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [path for path in paths if "h"+hive_nb in path and int(os.path.basename(path)[3]) in rpis] # Only keep the paths that contain the hive_nb and are in rpis
    # Order the paths alphabetically:
    paths.sort() # Now this contains the path to all RPis images included in rpis (arg)

    # For each dt in datetimes, find the image path that == dt for each RPi. Put the paths in a df where each row is a dt and each column is a RPi
    imgs_paths = pd.DataFrame(index=datetimes, columns=[os.path.basename(path)[:4] for path in paths])
    for dt in tqdm(datetimes, desc="Fetching images"):
        for path in paths:
            filename = "hive"+hive_nb+"_rpi"+path.split("/")[-1][3]+"_"+dt.strftime('%y%m%d-%H%M')
            # Find the file in os.listdir(path) that contains the dt (or startswith(dt))
            img_path = [os.path.join(path, f) for f in os.listdir(path) if filename in f]
            if len(img_path) == 1:
                imgs_paths.loc[dt, os.path.basename(path)[:4]] = img_path[0]
            else:
                imgs_paths.loc[dt, os.path.basename(path)[:4]] = None

    # Check how many images are missing
    print("Missing images before filtering: ", imgs_paths.isnull().sum().sum(), "out of", imgs_paths.shape[0]*imgs_paths.shape[1], "images.")

    # Fill the gaps with the images from the previous dt
    imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit, axis=0) if images_fill_limit > 0 else imgs_paths

    # Check if there are still missing images, if so, raise an error
    if imgs_paths_filtered.isnull().sum().sum() > 0:
        raise ValueError(f"Still missing images, desipite filling the gaps with the previous images up to {images_fill_limit} images.")
    
    return imgs_paths_filtered