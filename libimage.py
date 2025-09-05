'''
A general library used in all imaging scripts.

Author: Cyril Monette
Initial date: 18/07/2025
'''

import pandas as pd
import os, sys
from tqdm import tqdm
from dask import delayed, compute
from HiveOpenings.libOpenings import * # To filter out invalid datetimes

@delayed
def _fetch_single_datetime(dt:pd.Timestamp, paths, hive_nb:str):
    dt = dt.tz_convert('UTC')  # Ensure the datetime is in UTC. Will fail if not tz-aware.
    dt_result = {}
    for path in paths:
        rpi_name = os.path.basename(path)[:4]
        filename = f"hive{hive_nb}_rpi{path.split('/')[-1][3]}_{dt.strftime('%y%m%d-%H%M')}"
        files = os.listdir(path)
        img_path = next((os.path.join(path, f) for f in files if filename in f), None)
        dt_result[rpi_name] = img_path if img_path else None
    return dt, dt_result


def fetchImagesPaths(rootpath_imgs:str, datetimes:list[pd.Timestamp], hive_nb:str, invalid_recovery_time:int = None, images_fill_limit:int = None, rpis:list[int]=[1,2,3,4], verbose=False):
    '''
    Fetches the images' paths for a specific hive at specific datetimes using Dask for parallel processing.
    Parameters:
    - rootpath_imgs: str, root path to the images
    - datetimes: list of pd.Timestamps, datetimes for which we want the images. Precision at minute level. Needs to be tz-aware.
    - hive_nb: str, hive number (e.g., "1", "2", etc.)
    - invalid_recovery_time: int, if specified, will filter out invalid datetimes including the given recovery time in minutes (when the hives were being opened + recovery time [min]).
    - images_fill_limit: int, if provided, maximum number of images to fill the gaps with the previous images. If not provided, will not fill gaps (None in df).
    - rpis: list of int, list of RPi numbers to consider. Default is [1,2,3,4].
    Returns:
    - imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi. If validity is checked, the last column will indicate whether the datetime is valid or not (bool).
    '''

    if not all(dt.tzinfo is not None for dt in datetimes):
        raise ValueError("All datetimes must be tz-aware.")

    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [p for p in paths if f"h{hive_nb}" in p and int(os.path.basename(p)[3]) in rpis]
    paths.sort()

    columns = [os.path.basename(p)[:4] for p in paths]

    if invalid_recovery_time is not None:
        # Filter out datetimes that are not valid (i.e., when the hives were being opened)
        valid_datetimes = filter_timestamps(datetimes, int(hive_nb), invalid_recovery_time)

    validity = [dt in valid_datetimes for dt in datetimes] if invalid_recovery_time is not None else None

    if verbose:
        print(f"Datetimes: {datetimes}")
        print(f"Valid datetimes: {valid_datetimes}")

    # Delayed processing
    delayed_results = [_fetch_single_datetime(dt, paths, hive_nb) for dt in datetimes]
    results = compute(*delayed_results)

    # Build final DataFrame
    imgs_paths = pd.DataFrame(index=datetimes, columns=columns)
    if validity is not None:
        imgs_paths['valid'] = validity # Add a column for validity if it is checked
    
    for dt, dt_result in results:
        for rpi in columns:
            imgs_paths.loc[dt, rpi] = dt_result[rpi]

    if images_fill_limit is not None and images_fill_limit > 0:
        print(f"Missing images before filtering: {imgs_paths.isnull().sum().sum()} out of {len(datetimes) * len(columns)}")

        imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit, axis=0) if images_fill_limit > 0 else imgs_paths

        if imgs_paths_filtered.isnull().sum().sum() > 0:
            raise ValueError(f"Still missing images despite filling gaps up to {images_fill_limit} images.")
        
        return imgs_paths_filtered

    return imgs_paths