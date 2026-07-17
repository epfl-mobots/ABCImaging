'''
A general library used in all imaging scripts.

Author: Cyril Monette
Initial date: 18/07/2025
'''

import pandas as pd
import os, bisect
from tqdm import tqdm
from HiveOpenings.libOpenings import * # To filter out invalid datetimes
from typing import Dict, List, Tuple
import numpy as np

RPiCamV3_img_shape = (2592, 4608)   # Height, Width
RPiCamV3_img_shape_RGB = (2592, 4608, 3)   # Height, Width, Channels

# Full paths of folders where near-duplicate images were pruned down to ~1 image/hour (since
# consecutive images looked the same, e.g. no bees visible). For these folders, an exact per-minute
# filename match can't be expected, so we look for the nearest available image within
# SPARSE_MAX_TIME_DIFF minutes.
SPARSE_IMAGE_FOLDERS = {
    "/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.11-25.01_metabolism_OH/Images/h2r1_1minute",
    "/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.11-25.01_metabolism_OH/Images/h2r2_1minute",
    "/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.11-25.01_metabolism_OH/Images/h2r4_1minute"
}
SPARSE_MAX_TIME_DIFF = 60  # minutes

def estimate_affine_exposure_correction(ref_img, target_img, n_iter:int=3, keep_frac:float=0.7, max_samples:int=200_000, rng_seed:int=0):
    '''
    Robustly estimate a photometric gain/bias transform (ref ~= gain*target + bias) between two
    grayscale images, using iteratively reweighted least squares.

    Exposure changes are closer to a multiplicative gain on pixel intensity than a simple additive
    offset, so fitting both a gain and a bias corrects contrast differences that a mean-shift alone
    would miss. At each iteration, the pixels with the largest residuals (i.e. the ones most likely to
    correspond to real activity/movement rather than exposure differences) are excluded from the fit,
    so the estimated transform reflects the *background* exposure relationship rather than being biased
    by genuine activity.

    :param ref_img: reference image (e.g. img_slice1)
    :param target_img: image to be corrected to match ref_img's exposure (e.g. img_slice2)
    :param n_iter: number of reweighting iterations
    :param keep_frac: fraction of (lowest-residual) pixels kept at each reweighting step
    :param max_samples: max number of pixels used for the fit (subsampled for speed on large images)
    :param rng_seed: seed for the subsampling RNG, for reproducibility
    :return: (gain, bias) such that ref_img ~= gain * target_img + bias
    '''
    ref = ref_img.astype(np.float64).ravel()
    target = target_img.astype(np.float64).ravel()

    n = ref.size
    if n > max_samples:
        idx = np.random.default_rng(rng_seed).choice(n, size=max_samples, replace=False)
        ref, target = ref[idx], target[idx]

    gain, bias = 1.0, 0.0
    mask = np.ones(ref.shape, dtype=bool)
    for _ in range(n_iter):
        if mask.sum() < 10:
            break
        A = np.vstack([target[mask], np.ones(mask.sum())]).T
        (gain, bias), *_ = np.linalg.lstsq(A, ref[mask], rcond=None)
        residuals = np.abs(ref - (gain * target + bias))
        mask = residuals <= np.quantile(residuals, keep_frac)

    return gain, bias

def _build_file_ts_index(files:List[str], prefix:str)->Tuple[List[pd.Timestamp], List[str]]:
    '''
    Parses the timestamp encoded in each filename (that starts with prefix) and returns them
    sorted chronologically alongside their filenames, to allow efficient nearest-timestamp lookup.

    :param files: list of str, filenames in a directory.
    :param prefix: str, prefix that the relevant filenames start with (e.g. "hive2_rpi4_").
    :return: tuple of (sorted list of pd.Timestamp, corresponding list of filenames).
    '''
    entries = []
    for f in files:
        if not f.startswith(prefix):
            continue
        ts_part = f[len(prefix):].split('.')[0].rstrip('Z')
        try:
            file_dt = pd.to_datetime(ts_part, format='%y%m%d-%H%M%S').tz_localize('UTC')
        except ValueError:
            continue
        entries.append((file_dt, f))
    entries.sort(key=lambda e: e[0])
    timestamps = [e[0] for e in entries]
    filenames = [e[1] for e in entries]
    return timestamps, filenames

def _find_nearest_file(dt:pd.Timestamp, timestamps:List[pd.Timestamp], filenames:List[str], max_time_diff:int)->str:
    '''
    Finds the filename whose timestamp is closest to dt, within max_time_diff minutes, using binary search.

    :param dt: pd.Timestamp, tz-aware target datetime.
    :param timestamps: sorted list of pd.Timestamp (as returned by _build_file_ts_index).
    :param filenames: list of filenames corresponding to timestamps.
    :param max_time_diff: int, maximum time difference in minutes to accept a match.
    :return: the closest filename, or None if no file is within max_time_diff.
    '''
    if not timestamps:
        return None
    idx = bisect.bisect_left(timestamps, dt)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(timestamps)]
    best_file = None
    best_delta = None
    for i in candidates:
        delta = abs((timestamps[i] - dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_file = filenames[i]
    if best_delta is not None and best_delta <= max_time_diff * 60:
        return best_file
    return None

def _fetch_single_datetime(dt:pd.Timestamp, file_cache:Dict, paths:List[str], hive_nb:int, exact_image:bool=True,
                           ts_index_cache:Dict=None, sparse_paths:set=None, sparse_max_time_diff:int=SPARSE_MAX_TIME_DIFF,
                           rounded_max_time_diff:int=15):
    '''
    Fetches the image path for a specific datetime, for each of the given RPi paths.

    :param dt: pd.Timestamp, datetime for which we want the image. Needs to be tz-aware.
    :param file_cache: dict mapping path -> list of files in that directory.
    :param paths: list of str, list of paths to search for the images.
    :param hive_nb: int, hive number (e.g., 1, 2, etc.)
    :param exact_image: bool, if True, requires an exact (minute-level) filename match for paths that
        aren't in sparse_paths. If False, uses the nearest match within rounded_max_time_diff minutes instead.
    :param ts_index_cache: dict mapping path -> (sorted timestamps, filenames), as built by _build_file_ts_index.
        Required for paths in sparse_paths, or for all paths if exact_image is False.
    :param sparse_paths: set of str (full directory paths) where images were pruned to ~1/hour; these always
        use a nearest match within sparse_max_time_diff minutes, regardless of exact_image.
    :param sparse_max_time_diff: int, maximum time difference in minutes to use for sparse_paths.
    :param rounded_max_time_diff: int, maximum time difference in minutes to use when exact_image is False
        for paths not in sparse_paths.
    :return: dt, dict mapping rpi_name -> image path (or None if not found).
    '''
    dt = dt.tz_convert('UTC')  # Ensure the datetime is in UTC. Will fail if not tz-aware.
    sparse_paths = sparse_paths or set()
    ts_index_cache = ts_index_cache or {}
    dt_result = {}
    for path in paths:
        rpi_name = os.path.basename(path)[:4]
        is_sparse = os.path.normpath(path) in sparse_paths
        if not is_sparse and exact_image:
            rpi_num = path.split('/')[-1][3]
            filename = f"hive{hive_nb}_rpi{rpi_num}_{dt.strftime('%y%m%d-%H%M')}"
            files = file_cache[path]
            img_path = next((os.path.join(path, f) for f in files if filename in f), None)
        else:
            max_time_diff = sparse_max_time_diff if is_sparse else rounded_max_time_diff
            timestamps, filenames = ts_index_cache[path]
            best_file = _find_nearest_file(dt, timestamps, filenames, max_time_diff)
            img_path = os.path.join(path, best_file) if best_file is not None else None
        dt_result[rpi_name] = img_path
    return dt, dt_result


def fetchImagesPaths(rootpath_imgs:str, datetimes:List[pd.Timestamp], hive_nb:int, invalid_recovery_time:int = None, images_fill_limit:int = None, rpis:List[int]=[1,2,3,4], exact_image:bool=True, verbose=False):
    '''
    Fetches the images' paths for a specific hive at specific datetimes using Dask for parallel processing.

    :param rootpath_imgs: str, root path to the images
    :param datetimes: list of pd.Timestamps, datetimes for which we want the images. Precision at minute level. Needs to be tz-aware.
    :param hive_nb: int, hive number (e.g., 1, 2, etc.)
    :param invalid_recovery_time: int, if specified, will filter out invalid datetimes including the given recovery time in minutes (when the hives were being opened + recovery time [min]).
    :param images_fill_limit: int, if provided, maximum number of images to fill the gaps with the previous images. If not provided, will not fill gaps (None in df).
    :param rpis: list of int, list of RPi numbers to consider. Default is [1,2,3,4].
    :param exact_image: bool, if True, will look for an exact match of the datetime for regular folders. If False,
        will look for the nearest image instead. Folders listed in SPARSE_IMAGE_FOLDERS always use the nearest
        image (within SPARSE_MAX_TIME_DIFF minutes), regardless of this parameter.
    :return imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi. If validity is checked, the last column will indicate whether the datetime is valid or not (bool).
    '''

    if not all(dt.tzinfo is not None for dt in datetimes):
        raise ValueError("All datetimes must be tz-aware.")

    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [p for p in paths if f"h{hive_nb}" in p and int(os.path.basename(p)[3]) in rpis]
    paths.sort()
    if verbose:
        print(f"Using image paths: {paths}")

    columns = [os.path.basename(p)[:4] for p in paths]

    if invalid_recovery_time is not None:
        # Filter out datetimes that are not valid (i.e., when the hives were being opened)
        valid_datetimes = filter_timestamps(datetimes, hive_nb, invalid_recovery_time)

    validity = [dt in valid_datetimes for dt in datetimes] if invalid_recovery_time is not None else None

    if verbose:
        print(f"Datetimes: {datetimes}")
        if invalid_recovery_time is not None:
            print(f"Valid datetimes: {valid_datetimes}")

    # Build file cache once instead of calling os.listdir() for each datetime
    file_cache = {path: os.listdir(path) for path in paths}

    # Normalize the sparse folder paths once so they can be compared reliably against the
    # (already absolute) paths built above, regardless of trailing slashes, etc.
    sparse_paths = {os.path.normpath(p) for p in SPARSE_IMAGE_FOLDERS}

    # Build a sorted timestamp index once per path that needs nearest-match lookup (sparse folders
    # always need it, other folders only need it if exact_image is False), instead of re-parsing
    # every file's timestamp for every datetime.
    ts_index_cache = {}
    for path in paths:
        if os.path.normpath(path) in sparse_paths or not exact_image:
            rpi_num = path.split('/')[-1][3]
            prefix = f"hive{hive_nb}_rpi{rpi_num}_"
            ts_index_cache[path] = _build_file_ts_index(file_cache[path], prefix)

    # Direct iteration (no dask overhead)
    results = [
        _fetch_single_datetime(dt, file_cache, paths, hive_nb, exact_image=exact_image,
                                ts_index_cache=ts_index_cache, sparse_paths=sparse_paths)
        for dt in tqdm(datetimes, desc="Fetching image paths")
    ]

    # Build final DataFrame
    imgs_paths = pd.DataFrame(index=datetimes, columns=columns)
    
    for dt, dt_result in results:
        for rpi in columns:
            imgs_paths.loc[dt, rpi] = dt_result[rpi]
    
    if verbose:
        print("Non-null counts per column:")
        print(imgs_paths.notna().sum())
        print("Total non-nulls:", imgs_paths.notna().sum().sum())

    if imgs_paths.isna().all().all():
        raise ValueError("No images found for the given datetimes and hive number. " \
                         "There might be a timestamp mismatch.")
    
    if validity is not None:
        imgs_paths['valid'] = validity # Add a column for validity if it is checked

    if images_fill_limit is not None and images_fill_limit > 0:
        valid_imgs = imgs_paths[imgs_paths['valid'] == True].drop(columns=['valid']) if 'valid' in imgs_paths.columns else imgs_paths
        print(f"Missing images before filtering: {valid_imgs.isnull().sum().sum()} out of {valid_imgs.shape[0] * valid_imgs.shape[1]}")

        imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit, axis=0) if images_fill_limit > 0 else imgs_paths

        valid_imgs_filtered = imgs_paths_filtered[imgs_paths_filtered['valid'] == True].drop(columns=['valid']) if 'valid' in imgs_paths_filtered.columns else imgs_paths_filtered
        missing_after = valid_imgs_filtered.isnull().sum().sum()
        if missing_after > 0:
            print(f"[W]: There are still {missing_after} missing images after filling with limit {images_fill_limit}.")
        
        return imgs_paths_filtered

    return imgs_paths