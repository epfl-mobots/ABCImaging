'''
Locates the camera images for a given hive and timestamp.

Reuses fetchImagesPaths from Imaging/libimage.py so the labeling tool stays
consistent with the rest of the repo (same folder layout, same nearest-match /
sparse-folder handling).

Author: Cyril Monette
'''

import os
import re
import sys

import pandas as pd

_IMAGING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IMAGING_ROOT not in sys.path:
    sys.path.insert(0, _IMAGING_ROOT)

from libimage import fetchImagesPaths  # noqa: E402

DEFAULT_IMAGES_ROOT = "/Users/cyrilmonette/Library/CloudStorage/SynologyDrive-data/24.11-25.01_metabolism_OH/Images"

_HIVE_FOLDER_RE = re.compile(r"^h(\d+)r(\d+)_")


def list_hives(images_root: str) -> list[int]:
    '''
    Scans images_root for subfolders named like "h<hive>r<rpi>_..." and returns
    the sorted list of hive numbers found.

    :param images_root: str, root folder containing the h{hive}r{rpi}_... subfolders.
    :return: sorted list of int hive numbers.
    '''
    if not os.path.isdir(images_root):
        return []
    hives = set()
    for entry in os.listdir(images_root):
        if not os.path.isdir(os.path.join(images_root, entry)):
            continue
        match = _HIVE_FOLDER_RE.match(entry)
        if match:
            hives.add(int(match.group(1)))
    return sorted(hives)


def find_camera_images(hive_nb: int, timestamp: pd.Timestamp, images_root: str = DEFAULT_IMAGES_ROOT) -> dict:
    '''
    Finds, for each of the hive's cameras, the image closest to the requested timestamp.

    Tries an exact (minute-level) match first (fast, no filename-timestamp indexing needed).
    If any camera has no exact match, falls back to a nearest-match search (slower, since it
    has to parse every filename's timestamp) for all cameras of that hive.

    :param hive_nb: int, hive number (e.g. 1, 2).
    :param timestamp: pd.Timestamp, the target datetime. If not tz-aware, assumed UTC.
    :param images_root: str, root folder containing the h{hive}r{rpi}_... subfolders.
    :return: dict mapping camera name (e.g. "h1r1") -> image path (str), or None if nothing
        was found for that camera within the matching tolerance.
    '''
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    df = fetchImagesPaths(images_root, [timestamp], hive_nb, exact_image=True)
    row = df.iloc[0]

    if row.isna().any():
        # Fall back to nearest-match for every camera of this hive.
        df = fetchImagesPaths(images_root, [timestamp], hive_nb, exact_image=False)
        row = df.iloc[0]

    return {col: (row[col] if pd.notna(row[col]) else None) for col in df.columns if col != "valid"}
