'''
Appends bee-counting labels to a CSV file. One row per labeled camera image; the
4 rows sharing the same session_id form one full hive count (group-by + sum
estimated_total_bees to get the hive total).

Author: Cyril Monette
'''

import os

import pandas as pd

DEFAULT_RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "bee_counts.csv")

COLUMNS = [
    "session_id",
    "saved_at",
    "hive",
    "requested_timestamp",
    "camera",
    "image_path",
    "image_width_px",
    "image_height_px",
    "zone_x0",
    "zone_y0",
    "zone_x1",
    "zone_y1",
    "zone_area_px",
    "image_area_px",
    "area_ratio_suggestion",
    "bee_count",
    "multiplication_factor",
    "estimated_total_bees",
]


def append_results(rows: list[dict], csv_path: str = DEFAULT_RESULTS_CSV) -> None:
    '''
    Appends the given rows to the results CSV, creating it (with header) if needed.

    :param rows: list of dict, each dict must have all keys in COLUMNS.
    :param csv_path: str, path to the results CSV.
    '''
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
