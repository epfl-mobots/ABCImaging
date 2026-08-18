'''
Corrects the raw, per-session bee count totals in bee_counts.csv into a more
plausible estimate of each hive's population over time.

Author: Cyril Monette
'''

# Postpones evaluation of annotations to strings (PEP 563), so the `int | None`
# union syntax below doesn't raise at import time on Python 3.7/3.8.
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make results_store importable by its sibling-directory location rather than
# relying on the caller's sys.path, so this module works when imported from
# another repo (e.g. `sys.path.append(".../BeeCounter")`, or copied elsewhere
# alongside results_store.py).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from results_store import DEFAULT_RESULTS_CSV  # noqa: E402


class HiveBeeCountEstimator:
    '''
    Turns the raw per-session bee count totals from bee_counts.csv into a
    corrected estimate of each hive's population over time, without touching
    the CSV itself.

    Two facts about the raw counts motivate the correction:
    - outside of brood-rearing periods the population can only decrease over
      time (bees die or are lost, none are added), so the corrected total
      must be non-increasing over time for a given hive;
    - each raw total is a lower bound rather than a true count (bees standing
      underneath others are invisible in the photo), so correcting a session
      should only ever raise its total, never lower it.

    A session's corrected total is therefore the maximum of its own raw total
    and every later session's raw total for the same hive: the smallest
    non-increasing sequence that stays at or above every raw observation.
    '''

    def __init__(self, csv_path: str = DEFAULT_RESULTS_CSV):
        '''
        :param csv_path: str, path to the bee_counts.csv results file.
        '''
        self._raw_sessions = self._load_raw_sessions(csv_path)

    @staticmethod
    def _load_raw_sessions(csv_path: str) -> pd.DataFrame:
        '''
        Loads bee_counts.csv and collapses it to one row per session (one
        hive at one timestamp), summing estimated_total_bees across the
        session's labeled cameras.

        :param csv_path: str, path to the bee_counts.csv results file.
        :return: DataFrame with columns session_id, hive, timestamp,
            n_cameras, raw_total_bees; sorted by hive then timestamp.
        '''
        df = pd.read_csv(csv_path)
        df["requested_timestamp"] = pd.to_datetime(df["requested_timestamp"], utc=True)

        sessions = df.groupby("session_id").agg(
            hive=("hive", "first"),
            timestamp=("requested_timestamp", "first"),
            n_cameras=("camera", "count"),
            raw_total_bees=("estimated_total_bees", "sum"),
        )
        return sessions.reset_index().sort_values(["hive", "timestamp"])

    def corrected_estimates(self, hive: int | None = None) -> pd.DataFrame:
        '''
        Returns, per session, both the raw and the corrected bee count total.

        :param hive: int, restrict the result to a single hive number. None
            (default) returns all hives.
        :return: DataFrame sorted by hive then timestamp, with columns
            session_id, hive, timestamp, n_cameras, raw_total_bees,
            corrected_total_bees.
        '''
        sessions = self._raw_sessions
        if hive is not None:
            sessions = sessions[sessions["hive"] == hive]
        sessions = sessions.sort_values(["hive", "timestamp"]).copy()

        # Suffix max of raw_total_bees within each hive, computed by
        # reversing, taking the running max per hive, then reversing back:
        # the smallest per-hive sequence that is both non-increasing and
        # never below a raw observation.
        sessions["corrected_total_bees"] = (
            sessions.iloc[::-1].groupby("hive")["raw_total_bees"].cummax().iloc[::-1]
        )
        return sessions.reset_index(drop=True)

    def estimate_bee_count(self, hive: int, timestamp) -> float:
        '''
        Best estimate of a hive's bee population at an arbitrary timestamp.

        Labeled sessions are sparse (a handful per hive), so a query rarely
        lands exactly on one. Between two sessions, the count is linearly
        interpolated between their corrected totals; because those are
        non-increasing, this stays within the two known bounds. Outside the
        labeled range there are no such bounds, so that raises instead of
        extrapolating.

        :param hive: int, hive number.
        :param timestamp: anything accepted by pd.Timestamp (e.g. str,
            datetime.datetime). If not tz-aware, assumed UTC.
        :return: float, estimated bee count for that hive at that timestamp.
        :raises ValueError: if the hive has no labeled sessions, or timestamp
            falls outside the hive's labeled session range.
        '''
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")

        hive_sessions = self.corrected_estimates(hive=hive)
        if hive_sessions.empty:
            raise ValueError(f"No labeled sessions for hive {hive}.")

        first_ts, last_ts = hive_sessions["timestamp"].iloc[0], hive_sessions["timestamp"].iloc[-1]
        if timestamp < first_ts or timestamp > last_ts:
            raise ValueError(
                f"Timestamp {timestamp} is outside hive {hive}'s labeled session "
                f"range ({first_ts} to {last_ts})."
            )

        # np.interp needs numeric, increasing x; corrected_estimates is
        # already sorted by timestamp ascending.
        known_x = hive_sessions["timestamp"].astype("int64").to_numpy()
        known_y = hive_sessions["corrected_total_bees"].to_numpy()
        return float(np.interp(timestamp.value, known_x, known_y))
