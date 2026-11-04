"""
TrajectoryLoader Class - C-MAPSS Engine Trajectory Loader
Loads, validates, and slices operational engine histories from C-MAPSS data.

Roadmap Reference: Phase 5 - Steps 5.1, 5.2, 5.3, 5.4
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


C_MAPSS_COLUMNS = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


class TrajectoryLoader:
    """
    Robust reader for NASA C-MAPSS engine run-to-failure or test trajectories.
    Preserves the standard 26-column schema and validates monotonic cycle sequencing.
    """

    def __init__(self, data_dir: str = os.path.join("dataset", "raw", "C-MAPSS")):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"C-MAPSS raw directory not found at: {self.data_dir}")

    def get_filepath(self, subset: str, split: str = "train") -> str:
        """Resolve dataset file path for a given subset and split."""
        subset_upper = subset.upper()
        if subset_upper not in ["FD001", "FD002", "FD003", "FD004"]:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of ['FD001', 'FD002', 'FD003', 'FD004'].")

        split_lower = split.lower()
        if split_lower not in ["train", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be either 'train' or 'test'.")

        filename = f"{split_lower}_{subset_upper}.txt"
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        return filepath

    def load_dataset(self, subset: str, split: str = "train") -> pd.DataFrame:
        """Load entire subset dataset partition."""
        filepath = self.get_filepath(subset, split)
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=C_MAPSS_COLUMNS)
        if df.shape[1] != 26:
            raise ValueError(f"Expected 26 columns in {filepath}, found {df.shape[1]}.")
        return df

    def get_available_units(self, subset: str, split: str = "train") -> List[int]:
        """Return list of available engine unit IDs in the subset partition."""
        df = self.load_dataset(subset, split)
        return sorted(df["unit_number"].unique().astype(int).tolist())

    def load_engine_trajectory(
        self, subset: str, unit_number: int, split: str = "train"
    ) -> pd.DataFrame:
        """
        Load complete trajectory for a single engine unit.
        Validates engine existence, non-emptiness, and cycle monotonicity.
        """
        df_all = self.load_dataset(subset, split)
        available_units = sorted(df_all["unit_number"].unique().astype(int).tolist())

        if unit_number not in available_units:
            raise ValueError(
                f"Engine unit_number {unit_number} not found in {subset} {split}. "
                f"Available units: {available_units[:5]} ... {available_units[-5:]} (total {len(available_units)})."
            )

        df_engine = (
            df_all[df_all["unit_number"] == unit_number]
            .sort_values(by="time_cycles", ascending=True)
            .reset_index(drop=True)
        )

        # Validate strictly monotonic cycles starting at 1
        cycles = df_engine["time_cycles"].values
        if len(cycles) == 0:
            raise ValueError(f"Engine unit {unit_number} has 0 records.")
        if cycles[0] != 1:
            raise ValueError(f"Engine unit {unit_number} trajectory does not start at cycle 1 (starts at {cycles[0]}).")
        if not (np.diff(cycles) == 1).all():
            raise ValueError(f"Engine unit {unit_number} has non-consecutive or non-monotonic cycle steps.")

        return df_engine
