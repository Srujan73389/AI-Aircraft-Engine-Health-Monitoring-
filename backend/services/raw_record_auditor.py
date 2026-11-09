"""
Raw Record Auditor Service
Phase 9 - Step 9.2: Bit-for-bit verification of displayed sensor values against raw C-MAPSS records.

Provides parsing and precision audit utilities that directly inspect raw dataset files
(dataset/raw/C-MAPSS/{train|test}_FD*.txt) and compare parsed values with state vectors and API responses.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

RAW_COLUMNS = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

SENSOR_NAME_MAP = {
    "sensor_1": "T2",
    "sensor_2": "T24",
    "sensor_3": "T30",
    "sensor_4": "T50",
    "sensor_5": "P2",
    "sensor_6": "P15",
    "sensor_7": "P30",
    "sensor_8": "Nf",
    "sensor_9": "Nc",
    "sensor_10": "epr",
    "sensor_11": "Ps30",
    "sensor_12": "phi",
    "sensor_13": "NRf",
    "sensor_14": "NRc",
    "sensor_15": "BPR",
    "sensor_16": "farB",
    "sensor_17": "htBleed",
    "sensor_18": "Nf_dmd",
    "sensor_19": "PCNfR_dmd",
    "sensor_20": "W31",
    "sensor_21": "W32",
}

OP_SETTING_NAME_MAP = {
    "op_setting_1": "altitude",
    "op_setting_2": "mach_number",
    "op_setting_3": "throttle_resolver_angle",
}

DEFAULT_RAW_DIR = os.path.join("dataset", "raw", "C-MAPSS")


def get_raw_filepath(subset: str, split: str = "train", data_dir: Optional[str] = None) -> str:
    """Resolves full path to the raw NASA C-MAPSS text file."""
    base_dir = data_dir or DEFAULT_RAW_DIR
    sub_upper = subset.upper()
    split_lower = split.lower()
    if sub_upper not in ("FD001", "FD002", "FD003", "FD004"):
        raise ValueError(f"Invalid subset '{subset}'. Supported: ['FD001', 'FD002', 'FD003', 'FD004'].")
    if split_lower not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Supported: ['train', 'test'].")

    filename = f"{split_lower}_{sub_upper}.txt"
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw C-MAPSS record file not found at: {filepath}")
    return filepath


def get_raw_record(
    subset: str, split: str, unit: int, cycle: int, data_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Parses and returns the exact space-delimited telemetry record directly from the raw dataset file.
    Does not use any intermediate cache, model, or scaler transformations.
    """
    filepath = get_raw_filepath(subset, split, data_dir)
    target_unit_str = str(int(unit))
    target_cycle_str = str(int(cycle))

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 26 and tokens[0] == target_unit_str and tokens[1] == target_cycle_str:
                # Parsed tokens
                record: Dict[str, float] = {}
                for idx, col in enumerate(RAW_COLUMNS):
                    record[col] = float(tokens[idx])

                # Enrich with canonical aliases (T30, Nf, altitude, etc.)
                for s_key, s_alias in SENSOR_NAME_MAP.items():
                    record[s_alias] = record[s_key]
                for op_key, op_alias in OP_SETTING_NAME_MAP.items():
                    record[op_alias] = record[op_key]

                return record

    raise KeyError(
        f"Record not found for {subset.upper()} {split.lower()} Unit {unit} Cycle {cycle} in {filepath}."
    )


def audit_telemetry_match(
    subset: str,
    split: str,
    unit: int,
    cycle: int,
    api_state_vector: Dict[str, Any],
    tolerance: float = 1e-5,
    data_dir: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs bit-for-bit numerical parity audit between raw file record and state vector output.
    Audits all 21 sensor channels and 3 operational settings.
    """
    raw_record = get_raw_record(subset, split, unit, cycle, data_dir)
    comparisons: Dict[str, Any] = {}
    all_matched = True

    # Audit 21 Sensors
    for s_key, s_alias in SENSOR_NAME_MAP.items():
        raw_val = raw_record[s_key]

        # Check API vector by alias (e.g. 'T30'), sensor key ('sensor_3'), or G-prefixed key
        api_val = None
        if s_alias in api_state_vector:
            api_val = api_state_vector[s_alias]
        elif s_key in api_state_vector:
            api_val = api_state_vector[s_key]
        else:
            # Check G-prefixed keys (e.g., G8_T30)
            for k, v in api_state_vector.items():
                if isinstance(k, str) and (k.endswith(f"_{s_alias}") or k.endswith(f"_{s_key}")):
                    api_val = v
                    break

        if api_val is None:
            all_matched = False
            comparisons[s_alias] = {
                "raw_value": raw_val,
                "api_value": None,
                "delta": None,
                "match": False,
                "error": "Missing in state vector",
            }
        else:
            delta = abs(float(api_val) - float(raw_val))
            matched = delta <= tolerance
            if not matched:
                all_matched = False
            comparisons[s_alias] = {
                "raw_value": raw_val,
                "api_value": float(api_val),
                "delta": delta,
                "match": matched,
            }

    # Audit 3 Operational Settings
    for op_key, op_alias in OP_SETTING_NAME_MAP.items():
        raw_val = raw_record[op_key]
        api_val = api_state_vector.get(op_alias, api_state_vector.get(op_key))

        if api_val is None:
            all_matched = False
            comparisons[op_alias] = {
                "raw_value": raw_val,
                "api_value": None,
                "delta": None,
                "match": False,
                "error": "Missing in state vector",
            }
        else:
            delta = abs(float(api_val) - float(raw_val))
            matched = delta <= tolerance
            if not matched:
                all_matched = False
            comparisons[op_alias] = {
                "raw_value": raw_val,
                "api_value": float(api_val),
                "delta": delta,
                "match": matched,
            }

    audit_summary = {
        "subset": subset.upper(),
        "split": split.lower(),
        "unit": unit,
        "cycle": cycle,
        "tolerance": tolerance,
        "total_channels_audited": len(comparisons),
        "all_matched": all_matched,
        "channels": comparisons,
    }

    return all_matched, audit_summary
