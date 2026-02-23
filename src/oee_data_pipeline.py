#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime
import argparse
import sys

# =========================
# CONFIG
# =========================
DEBUG_VERBOSE = False

QUALITY_OK = "OK"
QUALITY_MISSING = "MISSING_DATA"
QUALITY_BAD = "BAD_DATA"

REQUIRED_COLUMNS = {
    "date": "datetime",
    "tester_id": "string",
    "device": "string",
    "avg_test_time_sec": "numeric",
    "units_tested": "numeric",
    "units_out": "numeric",
    "planned_prod_min": "numeric",
    "unplanned_downtime_min": "numeric",
}
CHECK_COLS = ["avg_test_time_sec", "units_tested", "units_out", "planned_prod_min", "unplanned_downtime_min"]

CONFIG = {
    "logging": {
        "log_to_file": True,
        "log_dir": "logs"
    }
}

DATASETS = {
    "oee": {
        "path": r"D:\Python_Scripts\semicapstone_data\tester_oee_daily.csv",
        "required_columns": REQUIRED_COLUMNS,
        "must_not_be_missing": ["date", "tester_id", "device"],
        "core_metric_cols": [
            "planned_prod_min",
            "unplanned_downtime_min",
            "avg_test_time_sec",
            "units_tested",
            "units_out",
        ],
        "check_cols": CHECK_COLS,
        "fill_with_mean": ["avg_test_time_sec"],
        "gates": {"warn_pct": 0.5, "fail_pct": 2.0},
        "apply_quality_rules": True,
    },

    "devices": {
        "path": r"D:\Python_Scripts\semicapstone_data\devices.csv",
        "required_columns": {
            "device": "string",
            "test_time_sec_target": "numeric",
        },
        "must_not_be_missing": ["device"],
        "core_metric_cols": ["device", "test_time_sec_target"],
        "check_cols": ["device", "test_time_sec_target"],
        "fill_with_mean": [],
        "gates": {"warn_pct": 0.0, "fail_pct": 0.1},  # dims should be nearly perfect
        "apply_quality_rules": False,
    },

    "testers": {
        "path": r"D:\Python_Scripts\semicapstone_data\testers.csv",
        "required_columns": {
            "tester_id": "string",
            "platform": "string",
        },
        "must_not_be_missing": ["tester_id"],
        "core_metric_cols": ["tester_id", "platform"],
        "check_cols": ["tester_id", "platform"],
        "fill_with_mean": [],
        "gates": {"warn_pct": 0.0, "fail_pct": 0.1},
        "apply_quality_rules": False,
    },
}

# =========================
# DATA LOADING
# =========================

def safe_load_csv(file_path, logger) -> pd.DataFrame | None:
    df = None
    try:
        df = pd.read_csv(file_path)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: Failed to parse CSV. Check for formatting or delimiter issues.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    else:
        print(f"Success! Loaded {len(df)} rows.")

    finally:
        print("Read attempt finished")
    return df

# =========================
# VALIDATION
# =========================

def validate_required_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    return missing

# Use pd.api.types to check dtype
def validate_column_types(df, schema):
    type_errors = {}

    for col, expected_type in schema.items():
        if col not in df.columns:
            continue

        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                type_errors[col] = str(df[col].dtype)

        elif expected_type == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                type_errors[col] = str(df[col].dtype)

        elif expected_type == "string":
            if not pd.api.types.is_string_dtype(df[col]):
                type_errors[col] = str(df[col].dtype)

    return type_errors

# =========================
# QUALITY RULES
# =========================

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        #Counts how many NaN (null) values exist in every column.
        "missing_count": df.isna().sum(), 
        # Calculates the percentage of missing data (e.g., if 10 out of 100 rows are empty, it returns 10.0)
        "missing_pct": (df.isna().mean() * 100).round(2),
        # Grabs the data type of each column so you can see if a column is missing data and has the wrong type simultaneously.
        "dtype": df.dtypes.astype(str)
    })

    # Discards columns that are 100% complete, so the report only focuses on the problem areas.
    summary = summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
    return summary

# function to hand back columns with NA values that will cause fatal errors in the downstream processes
def check_fatal_missing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    fatal_cols = []
    for c in cols:
        if c in df.columns and df[c].isna().any():
            fatal_cols.append(c)
    return fatal_cols

# Fill NA values using the mean of healthy row data on the column being checked
def fill_numeric_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    existing = [c for c in cols if c in df2.columns]
    df2[existing] = df2[existing].apply(pd.to_numeric, errors="coerce")
    df2[existing] = df2[existing].fillna(df2[existing].mean())
    return df2

# This function creates a "safety tag" for every row in your dataset, marking rows that are incomplete. 
# It’s a great way to filter out bad data later without deleting it immediately.
def add_missing_flag(df: pd.DataFrame, cols_to_check: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    existing = [c for c in cols_to_check if c in df2.columns]
    df2["missing_flag"] = df2[existing].isna().any(axis=1)
    return df2

# Check row quality
def classify_quality(row) -> str:
    try:
        # --- Missing data check ---
        if row[CHECK_COLS].isna().any():
            return QUALITY_MISSING

        # --- BAD_DATA rules ---
        if row["units_tested"] <= 0:
            return QUALITY_BAD

        if row["units_out"] < 0 or row["units_out"] > row["units_tested"]:
            return QUALITY_BAD

        if row["avg_test_time_sec"] <= 0:
            return QUALITY_BAD

        if row["planned_prod_min"] <= 0:
            return QUALITY_BAD

        if row["unplanned_downtime_min"] < 0:
            return QUALITY_BAD

        if row["unplanned_downtime_min"] > row["planned_prod_min"]:
            return QUALITY_BAD

        return QUALITY_OK

    except Exception:
        return QUALITY_BAD

# =========================
# REPORTING
# =========================

def build_breakdowns(df_qc: pd.DataFrame):
    tester_quality = (
        df_qc.groupby(["tester_id", "quality_flag"])
             .size()
             .unstack(fill_value=0)
    )
    device_quality = (
        df_qc.groupby(["device", "quality_flag"])
             .size()
             .unstack(fill_value=0)
    )

    # Sort by BAD_DATA if exists, else sort index
    if QUALITY_BAD in tester_quality.columns:
        tester_quality = tester_quality.sort_values(by=QUALITY_BAD, ascending=False)
    else:
        tester_quality = tester_quality.sort_index()

    if QUALITY_BAD in device_quality.columns:
        device_quality = device_quality.sort_values(by=QUALITY_BAD, ascending=False)
    else:
        device_quality = device_quality.sort_index()

    return tester_quality, device_quality

def generate_quality_report(df_qc: pd.DataFrame, tester_quality: pd.DataFrame, device_quality: pd.DataFrame) -> dict:
    report = {
        "total_rows": len(df_qc),
        "ok_pct": round((df_qc["quality_flag"] == QUALITY_OK).mean() * 100, 2),
        "missing_pct": round((df_qc["quality_flag"] == QUALITY_MISSING).mean() * 100, 2),
        "bad_pct": round((df_qc["quality_flag"] == QUALITY_BAD).mean() * 100, 2),
        "worst_tester": tester_quality.index[0] if not tester_quality.empty else None,
        "worst_device": device_quality.index[0] if not device_quality.empty else None,
    }
    return report

def print_final_report(report: dict, tester_quality: pd.DataFrame, device_quality: pd.DataFrame):
    print("\n--- FINAL QUALITY REPORT ---")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n--- TOP 5 TESTERS (by BAD_DATA if present) ---")
    print(tester_quality.head())

    print("\n--- TOP 5 DEVICES (by BAD_DATA if present) ---")
    print(device_quality.head())

def evaluate_quality_gates(df: pd.DataFrame, config: dict) -> tuple[str, dict]:
    core_cols = config["core_metric_cols"]
    warn_pct = config["gates"]["core_missing_warn_pct"]
    fail_pct = config["gates"]["core_missing_fail_pct"]

    core_missing_pct = df[core_cols].isna().any(axis=1).mean() * 100

    details = {
        "core_missing_pct": round(core_missing_pct, 2),
        "warn_threshold_pct": warn_pct,
        "fail_threshold_pct": fail_pct,
    }

    if core_missing_pct > fail_pct:
        return "FAIL", details
    if core_missing_pct > warn_pct:
        return "WARN", details
    return "PASS", details

def compute_availability(df: pd.DataFrame) -> pd.Series:
    """
    Availability = (Planned Production Time - Unplanned Downtime) / Planned Production Time
    """
    return np.where(
        df["planned_prod_min"] > 0,
        (df["planned_prod_min"] - df["unplanned_downtime_min"]) / df["planned_prod_min"],
        np.nan
    )

def compute_performance(df: pd.DataFrame) -> pd.Series:
    """
    Performance = Ideal Test Time / Actual Test Time
    """
    return np.where(
        df["avg_test_time_sec"] > 0,
        df["test_time_sec_target"] / df["avg_test_time_sec"],
        np.nan
    )

def compute_yield(df: pd.DataFrame) -> pd.Series:
    """
    Yield = Units Out / Units Tested
    """
    return np.where(
        df["units_tested"] > 0,
        df["units_out"] / df["units_tested"],
        np.nan
    )

def compute_oee(df: pd.DataFrame) -> pd.Series:
    """
    OEE = Availability × Performance × Yield
    """
    return df["availability"] * df["performance"] * df["yield"]

# =========================
# PIPELINE ORCHESTRATION
# =========================

def setup_logger(log_to_file=False, log_dir="logs"):
    # Ensure log directory exists
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("DataQualityPipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Optional)
    if log_to_file:
        log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="OEE Data Quality Pipeline", add_help=True)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to validate"
    )

    # ✅ Notebook-safe: ignore unknown args (like -f from ipykernel)
    args, _unknown = parser.parse_known_args()
    return args

def evaluate_gates(df: pd.DataFrame, core_cols: list[str], warn_pct: float, fail_pct: float):
    core_missing_pct = df[core_cols].isna().any(axis=1).mean() * 100
    details = {"core_missing_pct": round(core_missing_pct, 2), "warn_pct": warn_pct, "fail_pct": fail_pct}

    if core_missing_pct > fail_pct:
        return "FAIL", details
    if core_missing_pct > warn_pct:
        return "WARN", details
    return "PASS", details


def run_dataset(logger, name: str, cfg: dict):
    path = Path(cfg["path"])
    df = safe_load_csv(path, logger)
    if df is None:
        return False, None, {"dataset": name, "status": "FAIL", "reason": "Load failed"}

    missing_cols = validate_required_columns(df, cfg["required_columns"])
    if missing_cols:
        logger.error(f"[{name}] Missing required columns: {missing_cols}")
        return False, None, {"dataset": name, "status": "FAIL", "reason": "Missing columns", "missing_cols": missing_cols}

    type_issues = validate_column_types(df, cfg["required_columns"])
    if type_issues:
        logger.warning(f"[{name}] Type issues: {type_issues}")

    fatal = check_fatal_missing(df, cfg["must_not_be_missing"])
    if fatal:
        logger.error(f"[{name}] Fatal missing in identifiers: {fatal}")
        return False, df, {"dataset": name, "status": "FAIL", "reason": "Fatal identifier missing", "fatal_cols": fatal}

    # Optional fill
    if cfg["fill_with_mean"]:
        df = fill_numeric_mean(df, cfg["fill_with_mean"])

    # Gate decision
    status, gate_details = evaluate_gates(df, cfg["core_metric_cols"], cfg["gates"]["warn_pct"], cfg["gates"]["fail_pct"])
    if status == "FAIL":
        logger.error(f"[{name}] Gate FAIL: {gate_details}")
        return False, df, {"dataset": name, "status": "FAIL", "reason": "Gate failed", **gate_details}
    elif status == "WARN":
        logger.warning(f"[{name}] Gate WARN: {gate_details}")
    else:
        logger.info(f"[{name}] Gate PASS: {gate_details}")

    # Only apply row-level quality rules for fact datasets
    if cfg["apply_quality_rules"]:
        df = add_missing_flag(df, cfg["check_cols"])
        df["quality_flag"] = df.apply(classify_quality, axis=1)

    return True, df, {"dataset": name, "status": status, **gate_details}

def run_pipeline(logger, config: dict):
    oee_path = Path(config["paths"]["oee"])

    df = safe_load_csv(oee_path, logger)
    if df is None:
        return False, None, None, None, {"status": "FAIL", "reason": "Load failed"}

    # Schema checks
    required_columns = config["required_columns"]
    missing_cols = validate_required_columns(df, required_columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False, None, None, None, {"status": "FAIL", "reason": "Missing required columns", "missing_cols": missing_cols}

    type_issues = validate_column_types(df, required_columns)
    if type_issues:
        logger.warning(f"Type issues detected: {type_issues}")

    # Parse date (recoverable)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Fatal identifiers missing
    fatal = check_fatal_missing(df, config["must_not_be_missing"])
    if fatal:
        logger.error(f"Fatal missing in identifiers: {fatal}")
        return False, None, None, None, {"status": "FAIL", "reason": "Fatal identifier missing", "fatal_cols": fatal}

    # Optional fills
    df = fill_numeric_mean(df, config["fill_with_mean"])

    # Row flags
    df = add_missing_flag(df, config["check_cols"])
    df["quality_flag"] = df.apply(classify_quality, axis=1)

    # Gates (dataset-level)
    status, gate_details = evaluate_quality_gates(df, config)
    if status == "FAIL":
        logger.error(f"Quality gate FAIL: {gate_details}")
        return False, df, None, None, {"status": "FAIL", "reason": "Quality gate failed", **gate_details}
    elif status == "WARN":
        logger.warning(f"Quality gate WARN: {gate_details}")
    else:
        logger.info(f"Quality gate PASS: {gate_details}")

    tester_quality, device_quality = build_breakdowns(df)
    report = generate_quality_report(df, tester_quality, device_quality)

    # Attach status + gate details to final report
    report["status"] = status
    report.update(gate_details)

    logger.info(f"Final report: {report}")
    return True, df, tester_quality, device_quality, report

def safe_exit(code: int):
# In notebooks, avoid sys.exit() noise
    if "ipykernel" in sys.modules:
        return
    sys.exit(code)

# =========================
# Main: OEE Metric Calculations
# =========================

def main():
    args = parse_args()

    logger = setup_logger(
        log_to_file=CONFIG["logging"]["log_to_file"],
        log_dir=CONFIG["logging"]["log_dir"]
    )

    logger.info("Starting data quality pipeline")

    results = []
    frames = {}

    datasets_to_run = DATASETS.keys() if args.dataset == "all" else [args.dataset]

    for name in datasets_to_run:
        cfg = DATASETS[name]
        ok, df, result = run_dataset(logger, name, cfg)
        results.append(result)
        frames[name] = df

    print("\n--- DATASET STATUS SUMMARY ---")
    for r in results:
        print(r)

    # Stop publishing if any dataset FAILS
    if any(r["status"] == "FAIL" for r in results):
        logger.error("One or more datasets FAILED. Exiting with code 1.")
        safe_exit(1)
        return

    # =========================
    # KPI COMPUTATION (OEE only)
    # =========================
    if "oee" in frames and frames["oee"] is not None:
        oee_qc = frames["oee"].copy()

        # Join test_time_sec_target from devices onto oee (needed for performance)
        if "devices" in frames and frames["devices"] is not None:
            devices = frames["devices"][["device", "test_time_sec_target"]].copy()
            oee_qc = oee_qc.merge(devices, on="device", how="left")
        else:
            logger.warning("Devices dataset not loaded; performance metric may be NaN (missing test_time_sec_target).")

        if "quality_flag" not in oee_qc.columns:
            logger.warning("quality_flag not found. Metrics will be computed only where inputs are valid.")
            valid_mask = (
                oee_qc[["planned_prod_min","unplanned_downtime_min","avg_test_time_sec","units_tested","units_out"]]
                .notna().all(axis=1)
            )
        else:
            valid_mask = oee_qc["quality_flag"] == "OK"

        # Compute metrics
        oee_qc.loc[valid_mask, "availability"] = compute_availability(oee_qc.loc[valid_mask])
        oee_qc.loc[valid_mask, "yield"] = compute_yield(oee_qc.loc[valid_mask])

        # Performance needs test_time_sec_target; handle missing safely
        if "test_time_sec_target" in oee_qc.columns:
            oee_qc.loc[valid_mask, "performance"] = compute_performance(oee_qc.loc[valid_mask]).clip(0, 1.2)
        else:
            oee_qc["performance"] = np.nan

        oee_qc.loc[valid_mask, "oee"] = compute_oee(oee_qc.loc[valid_mask])

        # Quick KPI rollups (tester/device)
        tester_kpis = (
            oee_qc[valid_mask]
            .groupby("tester_id")
            .agg(avg_oee=("oee", "mean"),
                 avg_yield=("yield", "mean"),
                 total_units_tested=("units_tested", "sum"))
            .sort_values("avg_oee", ascending=False)
            .round(4)
        )

        device_kpis = (
            oee_qc[valid_mask]
            .groupby("device")
            .agg(avg_oee=("oee", "mean"),
                 avg_yield=("yield", "mean"),
                 total_units_tested=("units_tested", "sum"))
            .sort_values("avg_oee", ascending=False)
            .round(4)
        )

        print("\n--- TOP TESTER KPIs (avg) ---")
        print(tester_kpis.head())

        print("\n--- TOP DEVICE KPIs (avg) ---")
        print(device_kpis.head())

    logger.info("Pipeline completed successfully.")
    safe_exit(0)

if __name__ == "__main__":
    main()


# In[ ]:




