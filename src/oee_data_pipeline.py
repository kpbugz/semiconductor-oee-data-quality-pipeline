#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\tester_oee_daily.csv",
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
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\devices.csv",
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
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\testers.csv",
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

def safe_load_csv(file_path, logger=None, verbose=False) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"Success! Loaded {len(df)} rows.")
            print("Read attempt finished")
        return df
    except Exception as e:
        if logger:
            logger.error(f"Load failed for {file_path}: {e}")
        elif verbose:
            print(f"Load failed: {e}")
        return None

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
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Vectorized safe division: returns NaN where denominator <= 0 or denominator is NaN.
    """
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    mask = denom > 0
    out.loc[mask] = num.loc[mask] / denom.loc[mask]
    return out

def compute_oee_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds availability, performance, yield, and oee columns.
    Assumes df already contains:
      planned_prod_min, unplanned_downtime_min, avg_test_time_sec, test_time_sec_target,
      units_tested, units_out
    """
    df2 = df.copy()

    # Make sure numeric fields are numeric
    numeric_cols = [
        "planned_prod_min", "unplanned_downtime_min",
        "avg_test_time_sec", "test_time_sec_target",
        "units_tested", "units_out"
    ]
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Availability = (planned - unplanned) / planned
    df2["availability"] = safe_divide(
        df2["planned_prod_min"] - df2["unplanned_downtime_min"],
        df2["planned_prod_min"]
    )

    # Performance = target test time / actual avg test time
    df2["performance"] = safe_divide(
        df2["test_time_sec_target"],
        df2["avg_test_time_sec"]
    ).clip(lower=0, upper=1.2)  # guardband like you already do

    # Yield = units_out / units_tested
    df2["yield"] = safe_divide(df2["units_out"], df2["units_tested"])

    # OEE = A * P * Y
    df2["oee"] = df2["availability"] * df2["performance"] * df2["yield"]

    return df2

def build_kpi_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      tester_kpis: avg_oee, avg_yield, total_units_tested by tester_id
      device_kpis: avg_oee, avg_yield, total_units_tested by device
    Uses only rows where oee is not NaN.
    """
    dfk = df.copy()

    # Keep only rows where KPIs are computable
    kpi_mask = dfk["oee"].notna() & dfk["yield"].notna()
    dfk = dfk.loc[kpi_mask].copy()

    tester_kpis = (
        dfk.groupby("tester_id", dropna=False)
           .agg(
               avg_oee=("oee", "mean"),                # decimal 0..1
               avg_yield=("yield", "mean"),            # decimal 0..1
               total_units_tested=("units_tested", "sum"),
               oee_samples=("oee", "count")            # how many rows contributed
           )
    )

    # Convert to percentage (numeric) with 2 decimals
    tester_kpis["avg_oee_pct"] = (tester_kpis["avg_oee"] * 100).round(2)
    tester_kpis["avg_yield_pct"] = (tester_kpis["avg_yield"] * 100).round(2)

    # OEE bands (edit thresholds to match your floor standards)
    def oee_band(pct: float) -> str:
        if pd.isna(pct):
            return "NO_DATA"
        if pct >= 85:
            return "GREEN"
        if pct >= 70:
            return "YELLOW"
        return "RED"

    tester_kpis["oee_band"] = tester_kpis["avg_oee_pct"].apply(oee_band)

    # Sort by true numeric OEE (decimal) then units tested
    tester_kpis = tester_kpis.sort_values(["avg_oee", "total_units_tested"], ascending=[False, False])

    # Rank (1 = best). Use method="min" so ties share a rank.
    tester_kpis["oee_rank"] = tester_kpis["avg_oee"].rank(ascending=False, method="min").astype(int)

    device_kpis = (
        dfk.groupby("device", dropna=False)
           .agg(
               avg_oee=("oee", "mean"),
               avg_yield=("yield", "mean"),
               total_units_tested=("units_tested", "sum"),
           )
           .sort_values(["avg_oee", "total_units_tested"], ascending=[False, False])
    )

    return tester_kpis, device_kpis

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

def setup_logger(log_to_file: bool = False, log_dir: str = "logs", reset: bool = False):

    logger = logging.getLogger("DataQualityPipeline")
    logger.setLevel(logging.INFO)

    if reset:
        logger.handlers.clear()

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
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
    core_missing_pct = float(core_missing_pct)
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

    logger = setup_logger(..., reset=True)

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
    # ---- After you run datasets and have frames dict ----
    oee_df = frames.get("oee")
    devices_df = frames.get("devices")

    if oee_df is None or devices_df is None:
        logger.error("Missing required dataframes in memory (oee/devices).")
        safe_exit(1)
        return

    # Ensure date parsing (optional, but good)
    if "date" in oee_df.columns:
        oee_df["date"] = pd.to_datetime(oee_df["date"], errors="coerce")

    # Merge target test time into OEE (dimension join)
    # IMPORTANT: keep only one target column name
    oee_df = oee_df.drop(columns=["test_time_sec_target"], errors="ignore")
    oee_df = oee_df.merge(
        devices_df[["device", "test_time_sec_target"]],
        on="device",
        how="left"
    )

    # Only compute metrics on rows that passed row-level quality rules
    # (If your dims don't have quality_flag, this is only for oee_df which does.)
    if "quality_flag" not in oee_df.columns:
        logger.error("oee dataframe has no quality_flag. Make sure apply_quality_rules=True for oee dataset.")
        safe_exit(1)
        return

    ok_mask = oee_df["quality_flag"] == "OK"
    oee_ok = oee_df.loc[ok_mask].copy()

    # Compute metrics on OK rows only
    oee_ok = compute_oee_metrics(oee_ok)

    # Build KPI summaries from the same computed df
    tester_kpis, device_kpis = build_kpi_summaries(oee_ok)

    # print(oee_ok[["availability", "performance", "yield", "oee"]].isna().mean().round(4))

    print("\n--- DATASET STATUS SUMMARY ---")
    for r in results:
        print(r)

    # Stop publishing if any dataset FAILS
    if any(r["status"] == "FAIL" for r in results):
        logger.error("One or more datasets FAILED. Exiting with code 1.")
        safe_exit(1)
        return

    logger.info(f"OEE KPI rows used: {len(oee_ok)} / original OK rows: {ok_mask.sum()}")
    logger.info("Pipeline completed successfully.")

    TOP_N = 5

    print("\n--- TOP TESTERS BY OEE ---")
    print(tester_kpis[["avg_oee_pct", "oee_band", "avg_yield_pct", "total_units_tested", "oee_samples", "oee_rank"]].head(TOP_N))

    print("\n--- BOTTOM TESTERS BY OEE (worst first) ---")
    print(tester_kpis.sort_values(["avg_oee", "total_units_tested"], ascending=[True, False])
                    [["avg_oee_pct", "oee_band", "avg_yield_pct", "total_units_tested", "oee_samples", "oee_rank"]]
                    .head(TOP_N))

    print("\n--- OEE BAND COUNTS ---")
    print(tester_kpis["oee_band"].value_counts())

    safe_exit(0)

if __name__ == "__main__":
    main()


# In[ ]:




