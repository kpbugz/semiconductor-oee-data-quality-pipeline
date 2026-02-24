#!/usr/bin/env python
# coding: utf-8

# In[15]:


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
    "logging": {"log_to_file": True, "log_dir": "logs"}
}

DATASETS = {
    "oee": {
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\tester_oee_daily.csv",
        "required_columns": REQUIRED_COLUMNS,
        "must_not_be_missing": ["date", "tester_id", "device"],
        "core_metric_cols": ["planned_prod_min","unplanned_downtime_min","avg_test_time_sec","units_tested","units_out"],
        "check_cols": CHECK_COLS,
        "fill_with_mean": ["avg_test_time_sec"],
        "gates": {"warn_pct": 0.5, "fail_pct": 2.0},
        "apply_quality_rules": True,
    },
    "devices": {
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\devices.csv",
        "required_columns": {"device": "string", "test_time_sec_target": "numeric"},
        "must_not_be_missing": ["device"],
        "core_metric_cols": ["device", "test_time_sec_target"],
        "check_cols": ["device", "test_time_sec_target"],
        "fill_with_mean": [],
        "gates": {"warn_pct": 0.0, "fail_pct": 0.1},
        "apply_quality_rules": False,
    },
    "testers": {
        "path": r"D:\ASEKH\K02943\Python_Scripts\semicapstone_data\testers.csv",
        "required_columns": {"tester_id": "string", "platform": "string"},
        "must_not_be_missing": ["tester_id"],
        "core_metric_cols": ["tester_id", "platform"],
        "check_cols": ["tester_id", "platform"],
        "fill_with_mean": [],
        "gates": {"warn_pct": 0.0, "fail_pct": 0.1},
        "apply_quality_rules": False,
    },
}

# =========================
# UTILITIES
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
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    args, _unknown = parser.parse_known_args()
    return args

def safe_exit(code: int):
    if "ipykernel" in sys.modules:
        return
    sys.exit(code)

def safe_load_csv(file_path, logger) -> pd.DataFrame | None:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Load failed for {file_path}: {e}")
        return None

# =========================
# VALIDATION + RULES
# =========================
def validate_required_columns(df, required_columns):
    return [col for col in required_columns if col not in df.columns]

def validate_column_types(df, schema):
    type_errors = {}
    for col, expected_type in schema.items():
        if col not in df.columns:
            continue
        if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            type_errors[col] = str(df[col].dtype)
        elif expected_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            type_errors[col] = str(df[col].dtype)
        elif expected_type == "string" and not pd.api.types.is_string_dtype(df[col]):
            type_errors[col] = str(df[col].dtype)
    return type_errors

def check_fatal_missing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].isna().any()]

def fill_numeric_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    existing = [c for c in cols if c in df2.columns]
    df2[existing] = df2[existing].apply(pd.to_numeric, errors="coerce")
    df2[existing] = df2[existing].fillna(df2[existing].mean())
    return df2

def add_missing_flag(df: pd.DataFrame, cols_to_check: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    existing = [c for c in cols_to_check if c in df2.columns]
    df2["missing_flag"] = df2[existing].isna().any(axis=1)
    return df2

def classify_quality(row) -> str:
    try:
        if row[CHECK_COLS].isna().any():
            return QUALITY_MISSING
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

def evaluate_gates(df: pd.DataFrame, core_cols: list[str], warn_pct: float, fail_pct: float):
    core_missing_pct = float(df[core_cols].isna().any(axis=1).mean() * 100)
    details = {"core_missing_pct": round(core_missing_pct, 2), "warn_pct": warn_pct, "fail_pct": fail_pct}
    if core_missing_pct > fail_pct:
        return "FAIL", details
    if core_missing_pct > warn_pct:
        return "WARN", details
    return "PASS", details

def run_dataset(logger, name: str, cfg: dict):
    df = safe_load_csv(Path(cfg["path"]), logger)
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

    if cfg["fill_with_mean"]:
        df = fill_numeric_mean(df, cfg["fill_with_mean"])

    status, gate_details = evaluate_gates(df, cfg["core_metric_cols"], cfg["gates"]["warn_pct"], cfg["gates"]["fail_pct"])
    if status == "FAIL":
        logger.error(f"[{name}] Gate FAIL: {gate_details}")
        return False, df, {"dataset": name, "status": "FAIL", "reason": "Gate failed", **gate_details}
    elif status == "WARN":
        logger.warning(f"[{name}] Gate WARN: {gate_details}")
    else:
        logger.info(f"[{name}] Gate PASS: {gate_details}")

    if cfg["apply_quality_rules"]:
        df = add_missing_flag(df, cfg["check_cols"])
        df["quality_flag"] = df.apply(classify_quality, axis=1)

    return True, df, {"dataset": name, "status": status, **gate_details}

# =========================
# METRICS + KPIs
# =========================
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    mask = denom > 0
    out.loc[mask] = num.loc[mask] / denom.loc[mask]
    return out

def compute_oee_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in ["planned_prod_min","unplanned_downtime_min","avg_test_time_sec","test_time_sec_target","units_tested","units_out"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2["availability"] = safe_divide(df2["planned_prod_min"] - df2["unplanned_downtime_min"], df2["planned_prod_min"])
    df2["performance"] = safe_divide(df2["test_time_sec_target"], df2["avg_test_time_sec"]).clip(0, 1.2)
    df2["yield"] = safe_divide(df2["units_out"], df2["units_tested"])
    df2["oee"] = df2["availability"] * df2["performance"] * df2["yield"]
    return df2

def oee_band(pct: float) -> str:
    if pd.isna(pct):
        return "NO_DATA"
    if pct >= 85:
        return "GREEN"
    if pct >= 70:
        return "YELLOW"
    return "RED"

def build_kpi_summaries(df_ok: pd.DataFrame):
    dfk = df_ok.copy()
    kpi_mask = dfk["oee"].notna() & dfk["yield"].notna()
    dfk = dfk.loc[kpi_mask].copy()

    tester_kpis = (
        dfk.groupby("tester_id", dropna=False)
           .agg(avg_oee=("oee","mean"), avg_yield=("yield","mean"), total_units_tested=("units_tested","sum"), oee_samples=("oee","count"))
    )
    tester_kpis["avg_oee_pct"] = (tester_kpis["avg_oee"] * 100).round(2)
    tester_kpis["avg_yield_pct"] = (tester_kpis["avg_yield"] * 100).round(2)
    tester_kpis["oee_band"] = tester_kpis["avg_oee_pct"].apply(oee_band)
    tester_kpis = tester_kpis.sort_values(["avg_oee","total_units_tested"], ascending=[False, False])
    tester_kpis["oee_rank"] = tester_kpis["avg_oee"].rank(ascending=False, method="min").astype(int)

    device_kpis = (
        dfk.groupby("device", dropna=False)
           .agg(avg_oee=("oee","mean"), avg_yield=("yield","mean"), total_units_tested=("units_tested","sum"), oee_samples=("oee","count"))
    )
    device_kpis["avg_oee_pct"] = (device_kpis["avg_oee"] * 100).round(2)
    device_kpis["avg_yield_pct"] = (device_kpis["avg_yield"] * 100).round(2)
    device_kpis["oee_band"] = device_kpis["avg_oee_pct"].apply(oee_band)
    device_kpis = device_kpis.sort_values(["avg_oee","total_units_tested"], ascending=[False, False])
    device_kpis["oee_rank"] = device_kpis["avg_oee"].rank(ascending=False, method="min").astype(int)

    return tester_kpis, device_kpis

# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    logger = setup_logger(
        log_to_file=CONFIG["logging"]["log_to_file"],
        log_dir=CONFIG["logging"]["log_dir"],
        reset=True
    )

    logger.info("Starting data quality pipeline")

    results = []
    frames = {}

    datasets_to_run = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in datasets_to_run:
        ok, df, result = run_dataset(logger, name, DATASETS[name])
        results.append(result)
        frames[name] = df

    print("\n--- DATASET STATUS SUMMARY ---")
    for r in results:
        print(r)

    if any(r["status"] == "FAIL" for r in results):
        logger.error("One or more datasets FAILED. Exiting with code 1.")
        safe_exit(1)
        return

    oee_df = frames.get("oee")
    devices_df = frames.get("devices")
    if oee_df is None or devices_df is None:
        logger.error("Missing required dataframes in memory (oee/devices).")
        safe_exit(1)
        return

    if "date" in oee_df.columns:
        oee_df["date"] = pd.to_datetime(oee_df["date"], errors="coerce")

    oee_df = oee_df.drop(columns=["test_time_sec_target"], errors="ignore")
    oee_df = oee_df.merge(devices_df[["device","test_time_sec_target"]], on="device", how="left")

    if "quality_flag" not in oee_df.columns:
        logger.error("oee dataframe has no quality_flag. Make sure apply_quality_rules=True for oee dataset.")
        safe_exit(1)
        return

    ok_mask = oee_df["quality_flag"] == QUALITY_OK
    oee_ok = compute_oee_metrics(oee_df.loc[ok_mask].copy())

    tester_kpis, device_kpis = build_kpi_summaries(oee_ok)

    logger.info(f"OEE KPI rows used: {len(oee_ok)} / original OK rows: {ok_mask.sum()}")
    logger.info("Pipeline completed successfully.")

    TOP_N = 5

    # print("\n--- TOP TESTERS BY OEE ---")
    # print(tester_kpis[["avg_oee_pct","oee_band","avg_yield_pct","total_units_tested","oee_samples","oee_rank"]].head(TOP_N))

    print("\n--- BOTTOM TESTERS BY OEE (worst first) ---")
    print(tester_kpis.sort_values(["avg_oee","total_units_tested"], ascending=[True, False])
                    [["avg_oee_pct","oee_band","avg_yield_pct","total_units_tested","oee_samples","oee_rank"]]
                    .head(TOP_N))

    # print("\n--- TOP DEVICES BY OEE ---")
    # print(device_kpis[["avg_oee_pct","oee_band","avg_yield_pct","total_units_tested","oee_samples","oee_rank"]].head(TOP_N))

    print("\n--- BOTTOM DEVICES BY OEE (worst first) ---")
    print(device_kpis.sort_values(["avg_oee","total_units_tested"], ascending=[True, False])
                    [["avg_oee_pct","oee_band","avg_yield_pct","total_units_tested","oee_samples","oee_rank"]]
                    .head(TOP_N))

    print("\n--- OEE BAND COUNTS (TESTERS) ---")
    print(tester_kpis["oee_band"].value_counts())

    print("\n--- OEE BAND COUNTS (DEVICES) ---")
    print(device_kpis["oee_band"].value_counts())

    safe_exit(0)

if __name__ == "__main__":
    main()


# In[ ]:




