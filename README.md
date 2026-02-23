# Semiconductor OEE Data Quality Pipeline

## Overview
This project implements a Python-based data quality and OEE (Overall Equipment Effectiveness) pipeline inspired by real semiconductor test operations.

The goal is to:
- Validate production data before KPI computation
- Detect missing, invalid, or risky data early (fail fast)
- Compute core manufacturing metrics such as Yield, Availability, Performance, and OEE
- Produce actionable summaries by tester and device

This project is built incrementally as a learning capstone while transitioning into a data engineering role.

---

## Dataset Context (Simulated)
The datasets simulate a semiconductor test floor environment:
- **OEE fact table**: daily tester performance
- **Devices dimension**: target test time per device
- **Testers dimension**: tester metadata

All data is synthetic / anonymized.

---

## Key Concepts Implemented
- Defensive CSV loading
- Schema validation (columns + data types)
- Fatal vs recoverable data quality rules
- Dataset-level quality gates (WARN / FAIL)
- Row-level quality classification (OK / MISSING_DATA / BAD_DATA)
- Core OEE metric calculations:
  - Availability
  - Performance
  - Yield
  - OEE = Availability × Performance × Yield

---

## Current Pipeline Flow
1. Load datasets safely
2. Validate schema and required columns
3. Identify fatal vs recoverable issues
4. Apply quality gates per dataset
5. Classify row-level data quality
6. Compute OEE metrics on valid data only
7. Generate summaries by tester and device

## How to Run

Run the pipeline on all datasets:
- python oee_data_pipeline.py

---
Run the pipeline on each dataset:
- python oee_data_pipeline.py --dataset oee
- python oee_data_pipeline.py --dataset devices
- python oee_data_pipeline.py --dataset testers

---

## Computed Metrics (Semiconductor KPIs)

After schema validation and quality gating, the pipeline computes core manufacturing KPIs using trusted data only.

- **Availability**  
  (Planned Production − Unplanned Downtime) / Planned Production  
  → Measures tester uptime and utilization

- **Performance**  
  Target Test Time / Actual Average Test Time  
  → Detects slowdowns caused by test program drift, retest loops, or hardware issues

- **Yield**  
  Units Out / Units Tested  
  → Measures output quality and process stability

- **OEE**  
  Availability × Performance × Yield  
  → High-level indicator of tester effectiveness

Rows classified as **MISSING_DATA** or **BAD_DATA** are excluded from KPI computation to prevent silent metric corruption.

## KPI Interpretation (Operational Context)

This pipeline is designed to support real production decision-making.

**Daily monitoring (heartbeat):**
- Tester Availability / OEE to quickly identify downtime
- Units tested and units out
- Dataset quality gate status (PASS / WARN / FAIL)

**Weekly escalation (trend-based):**
- Device yield (using sufficient sample size)
- Persistent low performance by tester or device
- Recurring downtime patterns

When OEE degradation is observed, it is decomposed into Availability, Performance, and Yield to identify the dominant operational driver instead of treating OEE as a root cause.

## Action Playbook

- **Availability drops:**  
  Investigate unplanned downtime, tester alarms, handler jams, and WIP/dispatch issues.

- **Performance drops:**  
  Isolate by tester vs device to differentiate between hardware degradation and test program changes.

- **Yield drops:**  
  Validate denominator integrity (`units_tested`), avoid reacting to small sample sizes, and escalate only if persistent or high-impact.

- **OEE drops:**  
  Decompose into Availability, Performance, and Yield before escalation.

## Status
🚧 **Work in Progress**  
This repository is updated daily as part of a structured learning plan (Weeks 5–8).

---

## Next Steps
- Modularize code into separate Python modules
- Add logging to file
- Add KPI trend summaries
- Add SQL-based persistence layer
