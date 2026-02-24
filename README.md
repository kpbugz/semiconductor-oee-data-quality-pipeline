# Semiconductor OEE Data Quality Pipeline

## Project Summary

This project implements a Python-based data quality and OEE (Overall Equipment Effectiveness) pipeline inspired by real semiconductor test-floor operations.

The pipeline validates production data before KPI computation, applies dataset-level quality gates, and computes Availability, Performance, Yield, and OEE using trusted data only. Results are summarized by tester and device to support operational decision-making and prevent misleading production metrics.

This project was built as a capstone while transitioning deeper into data engineering from a manufacturing product engineering role, with strong emphasis on defensive coding, data quality, and domain-driven metric design.

The goal is to:
- Validate production data before KPI computation
- Detect missing, invalid, or risky data early (fail fast)
- Compute core manufacturing metrics such as Yield, Availability, Performance, and OEE
- Produce actionable summaries by tester and device

## Professional & Domain Context

While my title is Manufacturing Product Engineer, my current role is heavily data-focused.

I work extensively with:
- ETL and structuring of large test log files (.txt) and CSV outputs
- Parsing and validating device-level test results after production runs
- Automating data ingestion and cleanup using Python and Excel/VBA
- Producing yield, throughput, and OEE-related reports for stakeholders

Because inaccurate or incomplete data can directly lead to incorrect OEE reporting and poor production decisions, this project reflects real manufacturing constraints such as:
- Treating missing identifiers and denominators as fatal errors
- Avoiding overreaction to noisy yield metrics on small sample sizes
- Separating data validation from KPI computation
- Designing metrics to reflect how production teams actually consume them

The pipeline design is intentionally grounded in semiconductor test operations rather than abstract data examples.

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

✔ This section is now **fully closed**  
✔ Everything below will render normally

---

## Sample Output

2026-02-23 21:30:53,643 - INFO - Starting data quality pipeline
Success! Loaded 360 rows.
Read attempt finished
2026-02-23 21:30:53,715 - WARNING - [oee] Type issues: {'date': 'object'}
2026-02-23 21:30:53,720 - INFO - [oee] Gate PASS: {'core_missing_pct': np.float64(0.28), 'warn_pct': 0.5, 'fail_pct': 2.0}
Success! Loaded 4 rows.
Read attempt finished
2026-02-23 21:30:53,844 - INFO - [devices] Gate PASS: {'core_missing_pct': np.float64(0.0), 'warn_pct': 0.0, 'fail_pct': 0.1}
Success! Loaded 4 rows.
Read attempt finished
2026-02-23 21:30:53,847 - WARNING - [testers] Type issues: {'platform': 'int64'}
2026-02-23 21:30:53,850 - INFO - [testers] Gate PASS: {'core_missing_pct': np.float64(0.0), 'warn_pct': 0.0, 'fail_pct': 0.1}

--- DATASET STATUS SUMMARY ---
{'dataset': 'oee', 'status': 'PASS', 'core_missing_pct': np.float64(0.28), 'warn_pct': 0.5, 'fail_pct': 2.0}
{'dataset': 'devices', 'status': 'PASS', 'core_missing_pct': np.float64(0.0), 'warn_pct': 0.0, 'fail_pct': 0.1}
{'dataset': 'testers', 'status': 'PASS', 'core_missing_pct': np.float64(0.0), 'warn_pct': 0.0, 'fail_pct': 0.1}

--- TOP TESTER KPIs (avg) ---
           avg_oee  avg_yield  total_units_tested
tester_id                                        
T5600-01       NaN     0.9816             8695322
T5600-02       NaN     0.9821             8707172
T5601-01       NaN     0.9819             4535469
T5602-01       NaN     0.9827            17269997

--- TOP DEVICE KPIs (avg) ---
        avg_oee  avg_yield  total_units_tested
device                                        
AAPL        NaN     0.9725            16028105
GOOG        NaN     0.9873             5726919
META        NaN     0.9855             8549169
NVDA        NaN     0.9864             8903767

2026-02-23 21:30:53,893 - INFO - Pipeline completed successfully.

---

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
## Production Notes

- This pipeline is designed to **fail fast** on schema or fatal data issues.
- Quality gates return explicit PASS / WARN / FAIL signals suitable for schedulers (Airflow, cron, CI).
- Local file paths are used for development; paths can be parameterized for production or cloud storage.
- Raw production data is intentionally excluded from version control.

## Development Notes (AI-Assisted Workflow)

This project was developed using an AI-assisted workflow (ChatGPT) as a productivity and iteration tool, similar to how engineers use internal templates, documentation, or code references.

My primary contributions include:
- Defining the problem scope and success criteria
- Designing the data quality strategy (fatal vs recoverable rules, quality gates)
- Modeling semiconductor-relevant KPIs (Availability, Performance, Yield, OEE)
- Structuring the pipeline for modularity and future dataset expansion
- Validating outputs against real manufacturing and test-floor behavior
- Deciding which metrics are actionable daily vs weekly

AI-generated code was treated as a draft and reviewed, modified, or rejected as needed. All logic, rules, and interpretations were validated and owned by me to reflect real production environments.

## Next Steps
- Refactor pipeline into fully modular Python packages
- Persist quality results and KPIs using SQL
- Add historical KPI trend analysis
- Prepare dashboards for operational monitoring
