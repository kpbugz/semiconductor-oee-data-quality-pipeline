# Semiconductor OEE Data Quality Pipeline

## Project Summary

This project implements a Python-based data quality and OEE (Overall Equipment Effectiveness) pipeline inspired by real semiconductor test-floor operations.

The pipeline validates production data before KPI computation, applies dataset-level quality gates, and computes Availability, Performance, Yield, and OEE. Results are summarized by tester and device to support operational decision-making and prevent misleading production metrics.

This project was built as a capstone while transitioning deeper into data engineering from a manufacturing product engineering role, with strong emphasis on defensive coding, data quality, and domain-driven metric design.

The goal is to:
- Validate production data before KPI computation
- Detect missing, invalid, or risky data early (fail fast)
- Compute core manufacturing metrics such as Yield, Availability, Performance, and OEE
- Produce actionable summaries by tester and device

## Professional & Domain Context

While my title is Manufacturing Product Engineer in semiconductors field, my current role is heavily data-focused.

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

## Architecture & Design Decisions

This pipeline is intentionally structured to reflect real semiconductor manufacturing constraints.

Key design decisions:
- **Fail-fast on schema and identifier issues** to prevent corrupt KPIs
- **Separate data validation from metric computation**
- **Treat denominators as critical (units_tested, planned_prod_min)**
- **Apply dataset-level quality gates before row-level metrics**
- **Compute KPIs only on trusted (OK) rows**

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

---

## Data Quality Gates

Each dataset passes through explicit quality gates before metrics are computed.

| Gate Level | Meaning |
|-----------|--------|
| PASS | Data quality acceptable for KPI computation |
| WARN | Data usable but requires attention |
| FAIL | Data blocked from downstream processing |

Example (OEE fact table):
- WARN if core metric missing rate > 0.5%
- FAIL if core metric missing rate > 2.0%

Quality gates are designed to surface risk early without overreacting to minor noise.

## Metric Trust Model

Metrics are computed only on rows classified as `OK`.

Rows classified as:
- `MISSING_DATA`
- `BAD_DATA`

are **explicitly excluded** from KPI computation to prevent silent metric corruption.

This ensures:
- OEE is never inflated by invalid denominators
- Yield is not distorted by partial or malformed records
- Reported KPIs can be trusted by operations teams

## OEE Bands & Ranking

Average OEE values are classified into operational bands:

| Band | Meaning |
|----|--------|
| GREEN | Healthy performance |
| YELLOW | Requires monitoring |
| RED | Requires investigation |

Testers are ranked by:
- Average OEE
- Sample count (to avoid small-sample bias)

This allows teams to quickly identify both top and underperforming testers.

## Logging & Observability

The pipeline uses structured logging with:
- Console output for interactive use
- Optional file logging for scheduled runs

Duplicate logging is explicitly prevented to ensure clean, production-ready logs suitable for CI or schedulers.

## Sample Output

~~~text

2026-02-25 01:15:43,026 - INFO - Starting data quality pipeline
2026-02-25 01:15:43,031 - WARNING - [oee] Type issues: {'date': 'object'}
2026-02-25 01:15:43,036 - INFO - [oee] Gate PASS: {'core_missing_pct': 0.28, 'warn_pct': 0.5, 'fail_pct': 2.0}
2026-02-25 01:15:43,168 - INFO - [devices] Gate PASS: {'core_missing_pct': 0.0, 'warn_pct': 0.0, 'fail_pct': 0.1}
2026-02-25 01:15:43,171 - WARNING - [testers] Type issues: {'platform': 'int64'}
2026-02-25 01:15:43,173 - INFO - [testers] Gate PASS: {'core_missing_pct': 0.0, 'warn_pct': 0.0, 'fail_pct': 0.1}

--- DATASET STATUS SUMMARY ---
{'dataset': 'oee', 'status': 'PASS', 'core_missing_pct': 0.28, 'warn_pct': 0.5, 'fail_pct': 2.0}
{'dataset': 'devices', 'status': 'PASS', 'core_missing_pct': 0.0, 'warn_pct': 0.0, 'fail_pct': 0.1}
{'dataset': 'testers', 'status': 'PASS', 'core_missing_pct': 0.0, 'warn_pct': 0.0, 'fail_pct': 0.1}
2026-02-25 01:15:43,195 - INFO - OEE KPI rows used: 359 / original OK rows: 359
2026-02-25 01:15:43,196 - INFO - Pipeline completed successfully.

--- TOP TESTERS BY OEE ---
           avg_oee_pct oee_band  avg_yield_pct  total_units_tested  \
tester_id                                                            
T5601-01         90.93    GREEN          98.19             4535469   
T5600-02         90.89    GREEN          98.21             8707172   
T5602-01         89.87    GREEN          98.27            17269997   
T5600-01         89.09    GREEN          98.16             8695322   

           oee_samples  oee_rank  
tester_id                         
T5601-01            90         1  
T5600-02            90         2  
T5602-01            90         3  
T5600-01            89         4  

--- BOTTOM TESTERS BY OEE (worst first) ---
           avg_oee_pct oee_band  avg_yield_pct  total_units_tested  \
tester_id                                                            
T5600-01         89.09    GREEN          98.16             8695322   
T5602-01         89.87    GREEN          98.27            17269997   
T5600-02         90.89    GREEN          98.21             8707172   
T5601-01         90.93    GREEN          98.19             4535469   

           oee_samples  oee_rank  
tester_id                         
T5600-01            89         4  
T5602-01            90         3  
T5600-02            90         2  
T5601-01            90         1  

--- OEE BAND COUNTS ---
oee_band
GREEN    4
Name: count, dtype: int64

~~~

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

---
## Production Notes

- This pipeline is designed to **fail fast** on schema or fatal data issues.
- Quality gates return explicit PASS / WARN / FAIL signals suitable for schedulers (Airflow, cron, CI).
- Local file paths are used for development; paths can be parameterized for production or cloud storage.
- Raw production data is intentionally excluded from version control.

## Known Limitations

- Synthetic data is used (no proprietary production data)
- No persistence layer yet (in-memory processing)
- Thresholds are illustrative and should be tuned per factory
- No alerting or dashboard integration yet

## Roadmap

- Modularize into installable Python package
- Persist KPIs and quality results to SQL
- Add historical trend analysis
- Integrate with dashboards (Power BI / Tableau)

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
