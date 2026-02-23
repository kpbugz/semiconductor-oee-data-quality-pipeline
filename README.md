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

---

## Status
🚧 **Work in Progress**  
This repository is updated daily as part of a structured learning plan (Weeks 5–8).

---

## Next Steps
- Modularize code into separate Python modules
- Add logging to file
- Add KPI trend summaries
- Add SQL-based persistence layer
