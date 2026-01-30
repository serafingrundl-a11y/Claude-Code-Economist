# Run Log - DACA Replication Study 39

## Overview
This log documents the analysis process for estimating the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (defined as usually working 35 hours per week or more)?

## Data Source
- American Community Survey (ACS) 2006-2016 via IPUMS USA
- Supplemental state demographic and policy data (optional, not used in primary analysis)

---

## Session Log

### Step 1: Data Exploration
**Date:** 2026-01-25

**Files Identified:**
- `data/data.csv` - Main ACS data file (6.2GB)
- `data/acs_data_dict.txt` - Data dictionary
- `data/state_demo_policy.csv` - Optional state-level data
- `data/State Level Data Documentation.docx` - Documentation for state data

**Key Variables Identified from Data Dictionary:**
- YEAR: Census year (2006-2016)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL/BPLD: Birthplace (200/20000 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter
- AGE: Age
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight
- SEX, EDUC, MARST: Demographic controls
- STATEFIP: State FIPS code

### Step 2: DACA Eligibility Criteria Definition
Per program requirements, DACA eligibility requires:
1. Arrived in US before 16th birthday → YRIMMIG - BIRTHYR < 16
2. Born after June 15, 1981 (not yet 31 as of June 15, 2012) → BIRTHYR >= 1982 (conservative)
3. Lived continuously in US since June 15, 2007 → YRIMMIG <= 2007
4. Not a citizen or legal resident → CITIZEN == 3 (Not a citizen)
5. Mexican-born and Hispanic-Mexican ethnicity → BPL == 200 AND HISPAN == 1

**Key Decision:** Since we cannot distinguish documented vs undocumented among non-citizens, we assume all non-citizens without naturalization are undocumented (as per instructions).

**Age Constraint:** Must have been at least 15 years old at time of application (could have been in high school or GED program). Given DACA implementation in 2012, practical working-age sample: ages 18-35 in outcome years.

### Step 3: Identification Strategy
**Approach:** Difference-in-Differences (DiD)

**Treatment Group:** Hispanic-Mexican, Mexican-born non-citizens who meet DACA eligibility criteria based on:
- Arrival age < 16
- Born 1982 or later (to ensure < 31 as of June 15, 2012)
- Arrived by 2007 (continuous presence requirement)

**Control Group:** Hispanic-Mexican, Mexican-born non-citizens who are similar but do NOT meet DACA eligibility:
- Option A: Arrived at age 16+ (age-at-arrival cutoff)
- Option B: Born before 1981 (age cutoff)
- Option C: Arrived after 2007 (continuous presence requirement not met)

**Chosen Strategy:** Use arrival age cutoff (arrived before vs. after age 16) as primary source of variation. This creates a natural comparison group with similar characteristics.

**Time Periods:**
- Pre-DACA: 2006-2011
- Post-DACA: 2013-2016
- Excluded: 2012 (implementation year - cannot distinguish before/after within year)

### Step 4: Sample Restrictions
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3)
4. Working age: 18-45 years
5. Years: 2006-2011 (pre) and 2013-2016 (post), excluding 2012

### Step 5: Outcome Variable
Full-time employment: UHRSWORK >= 35

### Step 6: Analysis Commands

```python
# See analysis_39.py for full code
```

---

## Key Analytical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Treatment definition | Arrival age < 16 | Natural eligibility cutoff from DACA rules |
| Control group | Arrival age >= 16 | Similar demographics, ineligible for DACA |
| Age range | 18-45 | Working-age adults likely to be affected |
| Birth year cutoff | >= 1982 | Ensures under 31 as of June 2012 |
| Continuous presence | Arrived <= 2007 | Required for DACA eligibility |
| Exclude 2012 | Yes | Cannot distinguish pre/post within year |
| Weights | PERWT | Survey weights for population inference |
| Standard errors | Clustered by state | Account for within-state correlation |

---

## Commands Executed

### Python Analysis Script
See: analysis_39.py

### LaTeX Report Compilation
```bash
pdflatex replication_report_39.tex
```

---

## Results Summary

### Main Finding
DACA eligibility increased the probability of full-time employment by **3.4 percentage points** (SE = 0.0039, p < 0.0001).

### Key Results
| Specification | Coefficient | Std. Error | 95% CI |
|---------------|-------------|------------|--------|
| Basic DiD | 0.0636 | 0.0067 | [0.051, 0.077] |
| + Demographics | 0.0438 | 0.0039 | [0.036, 0.052] |
| + State & Year FE (preferred) | 0.0343 | 0.0039 | [0.027, 0.042] |

### Sample Size
- Total DiD sample: 123,322 observations
- Pre-DACA (2006-2011): 70,988 observations
- Post-DACA (2013-2016): 52,334 observations

### Robustness Checks
- Narrow age-at-arrival window (12-19): 0.0469 (SE = 0.007)
- Employment as outcome: 0.0339 (SE = 0.006)
- Labor force participation: 0.0314 (SE = 0.006)
- Men only: 0.0275 (SE = 0.005)
- Women only: 0.0383 (SE = 0.007)

### Event Study
Pre-treatment coefficients (2006-2010) are not statistically significant, supporting parallel trends.
Post-treatment effects emerge in 2013 and strengthen through 2016.

---

## Files Generated

1. `analysis_39.py` - Main Python analysis script
2. `results_39.json` - Analysis results in JSON format
3. `figure1_event_study.png` - Event study plot
4. `figure2_trends.png` - Trends by eligibility status
5. `replication_report_39.tex` - LaTeX source for report
6. `replication_report_39.pdf` - Final PDF report (22 pages)
7. `run_log_39.md` - This log file

---

## Session End
Analysis completed successfully on 2026-01-25.

