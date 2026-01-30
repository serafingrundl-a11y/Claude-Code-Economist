# Run Log - DACA Replication Study (Session 91)

## Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Date: 2026-01-25

---

## Step 1: Read and Understand Instructions

**Time:** Start of session

**Actions:**
- Read replication_instructions.docx
- Key research question: Effect of DACA eligibility (treatment) on full-time employment (outcome)
- Full-time employment defined as: usually working 35+ hours per week
- Target population: Ethnically Hispanic-Mexican, Mexican-born people in the US
- DACA implemented June 15, 2012
- Analysis period for effects: 2013-2016

**DACA Eligibility Criteria (per instructions):**
1. Arrived in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

---

## Step 2: Explore Data Files

**Time:** After reading instructions

**Files available:**
- data.csv: Main ACS data file (33,851,424 observations)
- acs_data_dict.txt: Data dictionary with variable definitions
- state_demo_policy.csv: Optional state-level data (not used)
- State Level Data Documentation.docx: Documentation for state data

**Key Variables Identified:**
- YEAR: Census year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- HISPAN/HISPAND: Hispanic origin (1=Mexican)
- BPL/BPLD: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week (35+ = full-time)
- EMPSTAT: Employment status (1=Employed)
- AGE: Age at time of survey
- PERWT: Person weight for statistical analysis

---

## Step 3: Sample Construction

**Initial Data:**
- Full ACS 2006-2016: 33,851,424 observations

**Filtering Steps:**
1. Hispanic-Mexican (HISPAN=1) AND Mexican-born (BPL=200): 991,261 observations
2. Non-citizens (CITIZEN=3): 701,347 observations
3. Working age (18-64) and exclude 2012: 547,614 observations
4. DiD sample (treatment + control): 113,154 observations

**Treatment Group (DACA-eligible):** 71,347 observations
- Arrived before age 16 (age_at_arrival < 16)
- Age <= 30 as of June 15, 2012
- Arrived by 2007 (YRIMMIG <= 2007)

**Control Group:** 41,807 observations
- Same arrival criteria (before age 16, arrived by 2007)
- Age 31-45 as of June 15, 2012 (too old for DACA)

---

## Step 4: Variable Definitions

**Outcome Variable:**
```python
fulltime = (EMPSTAT == 1) & (UHRSWORK >= 35)
```

**Treatment Variable:**
```python
daca_eligible = (age_at_arrival < 16) & (age_june2012 <= 30) & (YRIMMIG <= 2007)
```

**Age Calculation:**
```python
age_june2012 = 2012 - BIRTHYR
if BIRTHQTR >= 3:
    age_june2012 -= 1  # Birthday hadn't occurred by June 15
```

---

## Step 5: Descriptive Statistics

**Pre-treatment Full-time Employment Rates (weighted):**
- Treatment (DACA eligible): 46.5%
- Control (Age 31-45): 62.5%

**Post-treatment Full-time Employment Rates (weighted):**
- Treatment (DACA eligible): 52.5%
- Control (Age 31-45): 61.4%

**Simple DiD Calculation:**
- Treatment change: +6.0 pp
- Control change: -1.1 pp
- DiD estimate: +7.1 pp

---

## Step 6: Regression Analysis

### Model Specifications:

**Model 1: Basic DiD**
```
fulltime ~ treated + post + treated*post
```
- DiD effect: +0.071 (SE=0.008, p<0.001)

**Model 2: DiD with Demographic Controls**
```
fulltime ~ treated + post + treated*post + age + age^2 + female + married + educ_hs
```
- DiD effect: -0.012 (SE=0.008, p=0.132)

**Model 3: DiD with State and Year Fixed Effects (PREFERRED)**
```
fulltime ~ treated + treated*post + age + age^2 + female + married + educ_hs + state_FE + year_FE
```
- DiD effect: -0.020 (SE=0.008, p=0.011)
- 95% CI: [-0.035, -0.005]

---

## Step 7: Robustness Checks

| Specification | Effect | SE | p-value |
|---------------|--------|-----|---------|
| Main (full-time, with FE) | -0.020 | 0.008 | 0.011 |
| Any employment | +0.001 | 0.007 | 0.927 |
| Men only | -0.054 | 0.010 | <0.001 |
| Women only | +0.013 | 0.012 | 0.309 |
| Narrower control (31-40) | -0.042 | 0.009 | <0.001 |

---

## Step 8: Validity Checks

### Event Study (Reference year: 2011)
| Year | Coefficient | SE | Significant? |
|------|-------------|-----|--------------|
| 2006 | +0.040 | 0.016 | ** |
| 2007 | +0.032 | 0.015 | ** |
| 2008 | +0.034 | 0.015 | ** |
| 2009 | +0.023 | 0.015 | |
| 2010 | +0.027 | 0.015 | * |
| 2011 | 0 | - | (ref) |
| 2013 | +0.006 | 0.016 | |
| 2014 | +0.014 | 0.016 | |
| 2015 | -0.004 | 0.016 | |
| 2016 | -0.006 | 0.016 | |

**Finding:** Significant pre-trends in 2006-2008 suggest violation of parallel trends assumption.

### Placebo Test (2009 as fake implementation)
- Placebo DiD: -0.031 (SE=0.009, p<0.001)
- **Finding:** Significant placebo effect confirms pre-trend concerns.

---

## Step 9: Key Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Use DiD design | Standard approach for policy evaluation with pre/post and treatment/control |
| Exclude 2012 | DACA implemented mid-year (June 15), cannot distinguish pre/post |
| Define full-time as UHRSWORK >= 35 | Per instructions |
| Use CITIZEN=3 as proxy for undocumented | Per instructions: non-citizens without papers assumed undocumented |
| Control group: ages 31-45 | Same arrival patterns but too old for DACA; provides comparable group |
| Include state and year FE | Controls for time-invariant state factors and common shocks |
| Use person weights (PERWT) | Required for population-representative estimates |
| Robust standard errors (HC1) | Accounts for heteroskedasticity |

---

## Step 10: Final Results

**Preferred Estimate (Model 3):**
- DiD Effect: -0.020 (2.0 percentage point decrease)
- Standard Error: 0.008
- 95% CI: [-0.035, -0.005]
- p-value: 0.011
- Sample Size: 113,154

**Interpretation:**
DACA eligibility is associated with a 2.0 percentage point decrease in full-time employment probability. However, this finding should be interpreted with caution due to evidence of pre-trends suggesting the parallel trends assumption may not hold.

---

## Output Files Created

1. **analysis.py** - Main analysis script
2. **create_figures.py** - Figure generation script
3. **results_summary.csv** - Numerical results
4. **balance_table.csv** - Pre-treatment balance statistics
5. **event_study.csv** - Event study coefficients
6. **event_study_plot.png/pdf** - Event study figure
7. **pre_post_comparison.png/pdf** - Pre-post comparison figure
8. **trends_plot.png/pdf** - Employment trends figure
9. **replication_report_91.tex** - LaTeX report source
10. **replication_report_91.pdf** - Final report (20 pages)
11. **run_log_91.md** - This log file

---

## Software and Packages Used

- Python 3.14
- pandas (data manipulation)
- numpy (numerical computing)
- statsmodels (regression analysis)
- matplotlib (visualization)
- LaTeX/pdflatex (document preparation)

---

## Session Complete

All required deliverables have been produced:
- [x] replication_report_91.tex
- [x] replication_report_91.pdf
- [x] run_log_91.md
