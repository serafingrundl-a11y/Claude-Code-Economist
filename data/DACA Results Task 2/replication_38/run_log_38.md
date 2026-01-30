# Run Log - DACA Replication Study (ID: 38)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

## Study Design
- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at DACA implementation (would have been eligible but for age)
- **Method**: Difference-in-Differences estimation
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 due to mid-year implementation)

---

## Session Log

### Step 1: Data Exploration
**Date/Time**: Session start

**Files examined**:
- `data/data.csv` - Main ACS data file (~6.3GB)
- `data/acs_data_dict.txt` - Data dictionary
- `data/state_demo_policy.csv` - Optional state-level data (not used)

**Key Variables Identified**:
| Variable | Description | Usage |
|----------|-------------|-------|
| YEAR | Census year | Time period indicator |
| HISPAN | Hispanic origin (1=Mexican) | Sample selection |
| BPL | Birthplace (200=Mexico) | Sample selection |
| CITIZEN | Citizenship status (3=Not a citizen) | Sample selection |
| YRIMMIG | Year of immigration | DACA eligibility |
| BIRTHYR | Birth year | Age group determination |
| BIRTHQTR | Quarter of birth | Age calculation refinement |
| AGE | Age at survey | Additional check |
| UHRSWORK | Usual hours worked/week | Outcome (35+ = full-time) |
| EMPSTAT | Employment status | Employment indicator |
| PERWT | Person weight | Survey weights |
| SEX | Sex | Control variable |
| MARST | Marital status | Control variable |
| EDUCD | Education (detailed) | Control variable |
| STATEFIP | State FIPS code | State fixed effects |

### Step 2: Analysis Script Development
Created `analysis.py` with the following structure:
1. Data loading with chunked reading (500,000 rows per chunk)
2. Sample filtering by DACA eligibility criteria
3. Variable construction (age at June 2012, treatment indicator, outcome)
4. Descriptive statistics
5. Main DID regression models
6. Robustness checks
7. Event study analysis
8. Results export

### Step 3: Running Analysis
**Command executed**:
```bash
python analysis.py
```

**Sample Construction Results**:
- Total Hispanic-Mexican born in Mexico: 991,261
- After non-citizen filter: 701,347
- After age 26-35 filter: 181,229
- After arrived before age 16 filter: 46,817
- After residence since 2007 filter: 46,817
- After excluding 2012: 42,689
- **Final Sample**: 42,689 observations

**Group Sizes**:
- Treatment group (ages 26-30): 25,174
- Control group (ages 31-35): 17,515

### Step 4: Results

**Main Finding (Preferred Specification)**:
- DID Estimate: **0.043** (4.3 percentage points)
- Standard Error: 0.0107
- 95% CI: [0.022, 0.064]
- P-value: 0.0001
- Sample Size: 42,689

**Simple DID Calculation** (weighted means):
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.672 | 0.643 | -0.029 |
| **DID** | | | **0.058** |

**Robustness Checks**:
| Specification | Estimate | SE |
|---------------|----------|-----|
| Main (weighted, state FE) | 0.043 | 0.011 |
| Clustered SE by state | 0.043 | 0.010 |
| Narrow bandwidth (27-29 vs 32-34) | 0.037 | 0.014 |
| Any employment outcome | 0.042 | 0.010 |
| Males only | 0.033 | 0.013 |
| Females only | 0.047 | 0.018 |
| Placebo (2009 cutoff) | -0.001 | 0.013 |

**Event Study** (relative to 2011):
| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | 0.010 | 0.023 |
| 2007 | -0.029 | 0.022 |
| 2008 | 0.015 | 0.023 |
| 2009 | -0.000 | 0.024 |
| 2010 | -0.005 | 0.023 |
| 2013 | 0.040 | 0.024 |
| 2014 | 0.038 | 0.025 |
| 2015 | 0.023 | 0.025 |
| 2016 | 0.070 | 0.025 |

### Step 5: Report Generation
Created LaTeX report (`replication_report_38.tex`) and compiled to PDF.

**Commands**:
```bash
pdflatex -interaction=nonstopmode replication_report_38.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_38.tex  # Second pass (references)
pdflatex -interaction=nonstopmode replication_report_38.tex  # Third pass (final)
```

Output: `replication_report_38.pdf` (19 pages)

---

## Key Decisions

1. **Exclusion of 2012**: The ACS does not distinguish months of data collection, so 2012 observations cannot be cleanly categorized as pre/post DACA.

2. **Age Calculation**: Age as of June 15, 2012 calculated from BIRTHYR and BIRTHQTR. For those born in Q3/Q4, subtract 1 year since their birthday would be after June 15.

3. **Full-time Employment Definition**: UHRSWORK >= 35 hours per week, consistent with standard BLS definition.

4. **Immigration Timing**: Using YRIMMIG <= 2007 as proxy for continuous US residence since June 2007.

5. **Arrived Before 16**: Calculated as (YRIMMIG - BIRTHYR) < 16.

6. **Undocumented Status**: Per instructions, assume non-citizens without papers (CITIZEN == 3) are undocumented.

7. **Model Specification**: Weighted linear probability model (WLS) with HC1 robust standard errors, including state and year fixed effects, with controls for sex, marital status, and education.

8. **Survey Weights**: Used PERWT (ACS person weights) for population-representative estimates.

---

## Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `summary_statistics.csv` | Descriptive statistics by group/period |
| `event_study_results.csv` | Event study coefficients |
| `yearly_means.csv` | Full-time employment rates by year and treatment |
| `analysis_results.json` | Key results in JSON format |
| `replication_report_38.tex` | LaTeX source for report |
| `replication_report_38.pdf` | Final PDF report (19 pages) |
| `run_log_38.md` | This run log |

---

## Commands Executed

```bash
# Data exploration
head -n 5 data/data.csv
ls -la data/

# Analysis execution
python analysis.py

# LaTeX compilation
pdflatex -interaction=nonstopmode replication_report_38.tex
pdflatex -interaction=nonstopmode replication_report_38.tex
pdflatex -interaction=nonstopmode replication_report_38.tex

# Verification
ls -la *.pdf *.tex *.md
```

---

## Summary

**Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants.

**Method**: Difference-in-differences comparing ages 26-30 (treated) vs. 31-35 (control).

**Main Result**: DACA eligibility increased full-time employment by 4.3 percentage points (p < 0.001).

**Robustness**: Result is robust to clustering, alternative bandwidths, and placebo tests. Pre-trends are flat.

**Interpretation**: DACA's work authorization provision substantially improved labor market outcomes for eligible immigrants.
