# Run Log for DACA Replication Study (Replication 25)

## Project Overview
This document logs all commands, decisions, and key steps taken during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Session Start
**Date:** 2026-01-27

---

## Step 1: Reading and Understanding Instructions

### 1.1 Replication Instructions Summary
- **Research Question:** What was the causal impact of DACA eligibility on the probability of full-time employment (≥35 hours/week) among ethnically Hispanic-Mexican, Mexican-born people in the US?
- **Treatment Group:** Ages 26-30 at the time of DACA implementation (June 15, 2012)
- **Control Group:** Ages 31-35 at the time of DACA implementation (otherwise eligible if not for age)
- **Method:** Difference-in-Differences (DiD) comparing pre-treatment (2008-2011) to post-treatment (2013-2016)
- **Key Note:** 2012 data is omitted because treatment timing within the year is indeterminate

### 1.2 Key Variables Provided
- `ELIGIBLE`: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
- `FT`: 1 = full-time work (≥35 hours/week), 0 = not full-time
- `AFTER`: 1 = post-DACA years (2013-2016), 0 = pre-DACA years (2008-2011)
- `PERWT`: Person weight for ACS survey
- Additional covariates available for potential use

### 1.3 Analytical Decisions
1. Use provided `ELIGIBLE`, `FT`, and `AFTER` variables as specified
2. Do not drop any observations from the provided sample
3. Implement standard DiD specification: FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE × AFTER) + ε
4. The coefficient β₃ represents the treatment effect
5. Use survey weights (PERWT) for population-representative estimates
6. Consider robustness checks with covariates and standard error clustering

---

## Step 2: Data Exploration

### 2.1 Files Available
- `prepared_data_numeric_version.csv` - Numeric coded data (17,382 observations × 105 variables)
- `prepared_data_labelled_version.csv` - Labelled version
- `acs_data_dict.txt` - Data dictionary from IPUMS

### 2.2 Data Loading Command
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
```

### 2.3 Sample Characteristics
- **Total observations:** 17,382
- **Treatment group (ELIGIBLE=1):** 11,382 (65.5%)
- **Control group (ELIGIBLE=0):** 6,000 (34.5%)
- **Pre-DACA period (AFTER=0):** 9,527 observations
- **Post-DACA period (AFTER=1):** 7,855 observations
- **Full-time employed (FT=1):** 11,283 (64.9%)

### 2.4 Year Distribution
| Year | N |
|------|--------|
| 2008 | 2,354 |
| 2009 | 2,379 |
| 2010 | 2,444 |
| 2011 | 2,350 |
| 2013 | 2,124 |
| 2014 | 2,056 |
| 2015 | 1,850 |
| 2016 | 1,825 |

---

## Step 3: Analysis Implementation

### 3.1 Python Scripts Created
1. **`daca_analysis.py`** - Main analysis script containing:
   - Data loading and exploration
   - Summary statistics computation
   - Main DiD regression (7 specifications)
   - Heterogeneity analysis by sex
   - Pre-trend analysis
   - Results export

2. **`create_figures.py`** - Visualization script creating:
   - Figure 1: Parallel trends plot
   - Figure 2: DiD illustration (2x2 bar chart)
   - Figure 3: Coefficient plot across specifications
   - Figure 4: Sample distribution
   - Figure 5: Heterogeneity by sex
   - Figure 6: Pre-trends visualization

---

## Step 4: Key Analytical Decisions

### Decision 1: Model Specification
- **Primary specification:** OLS with interaction term (ELIGIBLE × AFTER)
- **Justification:** Standard DiD approach; interpretable coefficients; widely accepted methodology

### Decision 2: Survey Weights
- **Choice:** Include person weights (PERWT) in preferred specification
- Also report unweighted results for comparison
- **Justification:** ACS is a complex probability sample; weights produce population-representative estimates that reflect the target population rather than the sample

### Decision 3: Standard Errors
- **Primary:** Heteroskedasticity-robust standard errors (HC1)
- **Robustness:** State-clustered standard errors (51 clusters)
- **Justification:**
  - HC1 accounts for potential heteroskedasticity in a linear probability model
  - State clustering accounts for within-state correlation in errors

### Decision 4: Covariates
- **Base model:** No covariates (pure DiD)
- **Extended model:** Include FEMALE, AGE, MARRIED, education dummies (HS, Some College, Two-Year, BA+)
- **Justification:**
  - Base model relies on DiD's ability to difference out time-invariant heterogeneity
  - Covariates may improve precision but risk controlling for post-treatment variables
  - Preferred specification uses no covariates to avoid potential bias

### Decision 5: Sample
- **Choice:** Use full provided sample without restrictions
- **Justification:** Instructions explicitly state not to further limit the sample

### Decision 6: Preferred Specification Selection
- **Choice:** Model 2 (Weighted OLS with robust SE, no covariates)
- **Justification:**
  1. Survey weights ensure population representativeness
  2. Parsimonious specification avoids potentially endogenous controls
  3. Robust SE account for non-constant variance
  4. DiD design handles time-invariant confounders

---

## Step 5: Results Summary

### 5.1 Full-Time Employment Rates (2x2 Table)

|                     | Pre-DACA (2008-2011) | Post-DACA (2013-2016) | Change |
|---------------------|----------------------|-----------------------|--------|
| Control (31-35)     | 66.97%              | 64.49%                | -2.48pp |
| Treatment (26-30)   | 62.63%              | 66.58%                | +3.95pp |
| **Difference**      | -4.34pp             | +2.09pp               | **+6.43pp** |

### 5.2 Regression Results Summary

| Model | Specification | Estimate | SE | 95% CI | p-value |
|-------|--------------|----------|-----|--------|---------|
| (1) | Basic OLS, Robust SE | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| **(2)** | **Weighted, Robust SE (Preferred)** | **0.0748** | **0.0181** | **[0.039, 0.110]** | **<0.001** |
| (3) | With Covariates, Robust SE | 0.0536 | 0.0142 | [0.026, 0.081] | <0.001 |
| (4) | Weighted + Covariates | 0.0625 | 0.0167 | [0.030, 0.095] | <0.001 |
| (5) | State-Clustered SE | 0.0643 | 0.0141 | [0.037, 0.092] | <0.001 |
| (6) | Clustered + Covariates | 0.0536 | 0.0148 | [0.025, 0.083] | <0.001 |
| (7) | Year Fixed Effects | 0.0629 | 0.0152 | [0.033, 0.093] | <0.001 |

### 5.3 Heterogeneity by Sex
- **Males:** 0.0615 (SE=0.017), p<0.001, N=9,075
- **Females:** 0.0452 (SE=0.023), p=0.051, N=8,307

### 5.4 Pre-Trend Test
- Eligible × Year trend coefficient: 0.0151 (SE=0.0091, p=0.098)
- No statistically significant differential pre-trend at 5% level

### 5.5 Preferred Estimate (for reporting)
- **Effect Size:** 0.0748 (7.48 percentage points)
- **Standard Error:** 0.0181
- **95% CI:** [0.0393, 0.1102]
- **Sample Size:** 17,382 (unweighted); 2,416,349 (weighted)
- **p-value:** <0.001

---

## Step 6: Report Generation

### 6.1 LaTeX Report
- **Source file:** `replication_report_25.tex`
- **Output file:** `replication_report_25.pdf`
- **Length:** 23 pages
- **Contents:**
  - Abstract
  - Table of Contents
  - Introduction
  - Background on DACA
  - Data
  - Empirical Methodology
  - Results
  - Additional Analyses
  - Discussion
  - Conclusion
  - Appendices (Figures, Full Regression Output, Variable Definitions, Results Summary)

### 6.2 Figures Generated
1. `figure1_parallel_trends.png/pdf` - Employment trends by year and group
2. `figure2_did_illustration.png/pdf` - 2x2 DiD bar chart
3. `figure3_coefficient_plot.png/pdf` - Treatment effects across specifications
4. `figure4_sample_distribution.png/pdf` - Sample size by year and age distribution
5. `figure5_heterogeneity_sex.png/pdf` - Effects by sex
6. `figure6_pre_trends.png/pdf` - Pre-period difference trends

### 6.3 Supplementary Data Files
- `regression_results.csv` - All model estimates
- `employment_rates.csv` - Employment rates by group and period
- `yearly_trends.csv` - Year-by-year employment rates

---

## Commands Executed

```bash
# Data analysis
cd "C:\Users\seraf\DACA Results Task 3\replication_25"
python daca_analysis.py

# Figure generation
python create_figures.py

# LaTeX compilation (run 3 times for TOC and references)
pdflatex -interaction=nonstopmode replication_report_25.tex
pdflatex -interaction=nonstopmode replication_report_25.tex
pdflatex -interaction=nonstopmode replication_report_25.tex
```

---

## Deliverables Produced

| File | Description | Status |
|------|-------------|--------|
| `replication_report_25.tex` | LaTeX source for report | ✓ Complete |
| `replication_report_25.pdf` | Compiled PDF report (23 pages) | ✓ Complete |
| `run_log_25.md` | This run log | ✓ Complete |
| `daca_analysis.py` | Main analysis script | ✓ Complete |
| `create_figures.py` | Figure generation script | ✓ Complete |
| `figure1-6_*.png/pdf` | Visualization figures | ✓ Complete |
| `*.csv` | Supplementary data tables | ✓ Complete |

---

## Interpretation and Conclusion

The analysis finds that DACA eligibility increased the probability of full-time employment by approximately **7.5 percentage points** (preferred estimate: 0.0748, 95% CI: [0.039, 0.110]). This effect is:

1. **Statistically significant** at the 1% level across all specifications
2. **Robust** to different modeling choices (estimates range from 5.36 to 7.48 pp)
3. **Larger for males** (6.15 pp) than females (4.52 pp)
4. **Consistent with parallel trends** assumption (no significant differential pre-trend)

The positive effect is consistent with DACA achieving its intended purpose of expanding labor market access through legal work authorization.

---

## Session End
**Date:** 2026-01-27
**Status:** All deliverables complete
