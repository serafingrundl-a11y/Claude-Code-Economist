# Run Log - DACA Replication Study #59

## Date: January 27, 2026

---

## Overview

This document logs all commands and key decisions made during the independent replication of the DACA effect on full-time employment study.

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions

**Command:** Extracted text from `replication_instructions.docx` using Python's python-docx library.

**Key Information Extracted:**
- Research Question: Effect of DACA eligibility on full-time employment among Hispanic Mexican-born individuals
- Treatment Group: Ages 26-30 in June 2012 (ELIGIBLE=1)
- Control Group: Ages 31-35 in June 2012 (ELIGIBLE=0)
- Outcome: FT variable (1=full-time, 0=not full-time)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
- 2012 data excluded (implementation year)

### 1.2 Data Files Identified

```
data/prepared_data_labelled_version.csv
data/prepared_data_numeric_version.csv
data/acs_data_dict.txt
```

### 1.3 Data Loading

**Command:**
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```

**Result:**
- Shape: 17,382 observations, 105 variables
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-DACA period: 9,527 observations
- Post-DACA period: 7,855 observations

---

## 2. Key Analytical Decisions

### 2.1 Choice of Estimator

**Decision:** Linear Probability Model (LPM) with OLS

**Rationale:**
- LPM coefficients are directly interpretable as percentage point changes
- Consistent with difference-in-differences literature
- Robust to heteroskedasticity with appropriate standard errors
- Logit model run as robustness check confirms similar results

### 2.2 Standard Errors

**Decision:** Heteroskedasticity-robust standard errors (HC1) as baseline

**Rationale:**
- Binary outcome variable implies heteroskedasticity by construction
- HC1 (White's robust SE) provides consistent standard errors
- Clustered SEs by state run as robustness check

### 2.3 Control Variables

**Decision:** Include demographic controls, education, state FE, and year FE in preferred specification

**Included Controls:**
- AGE (continuous)
- FEMALE (derived from SEX variable, 1=female, 0=male)
- MARRIED (derived from MARST, 1=married, 0=not married)
- Education dummies: ED_HS, ED_SOMECOLLEGE, ED_TWOYEAR, ED_BA (reference: Less than HS)
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

**Rationale:**
- Demographic controls improve precision and control for compositional differences
- State FE control for time-invariant state-level factors
- Year FE control for common time trends
- Not necessary for identification but improve efficiency

### 2.4 Sample Definition

**Decision:** Use entire provided dataset without further sample restrictions

**Rationale:**
- Instructions explicitly state: "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics."
- Includes individuals not in labor force (with FT=0)

### 2.5 Treatment of Survey Weights

**Decision:** Report both unweighted and weighted estimates; prefer unweighted for inference

**Rationale:**
- Unweighted OLS provides consistent estimates under DiD assumptions
- Weighted estimates shown for population representativeness
- Both yield similar results (unweighted: 0.052, weighted: 0.062)

---

## 3. Analysis Commands

### 3.1 Main Analysis Script

Created `analysis.py` containing:

```python
# Key model specifications

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()

# Model 8 (PREFERRED): Full specification
model8 = smf.ols('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED +
                  ED_HS + ED_SOMECOLLEGE + ED_TWOYEAR + ED_BA +
                  C(STATEFIP) + C(YEAR)', data=df).fit(cov_type='HC1')
```

**Execution:**
```bash
python analysis.py 2>&1 | tee analysis_output.txt
```

### 3.2 Figure Generation

Created `create_figures.py` generating:
- figure1_trends.png: Time series of FT employment by group
- figure2_prepost.png: Pre/post comparison bar chart
- figure3_eventstudy.png: Event study coefficients
- figure4_robustness.png: Robustness across specifications
- figure5_heterogeneity.png: Heterogeneous effects

**Execution:**
```bash
python create_figures.py
```

### 3.3 LaTeX Compilation

```bash
pdflatex -interaction=nonstopmode replication_report_59.tex
pdflatex -interaction=nonstopmode replication_report_59.tex
pdflatex -interaction=nonstopmode replication_report_59.tex
```

(Run three times for proper cross-references)

---

## 4. Results Summary

### 4.1 Simple DiD Calculation

| | Pre-DACA | Post-DACA | Difference |
|---|----------|-----------|------------|
| Treatment (26-30) | 0.626 | 0.666 | +0.039 |
| Control (31-35) | 0.670 | 0.645 | -0.025 |
| **DiD** | | | **+0.064** |

### 4.2 Regression Results

| Specification | DiD Coefficient | SE | 95% CI |
|---------------|-----------------|-----|--------|
| Basic DiD | 0.0643 | 0.0153 | [0.034, 0.094] |
| + Demographics | 0.0553 | 0.0142 | [0.028, 0.083] |
| + Education | 0.0535 | 0.0142 | [0.026, 0.081] |
| + State FE | 0.0639 | 0.0153 | [0.034, 0.094] |
| + State & Year FE | 0.0626 | 0.0152 | [0.033, 0.092] |
| **PREFERRED** | **0.0520** | **0.0141** | **[0.024, 0.080]** |

### 4.3 Robustness Checks

| Check | Result |
|-------|--------|
| Clustered SE by state | DiD = 0.054 (SE = 0.015) |
| WLS with PERWT | DiD = 0.062 (SE = 0.014) |
| Logit (marginal effect) | ME = 0.063 |
| Placebo pre-trends | DiD = 0.016 (p = 0.44, n.s.) |

### 4.4 Heterogeneity

| Subgroup | DiD Coefficient | SE |
|----------|-----------------|-----|
| Male | 0.062 | 0.017 |
| Female | 0.045 | 0.023 |
| HS Only | 0.048 | 0.018 |
| Some College | 0.108 | 0.038 |
| Married | 0.059 | 0.021 |
| Not Married | 0.076 | 0.022 |

---

## 5. Preferred Estimate

**Model:** Linear Probability Model with DiD, robust standard errors
**Controls:** Age, sex, marital status, education, state FE, year FE

| Metric | Value |
|--------|-------|
| Effect Size | 0.052 (5.2 percentage points) |
| Standard Error | 0.014 |
| 95% CI | [0.024, 0.080] |
| p-value | < 0.001 |
| Sample Size | 17,382 |

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 5.2 percentage points among the eligible population, compared to similar individuals who were just above the age cutoff.

---

## 6. Output Files

### 6.1 Required Deliverables
- `replication_report_59.tex` - LaTeX source file
- `replication_report_59.pdf` - Compiled PDF report (22 pages)
- `run_log_59.md` - This file

### 6.2 Supporting Files
- `analysis.py` - Main analysis script
- `analysis_output.txt` - Full analysis output
- `create_figures.py` - Figure generation script
- `results_summary.csv` - Summary of all model results
- `figure1_trends.png` - Employment trends figure
- `figure2_prepost.png` - Pre-post comparison figure
- `figure3_eventstudy.png` - Event study figure
- `figure4_robustness.png` - Robustness check figure
- `figure5_heterogeneity.png` - Heterogeneity figure

---

## 7. Notes on Variable Coding

Per instructions, IPUMS binary variables are coded:
- 1 = No, 2 = Yes

Custom variables (FT, AFTER, ELIGIBLE, state policy variables) are coded:
- 0 = No, 1 = Yes

Variable derivations:
- FEMALE = 1 if SEX == 2, 0 otherwise
- MARRIED = 1 if MARST in [1, 2], 0 otherwise

---

## 8. Software Environment

- Python 3.x with pandas, numpy, statsmodels, scipy, matplotlib
- LaTeX with pdflatex (MiKTeX distribution)
- Windows operating system

---

## End of Run Log
