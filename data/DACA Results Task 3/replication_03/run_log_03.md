# Run Log - DACA Replication Study 03

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA (Deferred Action for Childhood Arrivals) impact study on full-time employment.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (35+ hours/week)?

**Treatment Group:** Ages 26-30 at policy implementation (June 2012)
**Control Group:** Ages 31-35 at policy implementation (June 2012)
**Pre-period:** 2008-2011
**Post-period:** 2013-2016

---

## Session Start
Date: 2026-01-26

---

## Step 1: Data Exploration

### 1.1 Load and inspect data structure
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(df.shape)  # (17382, 105)
```

**Key findings:**
- 17,382 observations, 105 variables
- Pre-constructed variables: ELIGIBLE (treatment indicator), AFTER (post-treatment), FT (full-time employment)
- Years: 2008-2011 (pre), 2013-2016 (post) - 2012 excluded
- AGE_IN_JUNE_2012 ranges from 26-35 as expected

### 1.2 Cross-tabulation of treatment groups
```python
pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
```
| ELIGIBLE | Before (0) | After (1) | Total |
|----------|------------|-----------|-------|
| 0 (31-35)| 3,294      | 2,706     | 6,000 |
| 1 (26-30)| 6,233      | 5,149     | 11,382|
| Total    | 9,527      | 7,855     | 17,382|

### 1.3 Full-time employment rates by group
```python
df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
```
| Group | Before | After | Change |
|-------|--------|-------|--------|
| Control (31-35) | 66.97% | 64.49% | -2.48% |
| Treated (26-30) | 62.63% | 66.58% | +3.94% |

**Raw DID:** 3.94% - (-2.48%) = **6.43 percentage points**

---

## Step 2: Identification Strategy

### Decision: Difference-in-Differences (DID)

**Rationale:**
1. Clear pre/post treatment periods (2012 policy implementation)
2. Natural comparison group based on age eligibility cutoff
3. Standard approach for policy evaluation with repeated cross-sections
4. Identification assumes parallel trends in absence of treatment

**Model Specification:**
```
FT_it = β₀ + β₁·ELIGIBLE_i + β₂·AFTER_t + β₃·(ELIGIBLE_i × AFTER_t) + X'_i·γ + ε_it
```

Where β₃ is the DID estimator of interest.

---

## Step 3: Model Estimation

### 3.1 Basic DID (no covariates)
```python
import statsmodels.formula.api as smf
model_basic = smf.ols('FT ~ ELIGIBLE * AFTER', data=df).fit()
```
**Result:** DID = 0.0643 (SE = 0.0153, p < 0.001)

### 3.2 DID with demographic covariates
**Covariates included:**
- SEX (gender) - coded as FEMALE dummy
- AGE (continuous age) + AGE² (quadratic term)
- MARRIED (marital status indicator)
- EDUC_RECODE (education categories: HS, Some College, AA, BA+)
- NCHILD (number of children)

**Result:** DID = 0.0590 (SE = 0.0193, p = 0.002)

### 3.3 DID with demographic + state policy controls
**Additional controls:**
- State policy variables (DRIVERSLICENSES, INSTATETUITION, EVERIFY, OMNIBUS)
- State economic conditions (LFPR, UNEMP)

**Result:** DID = 0.0254 (SE = 0.0197, p = 0.196) - NOT significant with individual policy controls

### 3.4 DID with State Fixed Effects
**Result:** DID = 0.0575 (SE = 0.0146, p < 0.001)

### 3.5 Weighted estimation (PREFERRED SPECIFICATION)
- Used PERWT (person weights) for population-representative estimates
- State fixed effects
- Clustered standard errors at state level

**Result:** DID = 0.0712 (SE = 0.0206, 95% CI: [0.0308, 0.1115], p = 0.0006)

---

## Step 4: Robustness Checks

### 4.1 Pre-trend analysis (Event Study)
Examined year-by-year effects relative to 2011 (reference year):

| Year | Coefficient | Std. Error | p-value |
|------|-------------|------------|---------|
| 2008 | -0.0451     | 0.0279     | 0.105   |
| 2009 | -0.0338     | 0.0251     | 0.179   |
| 2010 | -0.0694     | 0.0330     | 0.036   |
| 2013 | +0.0038     | 0.0345     | 0.912   |
| 2014 | -0.0307     | 0.0205     | 0.134   |
| 2015 | -0.0314     | 0.0356     | 0.377   |
| 2016 | +0.0342     | 0.0329     | 0.298   |

**Note:** Pre-trend coefficients are negative but mostly insignificant (except 2010). Some concern about parallel trends assumption.

### 4.2 Heterogeneity analysis

**By Gender:**
- Males: DID = 0.0631 (SE = 0.0237, p = 0.008)
- Females: DID = 0.0690 (SE = 0.0281, p = 0.014)

**By Education:**
- HS or less: DID = 0.0665 (SE = 0.0207, p = 0.001)
- Some college+: DID = 0.0903 (SE = 0.0428, p = 0.035)

### 4.3 Logit model for comparison
- Logit coefficient: 0.282
- Odds Ratio: 1.325
- Approximate marginal effect: 0.064

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Estimator | OLS (Linear Probability Model) | Ease of interpretation of DID coefficient; robust to model specification |
| Sample | Full sample as provided (N=17,382) | Per instructions: do not drop individuals |
| Weights | Used PERWT in preferred specification | Population representativeness |
| Standard Errors | Clustered by state | Account for within-state correlation |
| Covariates | Demographics + State FE | Improve precision, control for confounders while avoiding over-specification |
| Reference year | 2011 (for event study) | Last pre-treatment year |

---

## Main Results Summary

| Model | DID Coefficient | Std. Error | p-value | N | R² |
|-------|-----------------|------------|---------|---|----|
| (1) Basic DID | 0.0643 | 0.0153 | <0.001 | 17,382 | 0.002 |
| (2) + Demographics | 0.0590 | 0.0193 | 0.002 | 17,382 | 0.131 |
| (3) + State Policy | 0.0254 | 0.0197 | 0.196 | 17,382 | 0.135 |
| (4) + State FE | 0.0575 | 0.0146 | <0.001 | 17,382 | 0.134 |
| **(5) Weighted + State FE** | **0.0712** | **0.0206** | **0.001** | **17,382** | **0.135** |

---

## Preferred Estimate

**Effect Size:** 7.12 percentage points
**Standard Error:** 0.0206
**95% Confidence Interval:** [3.08, 11.15] percentage points
**p-value:** 0.0006
**Sample Size:** 17,382

**Interpretation:** DACA eligibility increased full-time employment by approximately 7.1 percentage points among the eligible population (ages 26-30 at policy implementation), compared to the control group (ages 31-35). This represents an approximately 11% relative increase from the pre-DACA baseline employment rate of 62.6%.

---

## Output Files

1. `replication_report_03.tex` - Full LaTeX report (~20 pages)
2. `replication_report_03.pdf` - Compiled PDF report (20 pages)
3. `run_log_03.md` - This file
4. `analysis_script.py` - Python script with full analysis
5. `create_figures.py` - Python script for figure generation
6. `figure1_trends.png/pdf` - Time trends figure
7. `figure2_eventstudy.png/pdf` - Event study figure
8. `figure3_did.png/pdf` - DID illustration figure
9. `figure4_heterogeneity.png/pdf` - Heterogeneity analysis figure
10. `results_summary.csv` - Summary results table

---

## Commands Executed

### Data Loading and Exploration
```bash
python -c "import pandas as pd; df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False); print(df.shape)"
```

### Full Analysis
```bash
python analysis_script.py
```

### Figure Generation
```bash
python create_figures.py
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_03.tex
pdflatex -interaction=nonstopmode replication_report_03.tex
pdflatex -interaction=nonstopmode replication_report_03.tex
```
(Three passes to resolve cross-references)

---

## Software Environment

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (visualizations)
- pdflatex (LaTeX compilation)

---

## Completion

Analysis completed: 2026-01-26
All deliverables generated and verified.
