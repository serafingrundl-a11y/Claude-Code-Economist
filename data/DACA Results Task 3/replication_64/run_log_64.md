# Run Log for DACA Replication Study (ID: 64)

## Overview
This log documents the commands executed and key decisions made during the independent replication of the DACA full-time employment study.

---

## 1. Initial Setup and Data Exploration

### Reading Instructions
- Read `replication_instructions.docx` to understand the research question and methodology
- Research question: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals

### Data Files Examined
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Labelled version (not used)
- `data/acs_data_dict.txt` - Data dictionary for variable definitions

### Initial Data Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
# Dataset: 17,382 observations, 105 variables
# Years: 2008-2011 (pre-DACA), 2013-2016 (post-DACA) - 2012 excluded
# Treatment (ELIGIBLE=1): 11,382 observations (ages 26-30 in June 2012)
# Control (ELIGIBLE=0): 6,000 observations (ages 31-35 in June 2012)
```

---

## 2. Key Decisions

### 2.1 Treatment and Control Group Definition
- **Decision:** Use the pre-constructed ELIGIBLE variable as instructed
- **Treatment:** Ages 26-30 in June 2012 (ELIGIBLE = 1)
- **Control:** Ages 31-35 in June 2012 (ELIGIBLE = 0)
- **Rationale:** As specified in instructions - do not create own eligibility variable

### 2.2 Outcome Variable
- **Decision:** Use FT variable as the outcome
- **Definition:** FT = 1 if usually working 35+ hours per week, 0 otherwise
- **Note:** Those not in the labor force are included as FT = 0

### 2.3 Time Period Definition
- **Decision:** Use the pre-constructed AFTER variable
- **Pre-DACA:** 2008-2011 (AFTER = 0)
- **Post-DACA:** 2013-2016 (AFTER = 1)
- **Note:** 2012 excluded due to timing uncertainty

### 2.4 Model Specification Choices

#### Baseline Model (Model 1)
- Simple DiD: FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
- Heteroskedasticity-robust standard errors (HC1)

#### Preferred Model (Model 4)
- DiD with year fixed effects + state fixed effects + individual covariates
- Covariates: SEX (recoded to MALE), MARST (recoded to MARRIED), AGE, NCHILD, EDUC_RECODE
- Year fixed effects: Separate dummies for each year
- State fixed effects: Separate dummies for each state (STATEFIP)
- Robust standard errors (HC1)

**Rationale for preferred specification:**
1. Year FEs account for common time trends affecting both groups
2. State FEs account for time-invariant differences across states
3. Covariates improve comparability between treatment and control groups
4. Robust SEs account for potential heteroskedasticity

### 2.5 Weighting Decision
- Provided both unweighted and weighted (PERWT) estimates
- Weighted estimates use ACS person weights for population representativeness
- Unweighted preferred in main specification for simplicity and interpretability

### 2.6 Standard Error Approach
- Used HC1 (heteroskedasticity-robust) standard errors throughout
- Did not cluster by state or year due to small number of clusters
- Alternative clustering approaches not implemented

---

## 3. Analysis Commands

### 3.1 Main Analysis Script (`analysis.py`)

```python
# Key analysis steps:

# 1. Load and prepare data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# 2. Simple DiD calculation
treated_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # 0.6263
treated_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()   # 0.6658
control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()  # 0.6697
control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()   # 0.6449
did_estimate = (treated_after - treated_before) - (control_after - control_before)  # 0.0643

# 3. Regression models
import statsmodels.api as sm

# Model 1: Basic DiD
model1 = sm.OLS(y, sm.add_constant(X1)).fit(cov_type='HC1')

# Model 4: Full specification (preferred)
model4 = sm.OLS(y, sm.add_constant(covariates_state_df)).fit(cov_type='HC1')
# Coefficient: 0.0546, SE: 0.0142, p < 0.001

# 4. Event study for parallel trends
# Reference year: 2011
# Pre-treatment coefficients tested for joint significance

# 5. Robustness checks
# - By gender
# - By marital status
# - Placebo test (fake treatment at 2010 using pre-period data only)
# - Narrower bandwidth (ages 28-33)
```

### 3.2 Figure Generation (`create_figures.py`)

```python
# Generated 4 figures:
# - figure1_parallel_trends.pdf: FT rates by year and eligibility
# - figure2_event_study.pdf: Event study coefficients
# - figure3_did_bars.pdf: DiD visualization (before/after bars)
# - figure4_coefficient_comparison.pdf: Estimates across specifications
```

---

## 4. Results Summary

### Main Finding
**Preferred estimate (Model 4):**
- Effect size: 0.055 (5.5 percentage points)
- Standard error: 0.014
- 95% CI: [0.027, 0.082]
- p-value: < 0.001
- Sample size: 17,382

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 5.5 percentage points among Mexican-born Hispanic individuals aged 26-30 compared to those aged 31-35.

### Robustness
- Results stable across model specifications (5.5-7.5 pp)
- Effects significant for males (6.2 pp) and marginally significant for females (4.5 pp)
- Similar effects for married (6.9 pp) and unmarried (6.6 pp)
- Placebo test: Not significant (p = 0.44)
- Narrower bandwidth: Still significant at 4.8 pp

### Parallel Trends Concerns
- Event study shows some significant pre-treatment coefficients (2008, 2010)
- Visual inspection suggests possible divergence in early pre-period
- Placebo test provides some reassurance
- Results should be interpreted with caution

---

## 5. Files Produced

### Required Deliverables
1. `replication_report_64.tex` - LaTeX source for report
2. `replication_report_64.pdf` - Compiled PDF report (18 pages)
3. `run_log_64.md` - This run log

### Supporting Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `analysis_results.py` - Saved results dictionary
- `model_summaries.txt` - Full regression output
- `trends_data.csv` - Trends data for figures
- `weighted_trends_data.csv` - Weighted trends data
- `figure1_parallel_trends.pdf/png`
- `figure2_event_study.pdf/png`
- `figure3_did_bars.pdf/png`
- `figure4_coefficient_comparison.pdf/png`

---

## 6. Software Environment

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (visualization)
- LaTeX (MiKTeX on Windows) for report compilation

---

## 7. Timestamps

- Analysis started: 2026-01-27
- Analysis completed: 2026-01-27
- Report compiled: 2026-01-27

---

## 8. Notes and Caveats

1. **ACS as repeated cross-section:** Cannot track individuals over time; estimates reflect population-level changes
2. **Intent-to-treat interpretation:** ELIGIBLE captures potential eligibility, not actual DACA receipt
3. **Age by construction:** Treatment and control differ by 5 years of age, which may confound trends
4. **Pre-trends concern:** Some evidence of differential trends in 2008-2010 period
5. **Binary variables coding:** IPUMS variables coded 1=No, 2=Yes; FT, AFTER, ELIGIBLE coded 0=No, 1=Yes
6. **Sampling weights:** Provided both weighted and unweighted estimates; slight differences in magnitude

---

*End of run log*
