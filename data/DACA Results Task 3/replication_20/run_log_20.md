# Run Log: DACA Replication Study (Replication 20)

## Project Overview
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the US
- **Treatment Group**: Individuals aged 26-30 at DACA implementation (June 2012)
- **Control Group**: Individuals aged 31-35 at DACA implementation
- **Outcome**: Full-time employment (FT = 1 if usually working 35+ hours/week)
- **Method**: Difference-in-Differences (DiD)
- **Data**: American Community Survey (ACS) 2008-2016, excluding 2012

---

## Key Decisions Made

### 1. Analytic Approach
- **Decision**: Use linear probability model (LPM) with difference-in-differences design
- **Rationale**: LPM coefficients are directly interpretable as marginal effects. DiD is the natural estimator given the quasi-experimental design with treatment defined by age eligibility.

### 2. Sample Definition
- **Decision**: Use entire provided dataset without further restrictions
- **Rationale**: Instructions explicitly state "do not further limit the sample by dropping individuals on the basis of their characteristics"

### 3. Treatment and Control Definition
- **Decision**: Use pre-constructed ELIGIBLE variable (1 = ages 26-30, 0 = ages 31-35 at June 2012)
- **Rationale**: Instructions specify to use the provided ELIGIBLE variable

### 4. Survey Weights
- **Decision**: Present both weighted (PERWT) and unweighted estimates; prefer weighted estimates
- **Rationale**: ACS is a complex survey design, weights ensure population representativeness

### 5. Standard Errors
- **Decision**: Use heteroskedasticity-robust (HC1) standard errors; also present clustered SE at state level
- **Rationale**: Binary outcome creates heteroskedasticity; state clustering addresses within-state correlation

### 6. Fixed Effects
- **Decision**: Include year and state fixed effects in main specification
- **Rationale**: Year FE control for common time trends; state FE control for time-invariant state differences

### 7. Covariates
- **Decision**: Include sex, age, marital status, number of children, and education in full specification
- **Rationale**: These are key predictors of employment that may improve precision and address potential confounders

### 8. Preferred Specification
- **Decision**: Model (6) - Weighted WLS with year FE, state FE, and individual covariates
- **Rationale**: Combines survey weights for representativeness, fixed effects for controlling confounders, and covariates for improved precision

---

## Commands and Analysis Sequence

### Step 1: Data Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(df.shape)  # (17382, 105)
print(df['YEAR'].value_counts().sort_index())
```

### Step 2: Basic DiD Calculation
```python
# Manual calculation
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
# Treated change: 0.6658 - 0.6263 = 0.0394
# Control change: 0.6449 - 0.6697 = -0.0248
# DiD = 0.0394 - (-0.0248) = 0.0643
```

### Step 3: Regression Models
```python
import statsmodels.formula.api as smf

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 2: With Year FE
model2 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 3: With Year + State FE
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 4: Full unweighted
model4 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + SEX_female + AGE + MARRIED + NCHILD + C(EDUC_RECODE)', data=df).fit(cov_type='HC1')

# Model 5: Weighted basic
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 6: Weighted full (PREFERRED)
model6 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + SEX_female + AGE + MARRIED + NCHILD + C(EDUC_RECODE)', data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 7: Clustered SE
model7 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

### Step 4: Robustness Checks
- Placebo test (fake treatment at 2010): Coef = 0.0171, p = 0.479 (non-significant supports parallel trends)
- Event study: Pre-treatment coefficients not significantly different from zero
- Heterogeneity by sex: Males 0.070, Females 0.048
- Logit marginal effect: 0.064 (consistent with LPM)

### Step 5: Visualization
```python
import matplotlib.pyplot as plt
# Created parallel_trends.png and event_study.png
```

---

## Main Results Summary

| Model | DiD Estimate | SE | 95% CI | N |
|-------|-------------|-----|--------|---|
| (1) Basic DiD | 0.0643*** | 0.0153 | [0.034, 0.094] | 17,382 |
| (2) + Year FE | 0.0629*** | 0.0152 | [0.033, 0.093] | 17,382 |
| (3) + State FE | 0.0626*** | 0.0152 | [0.033, 0.092] | 17,382 |
| (4) + Covariates | 0.0547*** | 0.0142 | [0.027, 0.082] | 17,379 |
| (5) Weighted Basic | 0.0748*** | 0.0181 | [0.039, 0.110] | 17,382 |
| (6) Weighted Full | 0.0615*** | 0.0166 | [0.029, 0.094] | 17,379 |
| (7) Clustered SE | 0.0710*** | 0.0202 | [0.031, 0.110] | 17,382 |

*** p < 0.001

### Preferred Estimate (Model 6)
- **Effect Size**: 0.0615 (6.15 percentage points)
- **Standard Error**: 0.0166
- **95% Confidence Interval**: [0.0289, 0.0941]
- **Sample Size**: 17,379
- **p-value**: 0.0002

---

## Files Generated
1. `run_log_20.md` - This file
2. `replication_report_20.tex` - LaTeX report
3. `replication_report_20.pdf` - Compiled PDF report
4. `parallel_trends.png` - Visualization of pre/post trends
5. `event_study.png` - Dynamic DiD coefficients

---

## Session Information
- Date: January 27, 2026
- Software: Python 3.x with pandas, numpy, statsmodels, matplotlib
- Data: ACS 2008-2016 (excluding 2012), N = 17,382 observations
