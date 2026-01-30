# DACA Replication Study 46 - Run Log

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA effect on full-time employment study.

## Date: January 27, 2026

---

## 1. Data Loading and Exploration

### 1.1 Read Instructions
- Read replication_instructions.docx using python-docx library
- Research Question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Pre-period: 2008-2011
- Post-period: 2013-2016

### 1.2 Load Data
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```
- Dataset shape: 17,382 observations x 105 variables
- Years available: 2008-2011, 2013-2016 (2012 omitted)

### 1.3 Key Variable Checks
- ELIGIBLE: 1 = treatment group (ages 26-30), 0 = control (ages 31-35)
- AFTER: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- FT: 1 = full-time employed (35+ hours/week), 0 = otherwise

Distribution:
- ELIGIBLE=1: 11,382 observations
- ELIGIBLE=0: 6,000 observations
- AFTER=0: 9,527 observations
- AFTER=1: 7,855 observations

---

## 2. Analysis Decisions

### 2.1 Estimation Method
- **Decision**: Use weighted least squares (WLS) with ACS person weights (PERWT)
- **Rationale**: PERWT weights are designed to produce population-representative estimates from the ACS sample

### 2.2 Standard Errors
- **Decision**: Use heteroskedasticity-robust standard errors (HC1)
- **Robustness check**: Also estimate with state-clustered standard errors
- **Rationale**: HC1 accounts for heteroskedasticity; clustering accounts for within-state correlation

### 2.3 Model Specification
- **Basic model**: FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
- **Extended models**: Add year fixed effects, demographic controls, education, state fixed effects
- **Preferred specification (Model 6)**: Year FE + State FE + Demographics + Education

### 2.4 Control Variables
Demographics:
- FEMALE: Binary (SEX == 2)
- MARRIED: Binary (MARST == 1)
- HAS_CHILDREN: Binary (NCHILD > 0)

Education (reference: High School):
- EDUC_SOMECOLL: Some college
- EDUC_AA: Two-year degree
- EDUC_BA: Bachelor's or higher

Fixed Effects:
- Year fixed effects (reference: 2008)
- State fixed effects (reference: first state in sorted STATEFIP)

### 2.5 Sample
- **Decision**: Use full provided sample, no additional restrictions
- **Rationale**: Instructions specify "do not further limit the sample"
- Individuals not in labor force retained, coded as FT=0

---

## 3. Commands and Code

### 3.1 Simple DiD Calculation
```python
# Weighted means by group
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
)

# Treatment: 0.637 (pre) -> 0.686 (post), diff = +0.049
# Control: 0.689 (pre) -> 0.663 (post), diff = -0.026
# DiD = 0.049 - (-0.026) = 0.075 (7.5 pp)
```

### 3.2 Main Regression Models
```python
import statsmodels.api as sm
from statsmodels.api import WLS

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic unweighted
X1 = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model1 = sm.OLS(df['FT'], X1).fit(cov_type='HC1')

# Model 2: Basic weighted
model2 = WLS(df['FT'], X1, weights=df['PERWT']).fit(cov_type='HC1')

# Models 3-6: Add year FE, demographics, education, state FE progressively
```

### 3.3 Robustness Checks

#### Event Study
```python
# Create ELIGIBLE x Year interactions (ref: 2011)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year)
```

#### Placebo Test
```python
df_pre = df[df['AFTER'] == 0]
df_pre['FAKE_AFTER'] = (df_pre['YEAR'] >= 2010).astype(float)
df_pre['FAKE_TREAT'] = df_pre['ELIGIBLE'] * df_pre['FAKE_AFTER']
```

#### Alternative Bandwidths
- Ages 27-34: Narrower window around cutoff
- Ages 28-33: Even narrower window

#### Clustered SE
```python
model.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

---

## 4. Results Summary

### 4.1 Main Results Table

| Model | Specification | Coefficient | SE | p-value |
|-------|--------------|-------------|-----|---------|
| 1 | Basic (Unweighted) | 0.064 | 0.015 | <0.001 |
| 2 | Basic (Weighted) | 0.075 | 0.018 | <0.001 |
| 3 | Year FE | 0.072 | 0.018 | <0.001 |
| 4 | Year FE + Demographics | 0.061 | 0.017 | <0.001 |
| 5 | Year FE + Demo + Educ + Region | 0.058 | 0.017 | <0.001 |
| 6 | Year + State FE + Demo + Educ | 0.058 | 0.017 | <0.001 |

### 4.2 Preferred Estimate
- **Effect size**: 0.058 (5.8 percentage points)
- **Standard error**: 0.017
- **95% CI**: [0.025, 0.090]
- **p-value**: < 0.001
- **N**: 17,382

### 4.3 Robustness Results

| Check | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Male subgroup | 0.059 | 0.020 | 0.003 |
| Female subgroup | 0.052 | 0.028 | 0.059 |
| Placebo (pre-period) | 0.018 | 0.024 | 0.461 |
| Ages 27-34 | 0.048 | 0.019 | 0.010 |
| Ages 28-33 | 0.055 | 0.022 | 0.013 |
| Clustered SE | 0.061 | 0.020 | 0.002 |

---

## 5. Visualizations Generated

1. **figure1_trends.png**: Full-time employment trends by eligibility status over time
2. **figure2_event_study.png**: Event study coefficients with confidence intervals
3. **figure3_specifications.png**: Bar chart of DiD estimates across specifications
4. **figure4_balance.png**: Covariate balance between treatment and control groups

---

## 6. Output Files

### Primary Deliverables
- `replication_report_46.tex`: LaTeX source file (18 pages)
- `replication_report_46.pdf`: Compiled PDF report
- `run_log_46.md`: This file

### Intermediate Files
- `model_results.csv`: Summary of all model estimates
- `summary_statistics.csv`: Descriptive statistics by group and period

---

## 7. Key Decisions Summary

1. **Weighting**: Used PERWT (ACS person weights) for population-representative estimates
2. **Standard errors**: HC1 robust SE as primary; clustered SE by state as robustness
3. **Fixed effects**: Included year FE and state FE in preferred specification
4. **Covariates**: Included sex, marital status, children, and education
5. **Sample**: Full provided sample without additional restrictions
6. **Functional form**: Linear probability model (OLS/WLS) for interpretability
7. **Reference categories**: 2008 (year), high school (education), first state (state FE)

---

## 8. Interpretation

The analysis finds that DACA eligibility increased full-time employment by approximately 5.8-7.5 percentage points among Hispanic-Mexican, Mexican-born individuals. The effect is statistically significant at the 1% level across all specifications and robust to various sensitivity analyses. Some concerns exist about pre-trends (event study shows some significant pre-treatment coefficients), but the placebo test passes and the effect is consistent across subgroups and bandwidths.

---

## 9. Software Environment

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (visualizations)
- LaTeX/pdflatex (document compilation)

---

*Log completed: January 27, 2026*
