# Replication Run Log - Study #86

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA employment study.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (defined as usually working 35+ hours per week)?

---

## Data Exploration

### 1. Reading Instructions
```python
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]
```

### 2. Listing Data Files
Data files identified:
- `data/prepared_data_labelled_version.csv`
- `data/prepared_data_numeric_version.csv`
- `data/acs_data_dict.txt`

### 3. Examining Dataset Structure
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)
# Output: Shape: (17382, 105)
```

### 4. Key Variables Identified
- **FT**: Full-time employment (0/1)
- **ELIGIBLE**: Treatment group indicator (1=ages 26-30, 0=ages 31-35)
- **AFTER**: Post-treatment period (1=2013-2016, 0=2008-2011)
- **PERWT**: ACS person weights

---

## Key Analytical Decisions

### Decision 1: Use Provided Variables
**Rationale**: Instructions explicitly state to use the provided ELIGIBLE, AFTER, and FT variables rather than constructing custom eligibility criteria.

### Decision 2: Include Survey Weights
**Rationale**: ACS is a complex survey design; PERWT weights are necessary for population-representative estimates. Implemented via weighted least squares (WLS).

### Decision 3: Covariates Selection
Covariates included:
- SEX (converted to FEMALE dummy)
- MARST (converted to MARRIED dummy)
- NCHILD (number of children)
- EDUC_RECODE (education category dummies)

**Rationale**: These covariates capture demographic differences between treatment and control groups that may affect employment outcomes.

### Decision 4: Fixed Effects
Included year fixed effects and state fixed effects in the preferred specification.

**Rationale**:
- Year FE control for common time trends (e.g., business cycle effects)
- State FE control for geographic differences in labor markets and policy environments

### Decision 5: Standard Errors
Used heteroskedasticity-robust (HC1) standard errors.

**Rationale**: The outcome is binary, so heteroskedasticity is expected by construction. Robust SEs provide valid inference.

### Decision 6: Preferred Specification
Model 4 (WLS with covariates + year FE + state FE) selected as preferred specification.

**Rationale**: Most comprehensive specification that controls for both individual characteristics and time/geographic variation while still identifying the treatment effect through the DiD interaction term.

---

## Analysis Commands

### Basic DiD Table (Weighted)
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# 2x2 DiD calculation
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

# Results:
# Pre-treatment: Treated=0.637, Control=0.689
# Post-treatment: Treated=0.686, Control=0.663
# DiD = (0.686-0.637) - (0.663-0.689) = 0.049 - (-0.026) = 0.075
```

### Main Regression Models
```python
import statsmodels.api as sm

df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Model 1: Basic OLS
X1 = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER']]
X1 = sm.add_constant(X1)
y = df['FT']
model1 = sm.OLS(y, X1).fit(cov_type='HC1')
# DiD coefficient: 0.0643 (SE: 0.0153)

# Model 2: WLS with survey weights
model2 = sm.WLS(y, X1, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0748 (SE: 0.0181)

# Model 3: WLS with covariates
# Includes: FEMALE, MARRIED, NCHILD, education dummies
# DiD coefficient: 0.0641 (SE: 0.0167)

# Model 4: WLS with covariates + year FE + state FE
# DiD coefficient: 0.0608 (SE: 0.0166)
```

### Parallel Trends Test
```python
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_CENTERED'] = pre_df['YEAR'] - 2008
pre_df['ELIGIBLE_X_YEAR'] = pre_df['ELIGIBLE'] * pre_df['YEAR_CENTERED']

X_pre = pre_df[['ELIGIBLE', 'YEAR_CENTERED', 'ELIGIBLE_X_YEAR']]
X_pre = sm.add_constant(X_pre)
model_pre = sm.WLS(pre_df['FT'], X_pre, weights=pre_df['PERWT']).fit(cov_type='HC1')

# ELIGIBLE_X_YEAR coefficient: 0.0174 (SE: 0.011, p=0.113)
# Conclusion: Cannot reject parallel trends assumption
```

### Event Study
```python
# Reference year: 2011 (last pre-treatment year)
# Coefficients relative to 2011:
# 2008: -0.068 (SE: 0.035)
# 2009: -0.050 (SE: 0.036)
# 2010: -0.082 (SE: 0.036) **
# 2013: +0.016 (SE: 0.038)
# 2014: +0.000 (SE: 0.038)
# 2015: +0.001 (SE: 0.038)
# 2016: +0.074 (SE: 0.038) *
```

### Subgroup Analysis
```python
# Males: DiD = 0.0716 (SE: 0.0199) ***
# Females: DiD = 0.0527 (SE: 0.0281) *
```

---

## Visualization Commands

### Figure 1: Trends Plot
```python
import matplotlib.pyplot as plt

trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trends.index, trends[1], 'b-o', label='Treated (Ages 26-30)')
ax.plot(trends.index, trends[0], 'r--s', label='Control (Ages 31-35)')
ax.axvline(x=2012, color='gray', linestyle=':', label='DACA Implementation')
plt.savefig('figure1_trends.png', dpi=150)
```

### Figure 2: Event Study Plot
```python
# Plotted event study coefficients with 95% CIs
plt.savefig('figure2_eventstudy.png', dpi=150)
```

### Figure 3: DiD Illustration
```python
# Simple 2-point DiD visualization with counterfactual
plt.savefig('figure3_did.png', dpi=150)
```

---

## Report Generation

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_86.tex
pdflatex -interaction=nonstopmode replication_report_86.tex  # Second pass for references
```

Output: `replication_report_86.pdf` (18 pages)

---

## Main Results Summary

### Preferred Estimate (Model 4)
- **Effect Size**: 0.0608 (6.08 percentage points)
- **Standard Error**: 0.0166
- **95% Confidence Interval**: [0.0282, 0.0934]
- **p-value**: 0.0003
- **Sample Size**: 17,382

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 6.1 percentage points among ethnically Hispanic-Mexican, Mexican-born individuals who were ages 26-30 at the time of policy implementation (compared to those ages 31-35). This effect is statistically significant at the 1% level and robust across model specifications.

---

## Files Generated

1. `replication_report_86.tex` - LaTeX source
2. `replication_report_86.pdf` - Final report (18 pages)
3. `run_log_86.md` - This log file
4. `figure1_trends.png` - Time trends visualization
5. `figure2_eventstudy.png` - Event study plot
6. `figure3_did.png` - DiD illustration
7. `analysis_results.json` - Saved regression coefficients

---

## Software Environment

- Python 3.14
- pandas
- numpy
- statsmodels
- matplotlib
- LaTeX (MiKTeX)

---

## Date
January 27, 2026
