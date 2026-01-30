# Run Log for DACA Replication Study (Replication 89)

## Date: January 27, 2026

---

## Overview

This document logs all commands executed and key analytical decisions made during the replication of the DACA effect on full-time employment study.

---

## 1. Data Preparation Phase

### 1.1 Data Files Identified
- `data/prepared_data_numeric_version.csv` - Main analysis data (17,382 observations)
- `data/prepared_data_labelled_version.csv` - Labelled version for reference
- `data/acs_data_dict.txt` - Data dictionary from IPUMS

### 1.2 Data Structure
- Years: 2008-2011 (pre-DACA) and 2013-2016 (post-DACA); 2012 excluded
- Treatment group (ELIGIBLE=1): 11,382 individuals aged 26-30 as of June 2012
- Control group (ELIGIBLE=0): 6,000 individuals aged 31-35 as of June 2012

---

## 2. Key Analytical Decisions

### 2.1 Research Design
- **Design**: Difference-in-differences (DiD)
- **Treatment**: DACA eligibility based on age at June 15, 2012
- **Outcome**: Full-time employment (FT = 1 if working 35+ hours/week)
- **Unit of analysis**: Individual-level (repeated cross-section, not panel)

### 2.2 Sample Definition
- **Decision**: Use the entire provided sample without further restrictions
- **Rationale**: Instructions specify "do not further limit the sample by dropping individuals on the basis of their characteristics"
- Individuals not in labor force retained with FT=0

### 2.3 Variable Coding
- **ELIGIBLE**: Pre-constructed (1=treatment, 0=control); did not create own eligibility variable per instructions
- **AFTER**: Pre-constructed (1=2013-2016, 0=2008-2011)
- **FT**: Pre-constructed (1=full-time, 0=not full-time)
- **MALE**: Created from SEX (1 if SEX=1, 0 if SEX=2)
- **MARRIED**: Created from MARST (1 if MARST=1 [married spouse present], 0 otherwise)
- **Education dummies**: Created from EDUC_RECODE with BA+ as reference category

### 2.4 Fixed Effects
- **Year FE**: Included (reference year: 2008)
- **State FE**: Tested but not included in preferred specification
- **Rationale**: State FE had minimal impact on point estimates; preferred specification avoids overfitting

### 2.5 Standard Errors
- **Choice**: Heteroskedasticity-robust (HC1) standard errors
- **Rationale**: Linear probability model with binary outcome requires robust SEs

### 2.6 Survey Weights
- **Main analysis**: Unweighted
- **Robustness**: Weighted results presented using PERWT
- **Rationale**: Both weighted and unweighted results reported for transparency

---

## 3. Model Specifications Estimated

### 3.1 Models Estimated
1. Basic DiD (no controls)
2. DiD with survey weights (PERWT)
3. DiD with robust standard errors
4. DiD with demographic controls (sex, married, age)
5. DiD with demographic + education controls
6. DiD with year fixed effects
7. **DiD with year FE + controls (PREFERRED)**
8. DiD with state + year FE
9. Full model (state FE + year FE + controls)

### 3.2 Preferred Specification Selection
- **Model 7**: Year FE + demographic/education controls
- **Rationale**:
  - Controls for common time trends (year FE)
  - Adjusts for individual characteristics (demographics, education)
  - Avoids overfitting from excessive state FE
  - State FE had negligible impact on estimates

---

## 4. Commands Executed

### 4.1 Python Analysis Script (analysis.py)
```python
# Key imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
data = pd.read_csv('data/prepared_data_numeric_version.csv')
data_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

# Create interaction term
data['ELIGIBLE_X_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Create covariates
data['MALE'] = (data['SEX'] == 1).astype(int)
data['MARRIED'] = (data['MARST'] == 1).astype(int)

# Main DiD regression
model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=data).fit(cov_type='HC1')

# Model with controls and year FE (preferred)
X = data[['ELIGIBLE', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'AGE'] + educ_cols + year_cols]
X = sm.add_constant(X)
model7 = sm.OLS(data['FT'], X).fit(cov_type='HC1')
```

### 4.2 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_89.tex
pdflatex -interaction=nonstopmode replication_report_89.tex  # Second pass for references
```

---

## 5. Key Results

### 5.1 Simple DiD Calculation
```
Treatment (ages 26-30):
  Pre-DACA FT rate:  0.6263
  Post-DACA FT rate: 0.6658
  Change:            +0.0394

Control (ages 31-35):
  Pre-DACA FT rate:  0.6697
  Post-DACA FT rate: 0.6449
  Change:            -0.0248

DiD estimate: 0.0394 - (-0.0248) = 0.0643
```

### 5.2 Preferred Estimate (Model 7)
```
ELIGIBLE_X_AFTER coefficient: 0.0524
Standard Error (robust):      0.0141
t-statistic:                  3.704
p-value:                      0.0002
95% Confidence Interval:      [0.0247, 0.0801]
Sample Size:                  17,382
R-squared:                    0.133
```

### 5.3 Robustness Results
| Specification | Estimate | SE | p-value |
|---------------|----------|-----|---------|
| Basic DiD | 0.0643 | 0.0153 | <0.001 |
| With controls | 0.0539 | 0.0142 | <0.001 |
| Year FE + controls | 0.0524 | 0.0141 | <0.001 |
| State + Year FE | 0.0626 | 0.0152 | <0.001 |
| Narrow bandwidth | 0.0637 | 0.0189 | <0.001 |
| Placebo test | 0.0157 | 0.0205 | 0.444 |
| With policy controls | 0.0618 | 0.0152 | <0.001 |
| Weighted (PERWT) | 0.0624 | 0.0167 | <0.001 |

---

## 6. Parallel Trends Assessment

### 6.1 Pre-Trend Test
- Differential pre-trend coefficient: 0.0151
- Standard error: 0.0091
- p-value: 0.097
- **Interpretation**: No statistically significant differential pre-trends

### 6.2 Event Study
- Pre-treatment coefficients (2008-2010) relative to 2011 are not systematically trending
- Post-treatment coefficients show gradual increase, especially in 2015-2016
- Pattern consistent with DACA effects materializing over time

---

## 7. Subgroup Analyses

### 7.1 By Sex
- Male: DiD = 0.0615 (SE=0.0170, p<0.001)
- Female: DiD = 0.0452 (SE=0.0232, p=0.051)

### 7.2 By Education
- High School: DiD = 0.0482 (SE=0.0180, p=0.008)
- Some College: DiD = 0.1075 (SE=0.0380, p=0.005)
- Two-Year Degree: DiD = 0.1256 (SE=0.0657, p=0.056)
- BA+: DiD = 0.0856 (SE=0.0588, p=0.145)

---

## 8. Output Files Generated

1. `analysis.py` - Main Python analysis script
2. `results_summary.csv` - Summary of key regression results
3. `model_output.txt` - Detailed regression output
4. `event_study_results.csv` - Event study coefficients
5. `trends_data.csv` - FT rates by year and group
6. `summary_statistics.csv` - Descriptive statistics by group/period
7. `replication_report_89.tex` - LaTeX source for report
8. `replication_report_89.pdf` - Final PDF report (20 pages)
9. `run_log_89.md` - This log file

---

## 9. Interpretation

The analysis finds that DACA eligibility increased full-time employment by approximately 5.2 percentage points (preferred estimate), representing an 8.4% increase relative to the pre-treatment mean. This effect is:

- Statistically significant at the 1% level
- Robust across multiple specifications
- Larger for men than women
- Larger for individuals with some college education
- Supported by parallel trends analysis and placebo tests

The findings are consistent with DACA's provision of legal work authorization facilitating transitions to formal full-time employment.

---

## 10. Notes and Caveats

1. **Intent-to-treat**: Effect is based on eligibility, not actual DACA receipt
2. **Comparison group**: Control group is older; age differences partially addressed by controls
3. **Generalizability**: Sample limited to Hispanic-Mexican, Mexican-born individuals
4. **Measurement**: FT based on "usual" hours, may not capture all employment dynamics
5. **No clustering**: Standard errors are heteroskedasticity-robust but not clustered (e.g., by state or birth cohort)

---

## 11. Session Information

- Analysis conducted: January 27, 2026
- Python version: 3.x
- Key packages: pandas, numpy, statsmodels, scipy
- LaTeX distribution: MiKTeX

---

*End of Run Log*
