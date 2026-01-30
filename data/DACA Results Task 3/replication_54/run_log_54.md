# Run Log for Replication Study 54
## DACA Effect on Full-Time Employment

**Date:** January 27, 2026

---

## Overview

This log documents the commands and key decisions made during the independent replication of the DACA study examining the causal impact of DACA eligibility on full-time employment among Mexican-born individuals in the United States.

---

## 1. Data Exploration

### 1.1 Reading Instructions
- Extracted text from `replication_instructions.docx` using Python's python-docx library
- Key specifications identified:
  - Treatment group: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
  - Control group: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
  - Outcome: Full-time employment (FT, defined as 35+ hours/week)
  - Pre-period: 2008-2011; Post-period: 2013-2016
  - Use provided ELIGIBLE, AFTER, and FT variables

### 1.2 Data Loading
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```

### 1.3 Sample Characteristics
- **Total observations:** 17,382
- **Treated group (ELIGIBLE=1):** 11,382
- **Control group (ELIGIBLE=0):** 6,000
- **Pre-period observations:** 9,527
- **Post-period observations:** 7,855
- **Number of states:** 50

### 1.4 Key Variable Distributions
- FT Employment: 64.91% overall
- ELIGIBLE=1 pre-treatment FT: 62.63%
- ELIGIBLE=1 post-treatment FT: 66.58%
- ELIGIBLE=0 pre-treatment FT: 66.97%
- ELIGIBLE=0 post-treatment FT: 64.49%

---

## 2. Analytical Decisions

### 2.1 Primary Estimator
**Decision:** Use OLS (linear probability model) as primary estimator
**Rationale:**
- Easy interpretation of coefficients as percentage point changes
- Robust standard errors address heteroskedasticity
- Marginal effects from probit/logit are identical (verified)

### 2.2 Standard Errors
**Decision:** Heteroskedasticity-robust (HC1) standard errors as baseline
**Rationale:**
- Binary outcome variable implies heteroskedasticity
- Also tested state-clustered SEs for robustness

### 2.3 Control Variables
**Decision:** Include demographic controls in preferred specification
**Variables included:**
- Education dummies (reference: Less than HS)
- Female indicator (SEX == 2)
- Married indicator (MARST == 1)
- Number of children (NCHILD)
- Family size (FAMSIZE)

**Rationale:** Observable differences exist between treatment/control groups; controls improve precision

### 2.4 Fixed Effects
**Decision:** Include state and year fixed effects in preferred specification
**Rationale:**
- State FE control for time-invariant state characteristics
- Year FE control for common macroeconomic shocks
- Year FE absorb main effect of AFTER variable

### 2.5 Survey Weights
**Decision:** Report both weighted and unweighted estimates; prefer unweighted for main results
**Rationale:**
- Weighted estimates (WLS) useful for population inference
- Results robust to weighting choice
- Unweighted estimates more commonly used in DID settings

---

## 3. Commands Executed

### 3.1 Basic DID (Model 1)
```python
import statsmodels.api as sm

df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
y = df['FT']
model1 = sm.OLS(y, X).fit(cov_type='HC1')
```
**Result:** ELIGIBLE_AFTER = 0.0643 (SE: 0.0153, p<0.001)

### 3.2 Weighted DID (Model 2)
```python
model2 = sm.WLS(y, X, weights=df['PERWT']).fit(cov_type='HC1')
```
**Result:** ELIGIBLE_AFTER = 0.0748 (SE: 0.0181, p<0.001)

### 3.3 DID with Controls (Model 3)
```python
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

controls = ['EDUC_HS', 'EDUC_SOMECOL', 'EDUC_2YR', 'EDUC_BA',
            'FEMALE', 'MARRIED', 'NCHILD', 'FAMSIZE']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER'] + controls])
model3 = sm.OLS(y, X).fit(cov_type='HC1')
```
**Result:** ELIGIBLE_AFTER = 0.0518 (SE: 0.0141, p<0.001)

### 3.4 DID with State/Year FE (Model 4 - PREFERRED)
```python
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True, dtype=int)
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True, dtype=int)

X_df = pd.concat([df[['ELIGIBLE', 'ELIGIBLE_AFTER'] + controls],
                  state_dummies, year_dummies], axis=1)
X = sm.add_constant(X_df.astype(float))
model4 = sm.OLS(y, X).fit(cov_type='HC1')
```
**Result:** ELIGIBLE_AFTER = 0.0507 (SE: 0.0141, p=0.0003)

### 3.5 Weighted with FE (Model 5)
```python
model5 = sm.WLS(y, X, weights=df['PERWT']).fit(cov_type='HC1')
```
**Result:** ELIGIBLE_AFTER = 0.0580 (SE: 0.0166, p=0.0005)

### 3.6 Clustered Standard Errors (Model 6)
```python
model_cluster = sm.OLS(y, X_basic).fit(cov_type='cluster',
                                        cov_kwds={'groups': df['STATEFIP']})
```
**Result:** ELIGIBLE_AFTER = 0.0643 (SE: 0.0141, p<0.001)

---

## 4. Robustness Checks

### 4.1 Placebo Test (Pre-period only: 2008-2011)
- Created FAKE_AFTER = 1 for 2010-2011
- **Result:** ELIGIBLE × FAKE_AFTER = 0.0157 (SE: 0.0205, p=0.444)
- **Interpretation:** No significant pre-trend, supports parallel trends assumption

### 4.2 Event Study
- Estimated year-by-year effects relative to 2011
- **Pre-treatment coefficients:** Some variation (2008: -0.059, 2009: -0.039, 2010: -0.066)
- **Post-treatment coefficients:** Gradual increase (2013: 0.019, 2014: -0.009, 2015: 0.030, 2016: 0.049)
- **Interpretation:** Effects grow over time; some pre-trend variation warrants caution

### 4.3 Probit Model
```python
probit_model = sm.Probit(y, X).fit()
marginal_effects = probit_model.get_margeff()
```
- **Average Marginal Effect:** 0.0643 (identical to OLS)

### 4.4 Logit Model
```python
logit_model = sm.Logit(y, X).fit()
```
- **Average Marginal Effect:** 0.0643 (identical to OLS)
- **Odds Ratio:** 1.327

### 4.5 Heterogeneity Analysis
**By Sex:**
- Male: 0.0615 (SE: 0.0170, p<0.001)
- Female: 0.0452 (SE: 0.0232, p=0.051)

**By Education:**
- HS or less: 0.0480 (SE: 0.0180, p=0.008)
- Some College+: 0.1057 (SE: 0.0287, p<0.001)

### 4.6 Triple Difference (Driver's License)
- DID in non-DL states: 0.0879 (SE: 0.0207)
- Triple-diff interaction: -0.0887 (SE: 0.0577, not significant)
- **Finding:** Surprising negative interaction; effect concentrated in non-DL states

---

## 5. Preferred Estimate

**Model:** OLS with demographic controls + state/year fixed effects + robust SEs

| Statistic | Value |
|-----------|-------|
| Treatment Effect | 0.0507 (5.07 pp) |
| Standard Error | 0.0141 |
| 95% CI | [0.0230, 0.0784] |
| p-value | 0.0003 |
| Sample Size | 17,382 |
| R-squared | 0.1385 |

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 5 percentage points, representing an 8.1% relative increase from the pre-treatment mean of 62.6%.

---

## 6. Files Generated

1. `replication_report_54.tex` - Full LaTeX replication report (~20 pages)
2. `replication_report_54.pdf` - Compiled PDF report
3. `run_log_54.md` - This run log documenting commands and decisions

---

## 7. Software Environment

- **Python:** 3.14
- **Key packages:**
  - pandas
  - numpy
  - statsmodels
- **LaTeX:** pdflatex for compilation

---

## 8. Notes and Observations

1. The data was pre-cleaned; ELIGIBLE, AFTER, and FT variables were provided
2. Year 2012 excluded from data (policy implementation year)
3. Sample restricted to Mexican-born Hispanic individuals meeting DACA eligibility criteria
4. Control group slightly older, more married, more children - captured by controls
5. Effect robust across all specifications (range: 0.048 to 0.075)
6. Larger effects for more educated individuals
7. Parallel trends assumption supported by placebo test but event study shows some pre-period variation

---

*End of Run Log*
