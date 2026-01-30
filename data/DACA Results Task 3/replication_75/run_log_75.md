# Run Log - DACA Replication Study (ID: 75)

## Overview
This log documents all commands executed and key decisions made during the independent replication of the DACA eligibility effect on full-time employment study.

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Extracted replication instructions from `replication_instructions.docx` using Python's `python-docx` library
- Key research question: Effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals

### 1.2 Data Files Identified
- `data/prepared_data_labelled_version.csv` - Labelled version with 105 columns
- `data/prepared_data_numeric_version.csv` - Numeric version used for analysis
- `data/acs_data_dict.txt` - Variable documentation from IPUMS

### 1.3 Initial Data Exploration

```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Dataset shape:', df.shape)  # (17382, 105)
```

**Key findings:**
- Total observations: 17,382
- Years covered: 2008-2011 (pre-DACA), 2013-2016 (post-DACA)
- 2012 omitted (treatment year ambiguity)
- ELIGIBLE=1: 11,382 observations (ages 26-30 at treatment)
- ELIGIBLE=0: 6,000 observations (ages 31-35 at treatment)

---

## 2. Analytical Decisions

### 2.1 Identification Strategy
- **Method**: Difference-in-Differences (DiD)
- **Treatment group**: Individuals ages 26-30 on June 15, 2012 (ELIGIBLE=1)
- **Control group**: Individuals ages 31-35 on June 15, 2012 (ELIGIBLE=0)
- **Pre-period**: 2008-2011 (AFTER=0)
- **Post-period**: 2013-2016 (AFTER=1)

### 2.2 Outcome Variable
- **FT**: Full-time employment (=1 if usually works 35+ hours/week)
- Individuals not in labor force coded as FT=0 (as per instructions)

### 2.3 Control Variables Selected
- **FEMALE**: Gender indicator (SEX=2)
- **AGE**: Age at time of survey
- **MARRIED**: Marital status (MARST in {1,2})
- **HAS_CHILDREN**: Has children in household (NCHILD > 0)
- **EDUC_HS**: High school degree (EDUC=7)
- **EDUC_SOMECOLL**: Some college (EDUC=8)
- **EDUC_BA_PLUS**: Bachelor's or higher (EDUC in {10,11})

### 2.4 Weighting Decision
- Used ACS person weights (PERWT) for population-representative estimates
- Unweighted results provided as robustness check

### 2.5 Standard Error Approach
- Primary: Heteroskedasticity-robust (HC1)
- Robustness: State-clustered standard errors

---

## 3. Main Analysis Commands

### 3.1 Raw DiD Calculation

```python
# Simple means calculation
ft_eligible_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # 0.6263
ft_eligible_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()   # 0.6658
ft_control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()   # 0.6697
ft_control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()    # 0.6449

# DiD = (0.6658 - 0.6263) - (0.6449 - 0.6697) = 0.0643
```

### 3.2 Regression Models

**Model 1: Basic DiD (unweighted)**
```python
import statsmodels.formula.api as smf
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
# DiD coefficient: 0.0643 (SE: 0.015, p<0.001)
```

**Model 2: DiD with weights**
```python
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0748 (SE: 0.018, p<0.001)
```

**Model 3: DiD with demographic controls (PREFERRED)**
```python
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['EDUC_HS'] = (df['EDUC'] == 7).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC'] == 8).astype(int)
df['EDUC_BA_PLUS'] = (df['EDUC'].isin([10, 11])).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_BA_PLUS'
model3 = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0617 (SE: 0.017, p<0.001)
```

**Model 4: Full model with state and year FE**
```python
formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_BA_PLUS + C(YEAR) + C(STATEFIP)'
model4 = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0585 (SE: 0.017, p<0.001)
```

---

## 4. Robustness Checks

### 4.1 Event Study Analysis
```python
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

# Results (relative to 2008):
# 2009: 0.018 (p=0.553) - pre-trend OK
# 2010: -0.013 (p=0.676) - pre-trend OK
# 2011: 0.065 (p=0.045) - borderline
# 2013: 0.079 (p=0.012) - post-DACA effect
# 2014: 0.051 (p=0.118)
# 2015: 0.054 (p=0.092)
# 2016: 0.124 (p<0.001) - strong effect
```

### 4.2 Placebo Test (Fake 2010 Treatment)
```python
df_pre = df[df['AFTER'] == 0].copy()
df_pre['AFTER_FAKE'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_AFTER_FAKE'] = df_pre['ELIGIBLE'] * df_pre['AFTER_FAKE']
# Placebo coefficient: 0.017 (p=0.443) - not significant as expected
```

### 4.3 Heterogeneity by Sex
```python
# Males: DiD = 0.061 (SE: 0.020, p=0.002)
# Females: DiD = 0.053 (SE: 0.028, p=0.056)
```

### 4.4 State Policy Controls
```python
# With DRIVERSLICENSES, INSTATETUITION, EVERIFY, SECURECOMMUNITIES
# DiD coefficient: 0.060 (p<0.001) - robust to policy controls
```

---

## 5. Visualizations Created

1. **figure1_trends.png**: Time trends in FT employment by eligibility status
2. **figure2_event_study.png**: Event study coefficients with confidence intervals
3. **figure3_did.png**: DiD visualization with counterfactual

---

## 6. Key Results Summary

### Preferred Estimate (Model 3)
- **Effect size**: 6.17 percentage points (0.0617)
- **Standard error**: 0.017
- **95% CI**: [0.029, 0.095]
- **p-value**: < 0.001
- **Sample size**: 17,382

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 6.2 percentage points among eligible individuals, representing a ~9.7% increase relative to the pre-treatment mean of 63.7%.

---

## 7. Files Produced

| File | Description |
|------|-------------|
| `replication_report_75.tex` | LaTeX source for replication report |
| `replication_report_75.pdf` | Compiled PDF report (23 pages) |
| `run_log_75.md` | This log file |
| `figure1_trends.png` | Employment trends figure |
| `figure2_event_study.png` | Event study figure |
| `figure3_did.png` | DiD visualization figure |

---

## 8. Software Environment

- **Python**: 3.14
- **Key packages**: pandas, numpy, statsmodels, matplotlib
- **LaTeX**: MiKTeX 25.12, pdfTeX 3.141592653-2.6-1.40.28

---

## 9. Decisions Rationale

1. **Why Model 3 as preferred?**
   - Includes person weights for population representativeness
   - Controls for key demographic confounders
   - Retains interpretable AFTER coefficient (unlike FE models)
   - Balances parsimony with covariate adjustment

2. **Why HC1 standard errors?**
   - Appropriate for potential heteroskedasticity in binary outcomes
   - Conservative compared to classical SEs
   - State-clustered SEs provided as robustness check

3. **Why include non-labor force as FT=0?**
   - Per instructions: "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis"
   - Captures extensive margin effects (labor force entry)

4. **Why not use regression discontinuity?**
   - Age cutoff at 31 is not sharp in the data (ages grouped)
   - DiD more appropriate for this repeated cross-section design
   - Comparison group well-matched on other eligibility criteria

---

*Log completed: January 27, 2026*
