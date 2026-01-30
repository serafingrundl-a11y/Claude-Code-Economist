# Run Log - DACA Replication Study (ID: 85)

## Date: January 27, 2026

---

## Overview

This document logs all commands executed and key decisions made during the independent replication of the DACA effect on full-time employment.

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read `replication_instructions.docx` using Python docx library
- Key parameters extracted:
  - **Research Question**: Effect of DACA eligibility on full-time employment
  - **Treatment Group**: Ages 26-30 at June 15, 2012
  - **Control Group**: Ages 31-35 at June 15, 2012
  - **Pre-period**: 2008-2011
  - **Post-period**: 2013-2016 (2012 excluded)
  - **Outcome**: FT (full-time employment, 35+ hours/week)

### 1.2 Data Files Identified
```
data/
├── acs_data_dict.txt (variable documentation)
├── prepared_data_labelled_version.csv (labeled data)
└── prepared_data_numeric_version.csv (numeric data for analysis)
```

### 1.3 Key Variables Confirmed
- `FT`: Full-time employment (0/1)
- `ELIGIBLE`: Treatment group indicator (0/1)
- `AFTER`: Post-DACA period indicator (0/1)
- `YEAR`: Survey year
- `STATEFIP`: State FIPS code
- `PERWT`: Person weight

---

## 2. Analysis Decisions

### 2.1 Model Specification Choices

**Decision 1: Linear Probability Model (OLS)**
- Rationale: Standard in applied econometrics literature; easy interpretation of coefficients as percentage point changes; robust to heteroskedasticity with HC1 standard errors
- Alternative considered: Logit/Probit (implemented as robustness check)

**Decision 2: Fixed Effects Structure**
- Included year fixed effects to control for macroeconomic trends
- Included state fixed effects to control for persistent state-level differences
- Rationale: Year FE captures business cycle effects; State FE captures differences in labor markets and immigration policy environments

**Decision 3: Control Variables**
- Included: Sex (female indicator), marital status, education categories, number of children, age
- Rationale: These are standard human capital and demographic controls that predict employment
- Caution: Did not include post-treatment variables that could be affected by DACA

**Decision 4: Standard Errors**
- Used heteroskedasticity-robust standard errors (HC1)
- Rationale: Binary outcome variable naturally induces heteroskedasticity

**Decision 5: Preferred Specification**
- Selected Model 3 (Year + State FE, no individual covariates) as preferred
- Rationale: Balances control for confounders with avoidance of "bad controls"
- Adding covariates (Model 4) reduces estimate slightly but conclusions unchanged

### 2.2 Sample Decisions

**Decision 6: Full Sample Used**
- Per instructions: "do not further limit the sample by dropping individuals on the basis of their characteristics"
- Used entire provided dataset without additional restrictions
- Sample size: 17,382 observations

**Decision 7: Use of Pre-Coded Variables**
- Used provided `ELIGIBLE`, `AFTER`, and `FT` variables as-is
- Per instructions: "Use this variable to identify individuals in the treated and comparison groups, and do not create your own eligibility variable"

---

## 3. Commands Executed

### 3.1 Data Loading and Exploration
```python
df = pd.read_csv('data/prepared_data_numeric_version.csv')
# Total observations: 17,382
# Treatment (ELIGIBLE=1): 11,382
# Control (ELIGIBLE=0): 6,000
# Pre-DACA (AFTER=0): 9,527
# Post-DACA (AFTER=1): 7,855
```

### 3.2 Descriptive Statistics
```python
# Full-time employment rates by group and period:
# Treatment, Pre-DACA: 62.63%
# Treatment, Post-DACA: 66.58%
# Control, Pre-DACA: 66.97%
# Control, Post-DACA: 64.49%
```

### 3.3 Regression Models
```python
# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + TREAT_POST', data=df).fit(cov_type='HC1')

# Model 2: Year Fixed Effects
model2 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + TREAT_POST', data=df).fit(cov_type='HC1')

# Model 3: Year + State Fixed Effects (PREFERRED)
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df).fit(cov_type='HC1')

# Model 4: Full Covariates
formula = 'FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST + FEMALE + MARRIED + EDUC_HS + EDUC_SOME_COLLEGE + EDUC_BA_PLUS + NCHILD + AGE'
model4 = smf.ols(formula, data=df).fit(cov_type='HC1')

# Model 5: State Policy Controls
model5 = smf.ols(formula + ' + DRIVERSLICENSES + INSTATETUITION + STATEFINANCIALAID + EVERIFY + SECURECOMMUNITIES', data=df).fit(cov_type='HC1')
```

### 3.4 Robustness Checks
```python
# Logit marginal effects
logit_model = smf.logit('FT ~ ELIGIBLE + C(YEAR) + TREAT_POST', data=df).fit()

# Weighted regression
model_weighted = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df, weights=df['PERWT']).fit()

# Gender subgroup analysis
model_male = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df[df['SEX']==1]).fit()
model_female = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + TREAT_POST', data=df[df['SEX']==2]).fit()

# Placebo test (fake treatment in 2010)
df_pre = df[df['AFTER'] == 0].copy()
df_pre['PLACEBO_POST'] = (df_pre['YEAR'] >= 2010).astype(int)
model_placebo = smf.ols('FT ~ ELIGIBLE + PLACEBO_POST + ELIGIBLE*PLACEBO_POST', data=df_pre).fit()

# Event study
event_formula = 'FT ~ ELIGIBLE + C(YEAR) + TREAT_2008 + TREAT_2009 + TREAT_2010 + TREAT_2013 + TREAT_2014 + TREAT_2015 + TREAT_2016'
model_event = smf.ols(event_formula, data=df).fit()
```

---

## 4. Key Results

### 4.1 Main Estimates

| Model | DiD Coefficient | Std. Error | 95% CI | R² |
|-------|----------------|------------|--------|-----|
| 1. Basic | 0.0643 | 0.0153 | [0.034, 0.094] | 0.002 |
| 2. Year FE | 0.0629 | 0.0152 | [0.033, 0.093] | 0.004 |
| **3. Year+State FE** | **0.0626** | **0.0152** | **[0.033, 0.092]** | **0.008** |
| 4. Covariates | 0.0546 | 0.0141 | [0.027, 0.082] | 0.137 |
| 5. Policies | 0.0537 | 0.0142 | [0.026, 0.081] | 0.137 |

### 4.2 Preferred Estimate (Model 3)
- **Effect Size**: 6.26 percentage points
- **Standard Error**: 0.0152
- **95% CI**: [3.27, 9.24] percentage points
- **P-value**: < 0.0001
- **Sample Size**: 17,382

### 4.3 Robustness Results
- **Logit marginal effect**: 0.063 (consistent with OLS)
- **Weighted regression**: 0.071 (slightly larger)
- **Male subsample**: 0.060
- **Female subsample**: 0.045
- **Placebo test**: 0.016 (p=0.44, not significant - supports parallel trends)

---

## 5. Files Generated

### 5.1 Analysis Files
- `analysis.py` - Main analysis script
- `analysis_results.json` - Results in JSON format

### 5.2 Figures
- `figure1_trends.png` - Full-time employment trends by group
- `figure2_event_study.png` - Event study plot
- `figure3_model_comparison.png` - Effect estimates across specifications
- `figure4_gender.png` - Effects by gender

### 5.3 Report
- `replication_report_85.tex` - LaTeX source (22 pages)
- `replication_report_85.pdf` - Final PDF report

---

## 6. Software Environment

- **Python**: 3.x
- **Key packages**:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis)
  - matplotlib (visualization)
- **LaTeX**: pdfTeX (MiKTeX distribution)

---

## 7. Interpretation Notes

### 7.1 Main Finding
DACA eligibility is associated with a statistically significant 6.26 percentage point increase in full-time employment. This represents approximately a 10% increase relative to the baseline rate of 62.63%.

### 7.2 Caveats
1. Age differences between groups may introduce confounding
2. Pre-treatment event study coefficients suggest some pre-trend differences
3. Estimate is intent-to-treat (effect of eligibility, not receipt)
4. Cross-sectional data cannot track individuals over time
5. Sample limited to Mexican-born Hispanic individuals

### 7.3 Conclusion
The positive effect of DACA on full-time employment is robust across specifications and supported by placebo tests. The finding is consistent with the hypothesis that legal work authorization improves labor market outcomes for eligible immigrants.

---

## 8. Completion Checklist

- [x] Read replication instructions
- [x] Load and explore data
- [x] Implement DiD analysis
- [x] Run robustness checks (logit, weighted, subgroups, placebo, event study)
- [x] Create visualizations (4 figures)
- [x] Write LaTeX report (~22 pages)
- [x] Compile PDF
- [x] Create run log

**All deliverables completed successfully.**
