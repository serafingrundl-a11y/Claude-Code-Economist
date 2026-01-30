# Run Log - DACA Replication Study (ID: 88)

## Overview

This document logs all commands, key decisions, and analytical choices made during the independent replication of the DACA employment effects study.

**Date:** January 26, 2026
**Research Question:** Effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants

---

## 1. Data Exploration

### 1.1 Initial Data Assessment

**Command:**
```bash
head -1 data/data.csv
```

**Output:** 54 columns including YEAR, SAMPLE, SERIAL, PERWT, demographic variables (SEX, AGE, BIRTHQTR, BIRTHYR), ethnicity (HISPAN, HISPAND), birthplace (BPL, BPLD), citizenship (CITIZEN), immigration (YRIMMIG), and employment variables (EMPSTAT, UHRSWORK).

**Command:**
```bash
wc -l data/data.csv
```

**Output:** 33,851,425 rows (including header)

### 1.2 Data Dictionary Review

Reviewed `acs_data_dict.txt` to understand variable coding:

| Variable | Description | Key Codes |
|----------|-------------|-----------|
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | Actual year |
| UHRSWORK | Usual hours worked/week | 0-99 |
| BIRTHQTR | Quarter of birth | 1-4 |
| BIRTHYR | Birth year | Actual year |
| EDUC | Educational attainment | 0-11 |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1=Married spouse present |
| NCHILD | Number of children | 0-9+ |
| PERWT | Person weight | Survey weight |

---

## 2. Sample Construction Decisions

### 2.1 Target Population Definition

**Decision:** Focus on Mexican-born Hispanic individuals who are non-citizens.

**Rationale:**
- HISPAN = 1 identifies Hispanic-Mexican ethnicity
- BPL = 200 identifies Mexican birthplace
- CITIZEN = 3 identifies non-citizens (proxy for undocumented status per instructions)

### 2.2 DACA Eligibility Criteria Applied

| Criterion | Implementation | IPUMS Variable(s) |
|-----------|---------------|-------------------|
| Arrived before 16th birthday | YRIMMIG - BIRTHYR < 16 | YRIMMIG, BIRTHYR |
| Not yet 31 by June 15, 2012 | age_2012 <= 30 for treatment | BIRTHYR, BIRTHQTR |
| Continuous residence since 2007 | YRIMMIG <= 2007 | YRIMMIG |
| Non-citizen status | CITIZEN = 3 | CITIZEN |
| Hispanic-Mexican ethnicity | HISPAN = 1 | HISPAN |
| Born in Mexico | BPL = 200 | BPL |

### 2.3 Age Calculation

**Decision:** Calculate age as of June 15, 2012 using birth year and quarter.

**Implementation:**
```python
df['age_2012'] = 2012 - df['BIRTHYR']
# Adjust for birth quarter: Q3-Q4 births haven't had birthday by June 15
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_2012'] = df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_2012'] - 1
```

**Rationale:** BIRTHQTR 3 (Jul-Sep) and 4 (Oct-Dec) would not have had their birthday by June 15, so their age is one year less.

### 2.4 Treatment and Control Groups

**Treatment Group:** Ages 26-30 as of June 15, 2012 (DACA-eligible)
**Control Group:** Ages 31-35 as of June 15, 2012 (DACA-ineligible due to age)

**Rationale:** Uses adjacent age cohorts to maximize comparability while ensuring clear treatment/control distinction based on the age eligibility cutoff.

### 2.5 Time Period Definition

**Pre-period:** 2006-2011 (ACS years before DACA)
**Post-period:** 2013-2016 (ACS years after DACA)
**Excluded:** 2012 (DACA implemented June 15, 2012; ACS month unknown)

---

## 3. Sample Construction Process

### 3.1 Sequential Filtering

| Step | Filter | Remaining Obs |
|------|--------|---------------|
| 1 | Load full ACS 2006-2016 | 33,851,424 |
| 2 | HISPAN = 1 (Hispanic-Mexican) | 2,945,521 |
| 3 | BPL = 200 (Born in Mexico) | 991,261 |
| 4 | CITIZEN = 3 (Non-citizen) | 701,347 |
| 5 | Ages 26-35 as of June 15, 2012 | 181,229 |
| 6 | Arrived before age 16 | 47,418 |
| 7 | YRIMMIG <= 2007 | 47,418 |
| 8 | Exclude year 2012 | 43,238 |

**Final Analysis Sample:** 43,238 observations

---

## 4. Variable Definitions

### 4.1 Outcome Variable

**Full-time Employment:** UHRSWORK >= 35

**Rationale:** Standard BLS definition of full-time work (35+ hours per week).

### 4.2 Control Variables

| Variable | Definition | IPUMS Code |
|----------|------------|------------|
| male | SEX = 1 | SEX |
| married | MARST = 1 (married, spouse present) | MARST |
| has_children | NCHILD > 0 | NCHILD |
| less_than_hs | EDUC < 6 | EDUC |
| hs_degree | EDUC = 6 | EDUC |
| some_college | EDUC in {7,8,9} | EDUC |
| college_plus | EDUC >= 10 | EDUC |

---

## 5. Statistical Analysis

### 5.1 Main Specification

**Model:** Weighted Least Squares (WLS) with person weights (PERWT)

```
fulltime = b0 + b1*treated + b2*post + b3*(treated*post) + X'gamma + state_FE + year_FE + error
```

**Standard Errors:** Heteroskedasticity-robust (HC1)

### 5.2 Models Estimated

| Model | Controls | State FE | Year FE |
|-------|----------|----------|---------|
| 1 | None | No | No |
| 2 | Demographics | No | No |
| 3 | Demographics | Yes | No |
| 4 (Preferred) | Demographics | Yes | Yes |

### 5.3 Pre-trend Analysis

**Specification:** Interacted treatment with year dummies for pre-period years (2007-2011) relative to 2006.

**Joint F-test:** Tests whether all pre-period treatment-year interactions are jointly zero.

### 5.4 Event Study

**Specification:** Interacted treatment with all year dummies (excluding 2011 as reference year).

---

## 6. Key Results

### 6.1 Main DiD Estimates

| Model | Estimate | SE | p-value |
|-------|----------|-----|---------|
| Basic DiD | 0.059 | 0.012 | <0.001 |
| With Controls | 0.044 | 0.011 | <0.001 |
| State FE | 0.043 | 0.011 | <0.001 |
| Year + State FE | 0.041 | 0.011 | <0.001 |

### 6.2 Preferred Estimate

**Effect Size:** 0.041 (4.1 percentage points)
**Standard Error:** 0.011
**95% CI:** [0.020, 0.062]
**p-value:** 0.0001
**Sample Size:** 43,238

### 6.3 Pre-trend Test

**Joint F-test:** F = 1.03, p = 0.396

**Interpretation:** Fail to reject null of parallel pre-trends.

### 6.4 Heterogeneity Analysis

| Subgroup | Estimate | SE |
|----------|----------|-----|
| Male | 0.046 | 0.013 |
| Female | 0.047 | 0.019 |
| Less than HS | 0.035 | 0.018 |
| HS or more | 0.079 | 0.016 |

---

## 7. Output Files

| File | Description |
|------|-------------|
| analysis.py | Main Python analysis script |
| yearly_employment_rates.csv | Year-by-year employment rates by group |
| event_study_coefficients.csv | Event study point estimates |
| regression_results.csv | Summary of DiD estimates |
| summary_statistics.csv | Descriptive statistics by group/period |
| replication_report_88.tex | LaTeX source for report |
| replication_report_88.pdf | Final compiled report (21 pages) |

---

## 8. Commands Log

### Data Loading and Processing
```python
# Load data with specified dtypes to reduce memory
df = pd.read_csv(data_path, usecols=usecols, dtype=dtypes)

# Apply filters
df = df[df['HISPAN'] == 1]
df = df[df['BPL'] == 200]
df = df[df['CITIZEN'] == 3]

# Calculate age as of June 15, 2012
df['age_2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_2012'] -= 1

# Filter to age groups and DACA criteria
df = df[(df['age_2012'] >= 26) & (df['age_2012'] <= 35)]
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immig'] < 16]
df = df[df['YRIMMIG'] <= 2007]
df = df[df['YEAR'] != 2012]
```

### Regression Analysis
```python
import statsmodels.formula.api as smf

# Basic DiD
model1 = smf.wls('fulltime ~ treated + post + treated_post', data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='HC1')

# DiD with controls and fixed effects
model4 = smf.wls('fulltime ~ treated + treated_post + male + married + has_children +
                  hs_degree + some_college + college_plus + C(state) + C(year_fe)',
                  data=df, weights=df['PERWT'])
results4 = model4.fit(cov_type='HC1')
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_88.tex
pdflatex -interaction=nonstopmode replication_report_88.tex
pdflatex -interaction=nonstopmode replication_report_88.tex
```

---

## 9. Key Analytical Decisions Summary

1. **Undocumented status proxy:** Used CITIZEN = 3 (non-citizen) as proxy since ACS doesn't distinguish documented from undocumented.

2. **Age calculation:** Adjusted for birth quarter to accurately determine age as of June 15, 2012.

3. **Age bandwidth:** Used 26-30 vs 31-35 (5-year windows on each side of cutoff).

4. **Arrival before 16:** Applied DACA requirement using YRIMMIG - BIRTHYR < 16.

5. **Continuous residence:** Used YRIMMIG <= 2007 as proxy for 5+ years residence by 2012.

6. **Exclusion of 2012:** Cannot determine pre/post status without survey month.

7. **Full-time definition:** Standard 35+ hours per week.

8. **Weighting:** Used PERWT for nationally representative estimates.

9. **Standard errors:** HC1 robust to heteroskedasticity.

10. **Preferred model:** Year and state fixed effects with demographic controls.

---

## 10. Interpretation

The preferred estimate indicates that DACA eligibility increased full-time employment by approximately 4.1 percentage points (SE = 0.011, p < 0.001). This represents a 6.7% increase relative to the pre-period treatment group mean of 61.5%.

The parallel trends assumption is supported by:
- Joint F-test failing to reject equal pre-trends (F = 1.03, p = 0.40)
- Event study showing no systematic pre-trends
- Visual inspection of employment trends

Effects are similar by gender but larger for those with HS education or higher (7.9 pp vs 3.5 pp), suggesting DACA's benefits complement human capital.

---

*End of Run Log*
