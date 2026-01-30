# Run Log - DACA Replication Study (ID: 66)

## Overview

This log documents all commands, key decisions, and analytical choices made during the replication of the DACA effect on full-time employment study.

---

## Date: 2026-01-26

---

## 1. Data Exploration

### 1.1 Initial File Inspection

**Files in data folder:**
- `data.csv` (6.3 GB, 33,851,425 rows) - Main ACS data file
- `acs_data_dict.txt` - Variable codebook from IPUMS
- `state_demo_policy.csv` - Optional state-level policy variables
- `State Level Data Documentation.docx` - Documentation for state data

**Data columns identified in data.csv:**
```
YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, REGION, STATEFIP, PUMA, METRO,
STRATA, GQ, FOODSTMP, PERNUM, PERWT, FAMSIZE, NCHILD, RELATE, RELATED, SEX,
AGE, BIRTHQTR, MARST, BIRTHYR, RACE, RACED, HISPAN, HISPAND, BPL, BPLD,
CITIZEN, YRIMMIG, YRSUSA1, YRSUSA2, HCOVANY, HINSEMP, HINSCAID, HINSCARE,
EDUC, EDUCD, EMPSTAT, EMPSTATD, LABFORCE, CLASSWKR, CLASSWKRD, OCC, IND,
WKSWORK1, WKSWORK2, UHRSWORK, INCTOT, FTOTINC, INCWAGE, POVERTY
```

**Key variables identified for analysis:**
- `HISPAN` = 1 (Mexican Hispanic origin)
- `BPL` = 200 (Born in Mexico)
- `CITIZEN` = 3 (Not a citizen)
- `YRIMMIG` = Year of immigration
- `BIRTHYR` = Birth year
- `BIRTHQTR` = Birth quarter (1-4)
- `UHRSWORK` = Usual hours worked per week
- `PERWT` = Person weight
- `YEAR` = Survey year

---

## 2. Key Analytical Decisions

### 2.1 Sample Definition

**Decision 1: Hispanic-Mexican Ethnicity**
- Restriction: HISPAN = 1
- Rationale: Research question specifies "ethnically Hispanic-Mexican" individuals
- Sample after: 2,945,521 observations

**Decision 2: Birthplace Mexico**
- Restriction: BPL = 200
- Rationale: Research question specifies "Mexican-born" individuals
- Sample after: 991,261 observations

**Decision 3: Non-citizen Status**
- Restriction: CITIZEN = 3
- Rationale: Instructions state "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes." Non-citizen status (CITIZEN=3) is the best available proxy for potentially undocumented status in the ACS.
- Note: This is an imperfect proxy that includes some documented non-citizens
- Sample after: 701,347 observations

**Decision 4: Exclude 2012**
- Restriction: YEAR != 2012
- Rationale: DACA was implemented June 15, 2012. ACS does not record interview month, so we cannot distinguish pre/post within 2012.
- Sample after: 636,722 observations

**Decision 5: Age Groups**
- Treatment: Ages 26-30 as of June 15, 2012 (birth years ~1982-1986)
- Control: Ages 31-35 as of June 15, 2012 (birth years ~1977-1981)
- Rationale: Per instructions, treatment is those under 31 who were DACA-eligible by age; control is those 31-35 who would have been eligible but for age.
- Sample after age restriction: 164,874 observations

**Decision 6: DACA Eligibility Criteria**
- Additional restrictions applied:
  - Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
  - In US since 2007: YRIMMIG <= 2007 and YRIMMIG > 0
- Rationale: DACA required arrival before 16th birthday and continuous presence since June 15, 2007. Using YRIMMIG <= 2007 is a reasonable proxy.
- Final sample: 43,238 observations

### 2.2 Age Calculation

**Decision 7: Age as of June 15, 2012**
- Method:
  - Base age = 2012 - BIRTHYR
  - For birth quarters 3-4 (Jul-Dec), subtract 1 (hadn't had birthday yet by June 15)
- Rationale: Birth quarter allows approximating whether individual had turned a year older by June 15.

### 2.3 Outcome Variable

**Decision 8: Full-Time Employment Definition**
- Definition: UHRSWORK >= 35
- Rationale: Instructions define full-time as "usually working 35 hours per week or more"
- Binary indicator: 1 if full-time, 0 otherwise

### 2.4 Time Periods

**Decision 9: Pre and Post Periods**
- Pre-period: 2006-2011 (6 years)
- Post-period: 2013-2016 (4 years)
- Rationale: Instructions specify examining effects in 2013-2016. Using all available pre-2012 data (back to 2006) provides more statistical power for pre-trends.

### 2.5 Estimation Method

**Decision 10: Weighted Least Squares**
- Used PERWT (person weights) for all main analyses
- Rationale: ACS weights are designed to produce population-representative estimates

**Decision 11: Standard Errors**
- Heteroskedasticity-robust standard errors (HC1)
- Rationale: Outcome is binary, so heteroskedasticity is expected

**Decision 12: Preferred Specification**
- Model includes: treatment indicator, year fixed effects, covariates (male, married, high school education)
- Does not include: state fixed effects (added little explanatory power)
- Rationale: Year fixed effects control for common shocks; covariates improve precision and control for compositional changes

---

## 3. Commands Executed

### 3.1 Data Loading and Preparation

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data (selected columns to manage memory)
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'HISPAN',
               'BPL', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'EDUC',
               'MARST', 'STATEFIP']
df = pd.read_csv('data/data.csv', usecols=cols_needed, low_memory=False)
```

### 3.2 Sample Restrictions

```python
# Hispanic-Mexican
df_mex = df[df['HISPAN'] == 1].copy()

# Born in Mexico
df_mex = df_mex[df_mex['BPL'] == 200].copy()

# Non-citizen
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()

# Exclude 2012
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()

# Calculate age as of June 15, 2012
df_mex['age_june2012'] = 2012 - df_mex['BIRTHYR']
df_mex.loc[df_mex['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# Treatment and control groups
df_mex['treated'] = ((df_mex['age_june2012'] >= 26) & (df_mex['age_june2012'] <= 30)).astype(int)
df_analysis = df_mex[(df_mex['age_june2012'] >= 26) & (df_mex['age_june2012'] <= 35)].copy()

# DACA eligibility: arrived before 16, in US by 2007
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis['daca_eligible_arrival'] = (
    (df_analysis['age_at_immigration'] < 16) &
    (df_analysis['YRIMMIG'] <= 2007) &
    (df_analysis['YRIMMIG'] > 0)
)
df_final = df_analysis[df_analysis['daca_eligible_arrival']].copy()
```

### 3.3 Variable Creation

```python
# Outcome: full-time employment
df_final['fulltime'] = (df_final['UHRSWORK'] >= 35).astype(int)

# Post period indicator
df_final['post'] = (df_final['YEAR'] >= 2013).astype(int)

# Interaction term
df_final['treat_post'] = df_final['treated'] * df_final['post']

# Covariates
df_final['male'] = (df_final['SEX'] == 1).astype(int)
df_final['married'] = (df_final['MARST'].isin([1, 2])).astype(int)
df_final['educ_hs'] = (df_final['EDUC'] >= 6).astype(int)
```

### 3.4 Main Analysis

```python
# Preferred model: DiD with year FE and covariates (weighted)
model4 = smf.wls(
    'fulltime ~ treated + treat_post + C(YEAR) + male + married + educ_hs',
    data=df_final,
    weights=df_final['PERWT']
).fit(cov_type='HC1')
```

---

## 4. Results Summary

### 4.1 Sample Sizes

| Group | Pre-period | Post-period | Total |
|-------|------------|-------------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| **Total** | **28,377** | **14,861** | **43,238** |

### 4.2 Weighted Full-Time Employment Rates

|  | Pre-DACA | Post-DACA | Difference |
|--|----------|-----------|------------|
| Treatment (26-30) | 63.05% | 65.97% | +2.92 pp |
| Control (31-35) | 67.31% | 64.33% | -2.99 pp |
| **DiD Estimate** | | | **+5.90 pp** |

### 4.3 Preferred Regression Estimate

- **Coefficient (treat_post):** 0.0465
- **Standard Error:** 0.0107
- **95% Confidence Interval:** [0.0255, 0.0674]
- **p-value:** < 0.0001

### 4.4 Robustness Checks

| Specification | DiD Estimate | SE |
|--------------|--------------|-----|
| Main (year FE, covariates) | 0.0465 | 0.0107 |
| With state FE | 0.0458 | 0.0107 |
| Employment (any) outcome | 0.0442 | 0.0102 |
| Narrow age bands (27-29 vs 32-34) | 0.0388 | 0.0137 |
| Close years (2010-11 vs 2013-14) | 0.0428 | 0.0168 |
| Males only | 0.0462 | 0.0125 |
| Females only | 0.0466 | 0.0185 |

---

## 5. Output Files Generated

1. `analysis.py` - Main analysis script
2. `replication_report_66.tex` - LaTeX source for report
3. `replication_report_66.pdf` - Compiled report (19 pages)
4. `run_log_66.md` - This log file
5. `event_study.png` - Event study figure
6. `trends.png` - Trends figure
7. `age_immigration.png` - Age at immigration distribution
8. `results_summary.txt` - Text summary of results

---

## 6. Software Used

- Python 3.14
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (visualization)
- LaTeX/pdflatex (document compilation)

---

## 7. Key Interpretation

DACA eligibility is estimated to have increased full-time employment by approximately 4.7 percentage points among Hispanic-Mexican, Mexican-born non-citizens. This represents a 7.4% increase relative to the pre-treatment full-time employment rate of 63.05% for the treatment group.

The event study analysis supports the parallel trends assumption: pre-treatment coefficients are small and statistically insignificant, while post-treatment coefficients are positive and significant.

The effect is robust across multiple specifications including:
- Adding state fixed effects
- Using narrower age bands
- Using years closer to policy implementation
- Analyzing subgroups by gender

---

*End of Run Log*
