# Run Log - Replication Study 67
## DACA Effect on Full-Time Employment

**Date:** January 25, 2026
**Analyst:** Replication 67

---

## 1. Overview

This document logs all commands executed and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens.

---

## 2. Data Loading and Initial Exploration

### Files Used:
- `data/data.csv` - Main ACS data file (33,851,424 observations)
- `data/acs_data_dict.txt` - Data dictionary for IPUMS variables
- `data/state_demo_policy.csv` - Optional state-level data (not used)

### Initial Data Check:
```
Total observations: 33,851,424
Years available: 2006-2016
```

---

## 3. Key Analytical Decisions

### Decision 1: Sample Definition
**Choice:** Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200), Non-citizens (CITIZEN=3)

**Rationale:**
- HISPAN=1 identifies Mexican Hispanic ethnicity per instructions
- BPL=200 identifies Mexico as birthplace
- CITIZEN=3 identifies non-citizens; per instructions, we assume non-citizens without immigration papers are undocumented for DACA purposes

### Decision 2: DACA Eligibility Criteria
**Implementation:**
1. **Under 31 on June 15, 2012:** Calculated as (2012 - BIRTHYR), adjusted for BIRTHQTR
2. **Arrived before age 16:** (YRIMMIG - BIRTHYR) < 16
3. **In US since 2007:** YRIMMIG <= 2007 and YRIMMIG > 0

**Rationale:** These criteria directly map the official DACA requirements to available ACS variables.

### Decision 3: Exclusion of 2012
**Choice:** Exclude 2012 from analysis

**Rationale:** DACA was implemented June 15, 2012, and ACS does not record month of interview. Cannot distinguish pre/post treatment in 2012.

### Decision 4: Working Age Restriction
**Choice:** Ages 18-64

**Rationale:** Standard labor economics practice to focus on working-age population. Excludes children and most retirees.

### Decision 5: Full-Time Employment Definition
**Choice:** EMPSTAT=1 (employed) AND UHRSWORK >= 35

**Rationale:** Follows Bureau of Labor Statistics standard definition of full-time work as 35+ usual hours per week.

### Decision 6: Estimation Strategy
**Choice:** Difference-in-differences with year fixed effects and demographic controls

**Rationale:**
- DiD exploits variation in DACA eligibility to identify treatment effect
- Year FE control for common time trends
- Controls for age, age^2, gender, marital status, education address compositional differences

### Decision 7: Standard Errors
**Choice:** Clustered by state (STATEFIP)

**Rationale:** Account for within-state correlation and state-level policy heterogeneity.

### Decision 8: Preferred Specification
**Choice:** Model 3 (Year FE + demographic controls, no state FE)

**Rationale:** Balances control for confounders while avoiding over-fitting. State FE results nearly identical (Model 4).

---

## 4. Sample Construction Summary

| Step | Restriction | N | % of Previous |
|------|-------------|---|---------------|
| 1 | Full ACS 2006-2016 | 33,851,424 | - |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 | 8.7% |
| 3 | Born in Mexico (BPL=200) | 991,261 | 33.7% |
| 4 | Non-citizen (CITIZEN=3) | 701,347 | 70.8% |
| 5 | Exclude 2012 | 636,722 | 90.8% |
| 6 | Ages 18-64 | 547,614 | 86.0% |

**Final sample:** 547,614 observations
- DACA-eligible: 71,347
- DACA-ineligible: 476,267

---

## 5. Commands Executed

### Python Analysis Script (analysis.py)
```python
# Key commands:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('data/data.csv')

# Sample selection
df = df[df['HISPAN'] == 1]  # Hispanic-Mexican
df = df[df['BPL'] == 200]    # Born in Mexico
df = df[df['CITIZEN'] == 3]  # Non-citizen

# DACA eligibility
df['age_jun2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_jun2012'] -= 1
df['under_31_jun2012'] = df['age_jun2012'] < 31
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['YRIMMIG'] > 0)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)
df['daca_eligible'] = (df['under_31_jun2012'] &
                       df['arrived_before_16'] &
                       df['in_us_since_2007'])

# Outcome
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Restrictions
df = df[df['YEAR'] != 2012]  # Exclude 2012
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]  # Working age

# DiD regression
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['did'] = df['daca_eligible_int'] * df['post']

model = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + male + married + hs_or_more',
                data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

### Figure Generation (create_figures.py)
```python
import matplotlib.pyplot as plt

# Event study plot
plt.errorbar(years, coefs, yerr=errors, fmt='o-')
plt.savefig('figure1_event_study.png')

# Trends plot
plt.plot(years, eligible_rate, label='DACA-Eligible')
plt.plot(years, ineligible_rate, label='DACA-Ineligible')
plt.savefig('figure2_trends.png')
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_67.tex
pdflatex -interaction=nonstopmode replication_report_67.tex  # Second pass for references
```

---

## 6. Main Results

### Preferred Estimate (Model 3: Year FE + Controls)
- **DiD Coefficient:** 0.0177 (1.77 percentage points)
- **Standard Error:** 0.0053 (clustered by state)
- **95% CI:** [0.0074, 0.0281]
- **p-value:** 0.0008
- **Sample Size:** 547,614
- **R-squared:** 0.190

### Interpretation
DACA eligibility is associated with a 1.77 percentage point increase in the probability of full-time employment (working 35+ hours per week) among Hispanic-Mexican, Mexican-born non-citizens. This represents a ~4% relative increase from the baseline eligible group rate of 44.1%.

### Robustness
| Specification | Coefficient | SE |
|---------------|-------------|-----|
| Basic DiD | 0.0597 | 0.004 |
| With controls | 0.0242 | 0.005 |
| Year FE (preferred) | 0.0177 | 0.005 |
| Year + State FE | 0.0171 | 0.005 |
| Weighted | 0.0143 | 0.004 |
| Prime age (25-54) | 0.0096 | 0.005 |

---

## 7. Output Files Generated

1. **replication_report_67.tex** - LaTeX source for replication report
2. **replication_report_67.pdf** - Compiled 22-page replication report
3. **run_log_67.md** - This run log
4. **analysis.py** - Main analysis script
5. **create_figures.py** - Figure generation script
6. **regression_results.csv** - Summary of regression results
7. **event_study_results.csv** - Event study coefficients
8. **descriptive_stats.csv** - Descriptive statistics by group/period
9. **sample_by_year.csv** - Sample counts by year
10. **figure1_event_study.png/pdf** - Event study plot
11. **figure2_trends.png/pdf** - Employment trends
12. **figure3_sample_size.png/pdf** - Sample size by year
13. **figure4_coefficients.png/pdf** - Coefficient comparison

---

## 8. Notes and Caveats

1. **Undocumented proxy:** Using non-citizen status as proxy for undocumented likely includes some documented non-citizens, attenuating estimates.

2. **DACA uptake:** Estimates are intent-to-treat effects; not all eligible individuals received DACA.

3. **Pre-trends:** Event study coefficients for 2006-2010 are small and insignificant, supporting parallel trends assumption.

4. **Dynamic effects:** Treatment effects grow over time (0.011 in 2013 to 0.036 in 2016), consistent with gradual DACA rollout.

---

## 9. Session Information

- **Software:** Python 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX:** pdfTeX via MiKTeX
- **Operating System:** Windows

---

*End of Run Log*
