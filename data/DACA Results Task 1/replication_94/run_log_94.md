# Run Log - DACA Replication Study (ID: 94)

## Overview
This log documents all commands, decisions, and analysis steps for the independent replication of the DACA employment effects study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

**Time Period:** Examining effects in 2013-2016 (post-DACA implementation on June 15, 2012)

---

## 1. Data Exploration

### 1.1 Data Files
- `data.csv`: Main ACS data file (6.26 GB, 33,851,424 observations)
- `acs_data_dict.txt`: Data dictionary for IPUMS ACS variables
- `state_demo_policy.csv`: Optional state-level policy data (not used in main analysis)

### 1.2 Initial Data Commands
```bash
# Examined data structure
head -5 data/data.csv

# Used Python to examine data types and distributions
python -c "
import pandas as pd
df = pd.read_csv('data/data.csv', nrows=100000)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
"
```

### 1.3 Key Variables Identified
From the data dictionary (acs_data_dict.txt):
- YEAR: Survey year (2006-2016)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter
- AGE: Age at survey
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight
- SERIAL: Household identifier (used for clustering)

---

## 2. DACA Eligibility Criteria Implementation

From the replication instructions, DACA eligibility requires:

### 2.1 Age at Arrival < 16
```python
age_at_arrival = YRIMMIG - BIRTHYR
arrived_before_16 = (age_at_arrival < 16)
```

### 2.2 Under 31 as of June 15, 2012
```python
# Born after June 15, 1981
under_31_june2012 = (BIRTHYR >= 1982) | ((BIRTHYR == 1981) & (BIRTHQTR >= 3))
```

### 2.3 Continuous US Residence Since June 15, 2007
```python
in_us_since_2007 = (YRIMMIG <= 2007)
```

### 2.4 Not a Citizen
```python
not_citizen = (CITIZEN == 3)
```

### 2.5 Combined Eligibility
```python
daca_eligible = arrived_before_16 & under_31_june2012 & in_us_since_2007 & not_citizen
```

**Key Decision:** Non-citizen status (CITIZEN = 3) is used as a proxy for undocumented status, as the ACS does not distinguish between documented and undocumented non-citizens.

---

## 3. Sample Selection

### 3.1 Target Population
```python
# Filter to target population
df_mex = df[(df['HISPAN'] == 1) &    # Hispanic-Mexican
            (df['BPL'] == 200) &      # Born in Mexico
            (df['CITIZEN'] == 3) &    # Non-citizen
            (df['YRIMMIG'] > 0) &     # Valid immigration year
            (df['AGE'] >= 16) &       # Working age
            (df['AGE'] <= 40)]        # Working age
```

### 3.2 Sample Sizes
- Total ACS observations: 33,851,424
- After target population filter: 387,872
- DACA eligible in sample: 91,428 (23.6%)

---

## 4. Identification Strategy

### 4.1 Design Choice: Difference-in-Differences
**Key Decision:** Use arrival age cutoff for treatment/control definition.

- **Treatment:** Arrived at ages 10-15 (DACA eligible due to arriving before age 16)
- **Control:** Arrived at ages 16-21 (Ineligible due to arriving at age 16+)
- Both groups meet other DACA criteria (under 31 in 2012, in US since 2007)

**Rationale:** This approach compares similar immigrants who differ only in their age at arrival, providing clean identification of the DACA eligibility effect.

### 4.2 Time Periods
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Excluded: 2012 (mid-year implementation)

### 4.3 Analysis Sample
After applying treatment/control definitions:
- Treatment (arrived 10-15): 34,617
- Control (arrived 16-21): 52,568
- Pre-period observations: 54,511
- Post-period observations: 32,674
- Total analysis sample: 87,185

---

## 5. Outcome Variable

### 5.1 Full-Time Employment Definition
```python
fulltime = (UHRSWORK >= 35).astype(int)
```

**Rationale:** 35+ hours per week is the standard BLS definition of full-time employment.

### 5.2 Baseline Full-Time Employment Rates (Pre-period)
- Treatment group: 50.2%
- Control group: 61.5%

---

## 6. Model Specification

### 6.1 Main DiD Equation
```
Y_it = β₀ + β₁×Treated_i + β₂×(Treated_i × Post_t) + X_i'γ + δ_s + τ_t + ε_it
```

Where:
- Y_it = Full-time employment indicator
- Treated_i = 1 if arrived at ages 10-15
- Post_t = 1 if year >= 2013
- X_i = Control variables
- δ_s = State fixed effects
- τ_t = Year fixed effects

### 6.2 Control Variables
- Age (linear and squared)
- Female indicator
- Married indicator
- High school education indicator (EDUCD >= 60)
- College attendance indicator (EDUCD >= 100)
- State fixed effects (51 categories)
- Year fixed effects (10 years, excluding 2012)

### 6.3 Standard Errors
Clustered at household level (SERIAL) to account for within-household correlation.

---

## 7. Commands Executed

### 7.1 Main Analysis Script
```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_94"
python analysis_final.py
```

### 7.2 Key Python Commands in Analysis
```python
# Load data with chunking for memory efficiency
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=500000):
    # Filter and append

# Run main regression
import statsmodels.formula.api as smf
model = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + educ_hs + educ_college + C(state) + C(year_str)',
                data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
```

---

## 8. Results Summary

### 8.1 Main DiD Estimates

| Model | Controls | Effect | SE | 95% CI |
|-------|----------|--------|-----|--------|
| 1 | None | 0.114 | 0.007 | [0.100, 0.128] |
| 2 | Demographics | 0.051 | 0.007 | [0.038, 0.065] |
| 3 | + State FE | 0.051 | 0.007 | [0.038, 0.064] |
| 4 | + Year FE (Preferred) | **0.042** | **0.007** | **[0.029, 0.055]** |

### 8.2 Preferred Estimate
- **Effect Size:** 0.042 (4.2 percentage points)
- **Standard Error:** 0.007
- **95% CI:** [0.029, 0.055]
- **p-value:** < 0.001
- **Sample Size:** 87,185
- **R-squared:** 0.241

### 8.3 Interpretation
DACA eligibility increased full-time employment by 4.2 percentage points, representing an 8.4% relative increase from the baseline rate of 50.2%.

---

## 9. Robustness Checks

### 9.1 Alternative Bandwidths

| Bandwidth | Effect | SE | N |
|-----------|--------|-----|---|
| [13, 18] | 0.051 | 0.008 | 52,275 |
| [12, 19] | 0.049 | 0.007 | 65,994 |
| [10, 21] (preferred) | 0.042 | 0.007 | 87,185 |
| [8, 23] | 0.036 | 0.006 | 102,997 |
| [5, 26] | 0.027 | 0.006 | 120,147 |

**Finding:** Effects are positive and significant across all bandwidths.

### 9.2 By Gender

| Gender | Effect | SE | N |
|--------|--------|-----|---|
| Male | 0.039 | 0.008 | 51,461 |
| Female | 0.044 | 0.011 | 35,724 |

**Finding:** Similar effects for both genders.

### 9.3 Placebo Test
Pre-period analysis (2006-2008 vs 2009-2011):
- Placebo DiD: 0.012 (SE = 0.008)
- p-value: 0.139 (not significant)

**Finding:** No significant differential pre-trends in immediate pre-period.

### 9.4 Event Study
Reference year: 2011

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.046 | 0.014 | 0.001 |
| 2007 | -0.049 | 0.013 | 0.000 |
| 2008 | -0.036 | 0.014 | 0.008 |
| 2009 | -0.018 | 0.014 | 0.179 |
| 2010 | -0.007 | 0.013 | 0.627 |
| 2013 | 0.013 | 0.014 | 0.357 |
| 2014 | 0.012 | 0.014 | 0.370 |
| 2015 | 0.027 | 0.014 | 0.051 |
| 2016 | 0.020 | 0.014 | 0.157 |

**Finding:** Some pre-trends in 2006-2008, but convergence by 2009-2011. Post-DACA effects emerge gradually, largest in 2015-2016.

---

## 10. Key Analytical Decisions

1. **Control group choice:** Used arrival age 16-21 as control (rather than non-Hispanic or citizens) to ensure comparable populations differing only in DACA eligibility.

2. **Bandwidth:** Selected [10, 21] as baseline to balance comparability and sample size; verified robustness to alternatives.

3. **Age restriction:** Limited to ages 16-40 to focus on working-age population and ensure group comparability.

4. **Excluding 2012:** DACA implemented mid-year (June 15), so 2012 observations cannot be clearly classified as pre- or post-treatment.

5. **Clustering:** Household-level clustering accounts for within-family correlation in employment outcomes.

6. **Year fixed effects:** Preferred specification includes year FE to control for common time trends affecting all immigrants.

---

## 11. Files Generated

1. `analysis_final.py` - Main analysis script
2. `results_final.json` - Machine-readable results
3. `replication_report_94.tex` - LaTeX report
4. `replication_report_94.pdf` - Final PDF report (23 pages)
5. `run_log_94.md` - This file

---

## 12. Session Summary

**Start:** Analysis of DACA effects on full-time employment
**End:** Completed analysis with 23-page replication report

**Main Finding:** DACA eligibility increased full-time employment by 4.2 percentage points (95% CI: 2.9-5.5 pp, p < 0.001) among Mexican-born non-citizens, using a difference-in-differences design exploiting the age-at-arrival eligibility cutoff.
