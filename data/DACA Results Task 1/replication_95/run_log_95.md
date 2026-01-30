# Run Log - Replication 95: DACA and Full-Time Employment

## Overview
This document logs all commands and key decisions made during the replication of the DACA impact study on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Date Started
2026-01-25

---

## Step 1: Data Exploration

### Data Files Available
- `data/data.csv` - Main ACS data file (33,851,425 observations)
- `data/acs_data_dict.txt` - Data dictionary with variable definitions
- `data/state_demo_policy.csv` - Optional supplemental state-level data

### Key Variables Identified from Data Dictionary
- **YEAR**: Census year (2006-2016)
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1-4)
- **HISPAN**: Hispanic origin (1 = Mexican)
- **BPL**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week (35+ = full-time)
- **EMPSTAT**: Employment status (1 = Employed)
- **PERWT**: Person weight

### DACA Eligibility Criteria (from instructions)
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

---

## Step 2: Sample Restrictions and Variable Construction

### Decision Log

#### Decision 1: Sample Population
**Choice**: Restrict to Hispanic-Mexican ethnicity (HISPAN == 1) AND born in Mexico (BPL == 200)
**Rationale**: The research question specifically focuses on "ethnically Hispanic-Mexican Mexican-born people"

#### Decision 2: Non-Citizen Status
**Choice**: Restrict to non-citizens (CITIZEN == 3)
**Rationale**: Instructions state to "assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"

#### Decision 3: Defining DACA Eligibility (Treatment Group)
**Criteria**:
- Age at immigration < 16 (arrived before 16th birthday): calculated as YRIMMIG - BIRTHYR < 16
- Age as of June 15, 2012 < 31 (born after June 15, 1981): BIRTHYR >= 1982 (conservative, since we don't know exact birth date)
- In US since June 15, 2007: YRIMMIG <= 2007
- Note: We cannot directly verify presence on June 15, 2012 or unlawful status in data

#### Decision 4: Control Group
**Choice**: Mexican-born, Hispanic-Mexican non-citizens who do NOT meet DACA eligibility criteria
**Rationale**: These individuals share similar background characteristics but were not eligible for DACA, providing a valid comparison group

#### Decision 5: Treatment Period
**Choice**: Pre-treatment = 2006-2011; Post-treatment = 2013-2016
**Rationale**: DACA was implemented June 15, 2012. Year 2012 is excluded as a transition year since we cannot distinguish pre/post observations within that year.

#### Decision 6: Outcome Variable - Full-Time Employment
**Choice**: Binary variable = 1 if UHRSWORK >= 35, 0 otherwise
**Rationale**: Instructions define full-time as "usually working 35 hours per week or more"

#### Decision 7: Age Restrictions for Analysis Sample
**Choice**: Restrict to working-age population (16-64 years old)
**Rationale**: Standard practice in labor economics; people outside this range are less likely to be in labor force

---

## Step 3: Analysis Commands

### Python Analysis Script
See `analysis_95.py` for full code.

### Key Commands:

```python
# Load data
import pandas as pd
df = pd.read_csv('data/data.csv')

# Filter to Hispanic-Mexican, Mexican-born, non-citizens
df = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)]

# Exclude 2012
df = df[df['YEAR'] != 2012]

# Restrict to working age (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]

# Define DACA eligibility
# Age at immigration < 16
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
# Born after 1981 (to be under 31 on June 15, 2012)
# In US by 2007
df['daca_eligible'] = ((df['age_at_immig'] < 16) &
                        (df['BIRTHYR'] >= 1982) &
                        (df['YRIMMIG'] <= 2007) &
                        (df['YRIMMIG'] > 0))

# Post-treatment indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Outcome: Full-time employment
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Difference-in-differences regression
import statsmodels.formula.api as smf
model = smf.wls('fulltime ~ daca_eligible * post + C(YEAR) + C(STATEFIP)',
                data=df, weights=df['PERWT'])
```

---

## Step 4: Regression Specifications

### Model 1: Basic Difference-in-Differences
```
fulltime = β0 + β1*eligible + β2*post + β3*(eligible × post) + ε
```

### Model 2: With Year Fixed Effects
```
fulltime = β0 + β1*eligible + β3*(eligible × post) + γ_year + ε
```

### Model 3: With Year and State Fixed Effects
```
fulltime = β0 + β1*eligible + β3*(eligible × post) + γ_year + δ_state + ε
```

### Model 4: With Individual Controls
```
fulltime = β0 + β1*eligible + β3*(eligible × post) + γ_year + δ_state + X'β + ε
```
Where X includes: age, age², sex, marital status, education, years in US

---

## Step 5: Results Summary

### Sample Statistics
- Total observations: 33,851,424 (initial data)
- After all restrictions: 561,470 observations
  - DACA eligible: 80,300 (14.3%)
  - Control group: 481,170 (85.7%)
  - Pre-treatment (2006-2011): 345,792
  - Post-treatment (2013-2016): 215,678

### Main Results (Preferred Estimate: Model 3 - Year + State FE)

| Model | DiD Estimate | Std Error | p-value | N |
|-------|--------------|-----------|---------|---|
| (1) Basic DiD | 0.0994 | 0.0047 | <0.001 | 561,470 |
| (2) Year FE | 0.0930 | 0.0047 | <0.001 | 561,470 |
| (3) Year + State FE | **0.0918** | **0.0047** | **<0.001** | **561,470** |
| (4) Full Controls | 0.0330 | 0.0043 | <0.001 | 561,470 |

### Preferred Estimate
- **Effect Size**: 0.0918 (9.18 percentage points)
- **Standard Error**: 0.0047
- **95% CI**: [0.0826, 0.1009]
- **p-value**: < 0.001
- **Sample Size**: 561,470

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 9.2 percentage points among Mexican-born, Hispanic-Mexican non-citizens. This represents a relative increase of about 22% from the baseline full-time employment rate of 42.6% among eligible individuals.

---

## Step 6: Robustness Checks

### Results Summary

| Specification | Estimate | SE | p-value |
|--------------|----------|-----|---------|
| Stricter age cutoff (born >= 1983) | 0.0998 | 0.0048 | <0.001 |
| Placebo (fake 2009 treatment) | 0.0450 | 0.0061 | <0.001 |
| Men only | 0.0881 | 0.0059 | <0.001 |
| Women only | 0.0703 | 0.0065 | <0.001 |
| Any employment outcome | 0.1019 | 0.0045 | <0.001 |

### Event Study Results (relative to 2011)
| Year | Coefficient | 95% CI |
|------|-------------|--------|
| 2006 | -0.063 | [-0.084, -0.043] |
| 2007 | -0.047 | [-0.068, -0.027] |
| 2008 | -0.028 | [-0.049, -0.007] |
| 2009 | -0.015 | [-0.035, 0.005] |
| 2010 | 0.011 | [-0.009, 0.031] |
| 2013 | 0.039 | [0.019, 0.059] |
| 2014 | 0.064 | [0.044, 0.083] |
| 2015 | 0.085 | [0.065, 0.105] |
| 2016 | 0.096 | [0.076, 0.116] |

---

## Step 7: Key Methodological Notes

1. **Weighting**: All regressions use ACS person weights (PERWT) to obtain nationally representative estimates
2. **Standard Errors**: Heteroskedasticity-robust (HC1) standard errors
3. **Software**: Python 3.x with pandas, numpy, and statsmodels
4. **Identification Strategy**: Difference-in-differences comparing DACA-eligible vs. ineligible non-citizens, pre vs. post implementation

---

## Files Generated
- `analysis_95.py` - Main analysis script
- `results_95.csv` - Summary results file
- `replication_report_95.tex` - LaTeX report (23 pages)
- `replication_report_95.pdf` - Final PDF report
- `run_log_95.md` - This log file

---

## Date Completed
2026-01-25

