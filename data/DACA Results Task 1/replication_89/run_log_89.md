# Replication Run Log - Task 89

## Overview
This log documents all commands, decisions, and key steps for the DACA employment replication study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (>=35 hours/week)?

---

## Data Exploration

### Data Source
- File: `data/data.csv`
- Data dictionary: `data/acs_data_dict.txt`
- Source: American Community Survey (ACS) via IPUMS USA
- Years available: 2006-2016
- Total observations: 33,851,424

### Key Variables Identified
1. **Outcome Variable**: UHRSWORK (usual hours worked per week)
   - Full-time employment defined as UHRSWORK >= 35

2. **Sample Selection Variables**:
   - HISPAN: Hispanic origin (1 = Mexican)
   - BPL: Birthplace (200 = Mexico)
   - CITIZEN: Citizenship status (3 = Not a citizen)

3. **DACA Eligibility Variables**:
   - BIRTHYR: Birth year
   - BIRTHQTR: Quarter of birth
   - YRIMMIG: Year of immigration
   - AGE: Current age

---

## DACA Eligibility Criteria (from instructions)

To be DACA-eligible, a person must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet turned 31 by June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 (no lawful status)

**Operationalization decisions**:
- Non-citizen and no immigration papers = undocumented (per instructions)
- CITIZEN == 3 (Not a citizen) will be used
- Age at arrival = YRIMMIG - BIRTHYR (must be < 16)
- Must have arrived by 2007 (continuous residence since June 2007)
- Born 1981 or later (to be under 31 on June 15, 2012)

---

## Analysis Approach

### Identification Strategy
Difference-in-differences (DiD) comparing:
- **Treatment group**: DACA-eligible individuals
- **Control group**: Non-eligible Mexican-born Hispanic non-citizens (similar demographics but don't meet age/arrival criteria)
- **Pre-period**: 2006-2011 (before DACA, June 2012)
- **Post-period**: 2013-2016 (after DACA implementation)
- Note: 2012 is excluded as it straddles DACA implementation

### Sample Restrictions
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Non-citizen (CITIZEN == 3)
4. Working-age population (16-64 years old)

---

## Commands Executed

### Step 1: Initial Data Exploration
```python
# Check data structure
import pandas as pd
df = pd.read_csv('data.csv', nrows=5)
print(df.columns.tolist())

# Check data size
df = pd.read_csv('data.csv', usecols=['YEAR'])
print('Total rows:', len(df))
print('Years:', sorted(df['YEAR'].unique()))
```
Results: 33,851,424 observations across years 2006-2016

---

### Step 2: Sample Selection Process
Applied sequential filters to the data:
1. Initial observations: 33,851,424
2. After Hispanic-Mexican restriction (HISPAN==1): 2,945,521
3. After Mexican-born restriction (BPL==200): 991,261
4. After non-citizen restriction (CITIZEN==3): 701,347
5. After working-age (16-64) restriction: 618,640
6. After excluding 2012: 561,470
7. Final sample (after dropping missing): 561,470

### Step 3: Variable Construction
- DACA-eligible observations: 85,466
- Non-DACA-eligible (control) observations: 476,004
- Post-DACA period observations: 215,678
- Pre-DACA period observations: 345,792

### Step 4: Main Analysis Results
Ran difference-in-differences with state and year fixed effects.

**Preferred Estimate (Model 4):**
- DiD Effect: 0.0276
- Standard Error: 0.0038
- p-value: <0.0001
- 95% CI: [0.020, 0.035]
- Interpretation: DACA eligibility associated with 2.76 percentage point increase in full-time employment

### Step 5: Robustness Checks
| Specification | Effect | SE |
|---------------|--------|-----|
| Ages 18-40 only | 0.0113 | 0.0045 |
| Men only | 0.0231 | 0.0055 |
| Women only | 0.0230 | 0.0059 |
| Employment outcome | 0.0374 | 0.0074 |
| Narrow bandwidth | 0.0262 | 0.0060 |

### Step 6: Event Study
Pre-trends appear relatively flat (no significant trends before 2012).
Post-DACA effects grow over time:
- 2015: 0.038 (p<0.05)
- 2016: 0.039 (p<0.05)

---

## Key Decisions and Justifications

1. **Sample definition**: Restricted to Hispanic-Mexican, Mexican-born, non-citizens as these are the population most likely to be affected by DACA.

2. **Exclusion of 2012**: DACA was implemented June 15, 2012; the ACS does not record interview month, so 2012 observations cannot be cleanly assigned to pre or post period.

3. **Control group**: Used non-DACA-eligible Mexican immigrants (same ethnicity/birthplace/citizenship but don't meet age/arrival criteria) to control for common trends affecting this population.

4. **Outcome variable**: Used UHRSWORK >= 35 as full-time threshold, which is the standard BLS definition.

5. **Clustering**: Standard errors clustered at state level to account for within-state correlation.

---

## Final Deliverables

All required deliverables have been produced:

1. **replication_report_89.tex** - LaTeX source file for the replication report
2. **replication_report_89.pdf** - Compiled 20-page PDF replication report
3. **run_log_89.md** - This log file documenting all commands and decisions

### Supporting Files Created

- `analysis.py` - Python script for all analyses
- `results_main.csv` - Main regression results
- `results_robustness.csv` - Robustness check results
- `results_event_study.csv` - Event study coefficients
- `descriptive_stats.csv` - Descriptive statistics by group and period

---

## Summary of Main Finding

**Preferred Estimate**: DACA eligibility is associated with a **2.76 percentage point** increase in the probability of full-time employment among Mexican-born non-citizen individuals.

- Effect: 0.0276
- SE: 0.0038
- 95% CI: [0.020, 0.035]
- p-value: < 0.0001
- Sample size: 561,470

This effect is statistically significant at conventional levels and robust across multiple specifications.
