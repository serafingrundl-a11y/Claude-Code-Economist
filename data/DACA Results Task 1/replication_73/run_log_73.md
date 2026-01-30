# Replication Run Log - Session 73

## Overview
This log documents all commands and key decisions for replicating the DACA effect on full-time employment study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Data Sources
- Primary: ACS data from IPUMS USA (2006-2016 one-year files)
- File: data.csv (~33.85 million observations)
- Data dictionary: acs_data_dict.txt
- Optional supplemental: state_demo_policy.csv

## Key Decisions and Methodology

### 1. Sample Selection
**Decision:** Focus on Hispanic-Mexican individuals born in Mexico
- HISPAN = 1 (Mexican)
- BPL = 200 (Mexico)

### 2. DACA Eligibility Criteria
Based on the program requirements:
1. Arrived unlawfully in the US before their 16th birthday
   - Age at immigration = YEAR - YRIMMIG < 16 (need to construct from available variables)
2. Had not yet had their 31st birthday as of June 15, 2012
   - BIRTHYR >= 1981 (born on or after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
   - YRIMMIG <= 2007
4. Were present in the US on June 15, 2012 and did not have lawful status
   - CITIZEN = 3 (Not a citizen) - treating non-citizens without papers as potentially undocumented

**Key assumptions:**
- Cannot directly observe undocumented status; using non-citizen status as proxy
- Cannot observe continuous presence; using immigration year as proxy
- Age at arrival calculated from YRIMMIG and BIRTHYR

### 3. Treatment and Control Groups
- **Treatment group:** Mexican-born Hispanic-Mexican non-citizens who meet DACA age/arrival criteria
- **Control group:** Mexican-born Hispanic-Mexican non-citizens who do NOT meet DACA criteria (e.g., too old or arrived too late)

### 4. Outcome Variable
- Full-time employment: UHRSWORK >= 35

### 5. Analysis Strategy
- Difference-in-differences approach
- Pre-period: 2006-2011 (before DACA implementation June 2012)
- Post-period: 2013-2016 (after DACA implementation)
- Exclude 2012 as transition year (ACS data collected throughout year, cannot distinguish before/after June 15)

### 6. Model Specification
Primary specification:
fulltime_emp = β0 + β1*eligible + β2*post + β3*(eligible×post) + controls + ε

Where β3 is the DiD estimate of DACA's effect on full-time employment.

Controls considered:
- Age, age squared
- Sex
- Education
- Marital status
- State fixed effects
- Year fixed effects

---

## Execution Log

### Step 1: Data Exploration
- Examined data dictionary (acs_data_dict.txt)
- Confirmed variable availability across years (2006-2016)
- Key variables identified:
  - YEAR: Survey year
  - HISPAN/HISPAND: Hispanic origin (1=Mexican)
  - BPL/BPLD: Birthplace (200=Mexico)
  - CITIZEN: Citizenship status (3=Not a citizen)
  - YRIMMIG: Year of immigration
  - BIRTHYR: Birth year
  - AGE: Age at survey
  - UHRSWORK: Usual hours worked per week
  - PERWT: Person weight

### Step 2: Python Analysis Script
Created comprehensive analysis script with the following components:
- Data loading and filtering
- DACA eligibility determination
- Descriptive statistics
- Difference-in-differences estimation
- Robustness checks
- Event study analysis

### Step 3: Data Processing Results
- Total ACS observations: ~33.85 million
- After filtering to Mexican-born Hispanic-Mexican: 991,261
- After working age restriction (16-64): 851,090
- After excluding 2012 (transition year): 771,888
- DACA eligible in final sample: 84,188 (10.9%)

### Step 4: Main Results

#### Preferred Specification (DiD with controls + year FE + state FE)
- **Effect estimate:** 0.0173 (1.73 percentage points)
- **Standard error:** 0.0040
- **95% CI:** [0.0093, 0.0252]
- **p-value:** < 0.0001
- **Sample size:** 771,888
- **R-squared:** 0.2083

#### Robustness Checks
| Specification | Coefficient | SE | N |
|--------------|-------------|-----|--------|
| Basic DiD | 0.0860 | 0.0045 | 771,888 |
| With controls | 0.0232 | 0.0041 | 771,888 |
| Controls + Year FE | 0.0179 | 0.0041 | 771,888 |
| Full model (preferred) | 0.0173 | 0.0040 | 771,888 |
| Age 18-35 subsample | 0.0017 | 0.0048 | 300,712 |
| Males only | 0.0155 | 0.0053 | 408,657 |
| Females only | 0.0106 | 0.0061 | 363,231 |
| Any employment outcome | 0.0343 | 0.0038 | 771,888 |

#### Event Study Results (Base year = 2011)
Pre-treatment coefficients are small and statistically insignificant, supporting parallel trends assumption.
Post-treatment coefficients show gradual increase: 0.014 (2013) → 0.021 (2014) → 0.038 (2015) → 0.041 (2016)

### Step 5: Interpretation
DACA eligibility is associated with a 1.73 percentage point increase in the probability of full-time employment among Mexican-born Hispanic non-citizens. This effect is statistically significant at conventional levels (p < 0.001).

The effect represents approximately a 4% increase relative to the pre-treatment eligible group mean of 45.9%.

---

## Commands Executed

```bash
# Data exploration
head -5 data/data.csv
wc -l data/data.csv

# Run analysis
python analysis.py
```

## Files Generated
- analysis.py: Main analysis script (Python)
- results.json: Structured results output
- run_log_73.md: This log file
- replication_report_73.tex: LaTeX source file (completed)
- replication_report_73.pdf: Final PDF report (26 pages, completed)

---

## Final Deliverables Summary

| File | Description | Status |
|------|-------------|--------|
| replication_report_73.tex | LaTeX source | Complete |
| replication_report_73.pdf | PDF report (26 pages) | Complete |
| run_log_73.md | Run log | Complete |

## Summary of Preferred Estimate

**Research Question:** Effect of DACA eligibility on full-time employment (35+ hours/week) among Mexican-born Hispanic immigrants

**Methodology:** Difference-in-differences with:
- Demographic controls (age, age^2, sex, marital status, education)
- Year fixed effects
- State fixed effects
- Heteroskedasticity-robust standard errors
- ACS person weights

**Key Result:**
- **Effect:** 0.0173 (1.73 percentage points)
- **Standard Error:** 0.0040
- **95% Confidence Interval:** [0.0093, 0.0252]
- **p-value:** < 0.0001
- **Sample Size:** 771,888
- **R-squared:** 0.208

**Interpretation:** DACA eligibility increased the probability of full-time employment by 1.73 percentage points, representing approximately a 4% increase relative to the pre-treatment mean (45.9%) for the eligible group.

---

## Session Complete
Date: 2025-01-25
