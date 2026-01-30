# Run Log - Replication 66: DACA Impact on Full-Time Employment

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

## Data Overview
- **Source**: American Community Survey (ACS) via IPUMS, 2006-2016
- **Primary file**: data.csv (~33.8 million observations)
- **Supplementary**: state_demo_policy.csv (state-level policy variables)
- **Data dictionary**: acs_data_dict.txt

---

## Analysis Design Decisions

### 1. Identification Strategy: Difference-in-Differences (DiD)

**Rationale**: DACA eligibility is determined by a set of criteria based on age, year of arrival, and citizenship status. This creates a natural comparison between:
- **Treatment group**: Hispanic-Mexican, Mexican-born non-citizens who meet DACA eligibility criteria
- **Control group**: Hispanic-Mexican, Mexican-born non-citizens who are similar but do NOT meet eligibility criteria

**Time periods**:
- Pre-treatment: 2006-2011 (before DACA implementation)
- Treatment transition: 2012 (DACA announced June 15, 2012 - excluded due to timing ambiguity)
- Post-treatment: 2013-2016

### 2. DACA Eligibility Criteria (as of June 15, 2012)

To be DACA-eligible, an individual must:
1. Arrived in the US before age 16
2. Born after June 15, 1981 (not yet 31 on June 15, 2012)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 with no lawful status
5. Not a citizen

**Operationalization**:
- Age at arrival = YRIMMIG - BIRTHYR
- Must have arrived before age 16: age_at_arrival < 16
- Born after June 1981: BIRTHYR > 1981 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
- CITIZEN = 3 (not a citizen, and not naturalized)
- Must have been in US since 2007: YRIMMIG <= 2007

### 3. Sample Restrictions

1. Hispanic-Mexican ethnicity: HISPAN = 1
2. Born in Mexico: BPL = 200 (Mexico)
3. Non-citizen (potential undocumented): CITIZEN = 3
4. Working-age population: AGE 16-64
5. Immigrated to US: YRIMMIG > 0
6. Exclude 2012: YEAR != 2012

### 4. Outcome Variable

**Full-time employment**: UHRSWORK >= 35
- Binary indicator (1 = full-time, 0 = not full-time or not employed)

### 5. Model Specification

Basic DiD model:
```
Y_ist = β0 + β1*Eligible_i + β2*Post_t + β3*(Eligible_i × Post_t) + ε_ist
```

Full specification:
```
Y_ist = β0 + β1*Eligible_i + β3*(Eligible_i × Post_t) + X_i'γ + δ_s + θ_t + ε_ist
```

Where:
- Y_ist = full-time employment indicator
- Eligible_i = 1 if DACA-eligible based on age/arrival criteria
- Post_t = 1 if year >= 2013
- X_i = controls (age, age², sex, education, marital status, years in US)
- δ_s = state fixed effects
- θ_t = year fixed effects
- β3 = coefficient of interest (DiD estimator)

---

## Commands and Execution Log

### Step 1: Environment Setup
```
Date: 2026-01-25
Working directory: C:\Users\seraf\DACA Results Task 1\replication_66
```

### Step 2: Data Loading
```python
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424
# Years: 2006-2016
```

### Step 3: Sample Construction
```
Initial ACS sample (2006-2016):     33,851,424
Hispanic-Mexican (HISPAN=1):         2,945,521 (8.7%)
Born in Mexico (BPL=200):              991,261 (33.7%)
Not a citizen (CITIZEN=3):             701,347 (70.8%)
Valid immigration year:                701,347 (100%)
Working age (16-64):                   618,640 (88.2%)
Exclude 2012:                          561,470 (90.8%)
Final analytic sample:                 561,470
```

### Step 4: DACA Eligibility Coding
```
DACA Eligibility Breakdown:
  Arrived before age 16:         140,390 (25.0%)
  Born after June 1981:          164,087 (29.2%)
  In US since 2007:              528,852 (94.2%)
  DACA Eligible (all criteria):   83,611 (14.9%)
```

### Step 5: Outcome Variable Construction
```
Full-time employment rate: 57.4%
Employment rate: 64.0%
Labor force participation: 70.0%

Pre-DACA observations (2006-2011): 345,792
Post-DACA observations (2013-2016): 215,678
```

### Step 6: Descriptive Statistics

**Weighted Full-time Employment Rates:**
| Group | Pre (2006-11) | Post (2013-16) | Difference |
|-------|---------------|----------------|------------|
| Not Eligible | 62.76% | 60.13% | -2.63 pp |
| Eligible | 45.22% | 52.14% | +6.92 pp |

**Simple DiD Estimate:** +9.56 percentage points

### Step 7: Regression Results

**Model 1: Basic DiD (no controls)**
```
eligible_post: 0.0956 (SE: 0.0041, p < 0.001)
```

**Model 2: DiD with Demographic Controls**
```
eligible_post: 0.0381 (SE: 0.0043, p < 0.001)
Controls: age, age², female, married, educ_hs, educ_college, years_in_us
```

**Model 3: DiD with Demographics + Year FE**
```
eligible_post: 0.0309 (SE: 0.0039, p < 0.001)
```

**Model 4: Full Model (State + Year FE) - PREFERRED**
```
eligible_post: 0.0301 (SE: 0.0039, p < 0.001)
95% CI: [0.0224, 0.0378]
```

### Step 8: Robustness Checks

| Specification | Coefficient | SE | p-value |
|--------------|-------------|-----|---------|
| Full-time (preferred) | 0.0301 | 0.0039 | <0.001 |
| Employment | 0.0409 | 0.0070 | <0.001 |
| Labor Force Participation | 0.0431 | 0.0074 | <0.001 |
| Males Only | 0.0262 | 0.0057 | <0.001 |
| Females Only | 0.0264 | 0.0060 | <0.001 |

### Step 9: Event Study Results (Reference: 2011)

| Year | Coefficient | SE | 95% CI | Sig |
|------|------------|-----|--------|-----|
| 2006 | -0.0155 | 0.0135 | [-0.042, 0.011] | |
| 2007 | -0.0142 | 0.0077 | [-0.029, 0.001] | * |
| 2008 | -0.0012 | 0.0129 | [-0.027, 0.024] | |
| 2009 | 0.0052 | 0.0117 | [-0.018, 0.028] | |
| 2010 | 0.0076 | 0.0156 | [-0.023, 0.038] | |
| 2011 | 0.0000 | -- | -- | (ref) |
| 2013 | 0.0122 | 0.0108 | [-0.009, 0.033] | |
| 2014 | 0.0223 | 0.0145 | [-0.006, 0.051] | |
| 2015 | 0.0381 | 0.0133 | [0.012, 0.064] | *** |
| 2016 | 0.0398 | 0.0117 | [0.017, 0.063] | *** |

**Interpretation**: Pre-trends are not significantly different from zero (supporting parallel trends assumption). Effects emerge and grow following DACA implementation, becoming statistically significant in 2015-2016.

---

## Final Results Summary

### Preferred Estimate
- **DiD Coefficient**: 0.0301 (3.01 percentage points)
- **Standard Error**: 0.0039
- **95% Confidence Interval**: [0.0224, 0.0378] or [2.24, 3.78] pp
- **P-value**: < 0.001
- **Sample Size**: 561,470

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately **3.0 percentage points** among Hispanic-Mexican, Mexican-born non-citizens. Given a baseline full-time employment rate of 45.2% for eligible individuals in the pre-DACA period, this represents approximately a **6.6% relative increase**.

---

## Key Decisions Made

1. **Exclusion of 2012**: Excluded because DACA was announced mid-year (June 15), making it impossible to distinguish pre/post status.

2. **Control group definition**: Used Mexican-born non-citizens who arrived as children but do not meet all eligibility criteria (too old, arrived too late, or arrived after age 16).

3. **Weighted estimation**: Used person weights (PERWT) for population-representative estimates.

4. **Standard errors**: Clustered at state level to account for within-state correlation.

5. **Fixed effects**: Included state and year fixed effects in preferred specification to control for unobserved state-level factors and national trends.

6. **Birth quarter handling**: Used BIRTHQTR to precisely identify those born after June 1981 (BIRTHQTR >= 3 means July-September or later).

---

## Output Files Generated

1. `analysis.py` - Main analysis script
2. `regression_results.csv` - Summary of all regression results
3. `event_study_results.csv` - Event study coefficients
4. `descriptive_stats.csv` - Descriptive statistics by group and period
5. `replication_report_66.tex` - LaTeX source for report
6. `replication_report_66.pdf` - Final compiled report (~22 pages)
7. `run_log_66.md` - This log file

---

*Log completed: 2026-01-25*
