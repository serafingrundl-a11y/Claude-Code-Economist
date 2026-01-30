# Run Log - DACA Replication Study (ID: 65)

## Date: 2026-01-25

## Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

---

## 1. Initial Setup and Data Exploration

### 1.1 Data Files Available
- `data.csv`: Main ACS data file (6.26 GB)
- `acs_data_dict.txt`: Data dictionary for ACS variables
- `state_demo_policy.csv`: Optional state-level data
- `State Level Data Documentation.docx`: Documentation for state data

### 1.2 Key Variables Identified from Data Dictionary
Based on the replication instructions and data dictionary:

**Identification Variables:**
- YEAR: Census year (2006-2016)
- SAMPLE: IPUMS sample identifier
- SERIAL: Household serial number
- PERWT: Person weight

**Demographic Variables:**
- AGE: Age
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- SEX: Sex (1=Male, 2=Female)
- MARST: Marital status

**Ethnicity/Nativity Variables:**
- HISPAN: Hispanic origin (1=Mexican)
- HISPAND: Hispanic origin detailed
- BPL: Birthplace (200=Mexico)
- BPLD: Birthplace detailed
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration

**Employment Variables:**
- EMPSTAT: Employment status (1=Employed, 2=Unemployed, 3=Not in labor force)
- LABFORCE: Labor force status
- UHRSWORK: Usual hours worked per week (35+ = full-time)
- WKSWORK2: Weeks worked last year (intervalled)

**Other Variables:**
- EDUC/EDUCD: Educational attainment
- STATEFIP: State FIPS code
- METRO: Metropolitan status

---

## 2. DACA Eligibility Criteria

From the replication instructions, DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

### 2.1 Operationalization of DACA Eligibility

**Age at arrival < 16:**
- Calculate age at arrival = YRIMMIG - BIRTHYR
- Require age_at_arrival < 16

**Under 31 as of June 15, 2012:**
- Born after June 15, 1981
- Using BIRTHYR and BIRTHQTR:
  - If BIRTHYR > 1981: eligible
  - If BIRTHYR == 1981 and BIRTHQTR >= 3: eligible (born Jul-Dec 1981)
  - If BIRTHYR == 1981 and BIRTHQTR <= 2: Not eligible (born Jan-Jun 1981 or earlier)

**Continuous residence since June 15, 2007:**
- YRIMMIG <= 2007

**Present in US on June 15, 2012 without lawful status:**
- CITIZEN == 3 (Not a citizen)
- Assume non-citizens without naturalization are undocumented (per instructions)

### 2.2 Sample Restrictions
- Hispanic-Mexican ethnicity: HISPAN == 1 (Mexican)
- Born in Mexico: BPL == 200
- Non-citizen: CITIZEN == 3
- Years 2006-2016 (pre-period: 2006-2011, post-period: 2013-2016)
- Exclude 2012 as transition year (cannot distinguish pre/post DACA within 2012)

---

## 3. Research Design

### 3.1 Identification Strategy: Difference-in-Differences

**Treatment Group:**
DACA-eligible Hispanic-Mexican, Mexican-born non-citizens

**Control Group:**
Non-eligible Hispanic-Mexican, Mexican-born non-citizens (those who don't meet age/arrival criteria)

**Pre-Period:** 2006-2011 (before DACA)
**Post-Period:** 2013-2016 (after DACA implementation)

### 3.2 Outcome Variable
Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise

---

## 4. Analysis Commands and Decisions

### 4.1 Data Loading

```python
# Load full ACS data
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424
# Years: 2006-2016
```

### 4.2 Sample Construction

```python
# Sample restrictions applied sequentially:
# 1. Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1]  # N = 2,945,521

# 2. Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]  # N = 991,261

# 3. Non-citizens (CITIZEN == 3)
df = df[df['CITIZEN'] == 3]  # N = 701,347

# 4. Exclude 2012 transition year
df = df[df['YEAR'] != 2012]  # N = 636,722

# 5. Working age (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]  # N = 561,470
```

### 4.3 Variable Creation

```python
# Post-DACA indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA Eligibility (all three criteria must be met):
# 1. Arrived before age 16
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)

# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
df['under_31_june2012'] = ((df['BIRTHYR'] > 1981) |
                           ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)

# 3. In US since June 2007
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# Combined eligibility
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                       (df['under_31_june2012'] == 1) &
                       (df['in_us_since_2007'] == 1)).astype(int)

# Outcome: Full-time employment
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# DiD interaction
df['daca_x_post'] = df['daca_eligible'] * df['post']

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['educ_lesshs'] = (df['EDUC'] <= 5).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecoll'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['educ_ba_plus'] = (df['EDUC'] >= 10).astype(int)
```

### 4.4 Regression Specifications

Six models estimated with progressively more controls:

**Model 1: Basic DiD (no controls)**
```
fulltime ~ daca_eligible + post + daca_x_post
```

**Model 2: + Demographics**
```
fulltime ~ daca_eligible + post + daca_x_post + AGE + AGE^2 + female + married
```

**Model 3: + Education**
```
fulltime ~ ... + educ_hs + educ_somecoll + educ_ba_plus
```

**Model 4: + State Fixed Effects**
```
fulltime ~ ... + C(STATEFIP)
```

**Model 5: + Year Fixed Effects**
```
fulltime ~ ... + C(YEAR)
```

**Model 6: Weighted (Preferred Specification)**
```
Same as Model 5, but estimated using WLS with PERWT as weights
```

All models use heteroskedasticity-robust (HC1) standard errors.

---

## 5. Key Analytical Decisions

### Decision 1: Exclusion of 2012
**Choice:** Exclude year 2012 from the analysis entirely.
**Justification:** DACA was implemented mid-year (June 15, 2012), and the ACS does not record the month of survey response. Including 2012 would introduce measurement error as some observations would be pre-treatment and others post-treatment, with no way to distinguish them.

### Decision 2: Control Group Definition
**Choice:** Use non-eligible Mexican-born non-citizens as the control group, rather than a different immigrant population (e.g., non-Mexican non-citizens).
**Justification:** This maximizes comparability between treatment and control groups on unobserved characteristics. Both groups face similar barriers to employment as undocumented immigrants from Mexico, but differ in DACA eligibility based on age/arrival timing.

### Decision 3: Age Restriction
**Choice:** Restrict sample to ages 16-64 (working age).
**Justification:** Standard practice in labor economics. Including children or elderly would introduce observations where the outcome is structurally determined (cannot work or retired).

### Decision 4: Full-Time Definition
**Choice:** Define full-time as UHRSWORK >= 35 hours per week.
**Justification:** This is the standard BLS definition of full-time employment and matches the research question specification.

### Decision 5: Linear Probability Model
**Choice:** Use LPM rather than logit/probit.
**Justification:** DiD coefficients from LPM are easily interpretable as average partial effects. The concern about predicted probabilities outside [0,1] is minimal for estimating treatment effects rather than predictions.

### Decision 6: Person Weights
**Choice:** Use PERWT in the preferred specification to make estimates representative.
**Justification:** ACS is a complex survey design, and weights adjust for sampling design and non-response. Weighted estimates represent population effects.

### Decision 7: Not Using State-Level Data
**Choice:** Did not incorporate the optional state_demo_policy.csv file.
**Justification:** State fixed effects already control for time-invariant state characteristics. Including state-level controls could introduce endogeneity if states with different policies also have different DACA take-up rates.

---

## 6. Results Summary

### 6.1 Sample Characteristics
- Final sample: 561,470 observations
- DACA-eligible: 83,611 (14.9%)
- Non-eligible: 477,859 (85.1%)

### 6.2 Raw DiD Estimate
|  | Pre-DACA | Post-DACA | Change |
|---|---|---|---|
| DACA-Eligible | 0.431 | 0.496 | +0.065 |
| Non-Eligible | 0.604 | 0.579 | -0.025 |
| **DiD** | | | **+0.090** |

### 6.3 Regression Results Summary
| Model | Coefficient | Std. Error |
|-------|-------------|------------|
| (1) Basic DiD | 0.0902 | 0.0038 |
| (2) + Demographics | 0.0421 | 0.0035 |
| (3) + Education | 0.0387 | 0.0035 |
| (4) + State FE | 0.0382 | 0.0035 |
| (5) + Year FE | 0.0327 | 0.0035 |
| (6) Weighted (Preferred) | 0.0304 | 0.0042 |

### 6.4 Preferred Estimate
- **Effect size:** 0.0304 (3.04 percentage points)
- **Standard error:** 0.0042
- **95% Confidence Interval:** [0.0222, 0.0386]
- **P-value:** < 0.0001
- **Sample size:** 561,470

### 6.5 Robustness Checks
- Alternative outcome (any employment): 0.0401 (SE: 0.0041)
- Alternative outcome (labor force participation): 0.0426 (SE: 0.0039)
- Males only: 0.0263 (SE: 0.0055)
- Females only: 0.0258 (SE: 0.0062)

### 6.6 Event Study
Pre-DACA coefficients (relative to 2011):
- 2006: -0.015 (n.s.)
- 2007: -0.014 (n.s.)
- 2008: -0.001 (n.s.)
- 2009: 0.005 (n.s.)
- 2010: 0.008 (n.s.)

Post-DACA coefficients (relative to 2011):
- 2013: 0.012 (n.s.)
- 2014: 0.023**
- 2015: 0.039***
- 2016: 0.041***

**Interpretation:** No evidence of differential pre-trends. Effect emerges after DACA implementation and grows over time.

---

## 7. Files Generated

1. `analysis.py` - Main analysis script
2. `analysis_results.json` - Stored numerical results
3. `descriptive_stats.csv` - Summary statistics
4. `fulltime_by_year.csv` - Year-by-year employment rates
5. `replication_report_65.tex` - LaTeX report
6. `replication_report_65.pdf` - Final PDF report (18 pages)
7. `run_log_65.md` - This log file

---

## 8. Conclusion

DACA eligibility is associated with a 3.0 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens. This effect is statistically significant (p < 0.001), robust to controls and fixed effects, and supported by an event study showing no pre-trends. The results suggest that legal work authorization through DACA facilitated transitions into formal, full-time employment.

---
