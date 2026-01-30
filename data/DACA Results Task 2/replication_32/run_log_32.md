# Run Log: DACA Replication Study (Replication 32)

## Overview
Independent replication examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

---

## Session Log

### 2026-01-26: Initial Setup and Data Exploration

#### 1. Read Replication Instructions
- Extracted instructions from `replication_instructions.docx`
- Research Question: Effect of DACA eligibility (treatment) on probability of full-time employment (working 35+ hours/week)
- Treatment group: Ages 26-30 at time of policy (June 15, 2012)
- Control group: Ages 31-35 at time of policy (otherwise eligible if not for age)
- Pre-period: Before 2012; Post-period: 2013-2016
- Design: Difference-in-Differences

#### 2. Data Files Identified
- `data/data.csv`: Main ACS data file (~6.2GB)
- `data/acs_data_dict.txt`: Data dictionary with variable definitions
- `data/state_demo_policy.csv`: Optional state-level data
- ACS samples: 2006-2016 one-year ACS files

#### 3. Key Variables Identified (from data dictionary)
**Identifiers and Weights:**
- YEAR: Census year (2006-2016)
- PERWT: Person weight
- STATEFIP: State FIPS code

**Demographics:**
- AGE: Age of respondent
- BIRTHYR: Year of birth
- BIRTHQTR: Quarter of birth (1=Q1, 2=Q2, 3=Q3, 4=Q4)
- SEX: 1=Male, 2=Female

**Hispanic/Immigrant Status:**
- HISPAN: Hispanic origin (1=Mexican)
- HISPAND: Detailed Hispanic (100-107 = Mexican categories)
- BPL: Birthplace (200=Mexico)
- BPLD: Detailed birthplace (20000=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration

**Employment:**
- EMPSTAT: Employment status (1=Employed, 2=Unemployed, 3=Not in labor force)
- UHRSWORK: Usual hours worked per week (0=N/A, 1-99=hours)

---

## Sample Selection Criteria

### DACA Eligibility Criteria (from instructions):
1. Arrived in US before 16th birthday
2. Under 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

### Operationalization:
1. **Hispanic-Mexican Mexican-born**: HISPAN=1 AND BPL=200
2. **Non-citizen (proxy for undocumented)**: CITIZEN=3
3. **Arrived before age 16**: (YRIMMIG - BIRTHYR) < 16 OR calculate age at immigration
4. **Age at policy (June 15, 2012)**:
   - Treatment (ages 26-30): Born 1982-1986 (turning 26-30 in 2012)
   - Control (ages 31-35): Born 1977-1981 (turning 31-35 in 2012)
5. **Continuous residence since 2007**: YRIMMIG <= 2007

### Outcome Variable:
- Full-time employment: UHRSWORK >= 35

### Time Periods:
- Pre-treatment: 2006-2011 (excluding 2012 which is ambiguous)
- Post-treatment: 2013-2016

---

## Analysis Plan

### 1. Data Preparation
- Load data and filter to relevant population
- Create treatment and control group indicators
- Create pre/post indicator
- Create full-time employment outcome

### 2. Descriptive Statistics
- Sample sizes by group and period
- Balance table comparing treatment and control
- Trends in full-time employment over time

### 3. Difference-in-Differences Estimation
- Basic DiD without covariates
- DiD with demographic controls (age, sex, education, marital status)
- Robust standard errors (clustered by state)

### 4. Robustness Checks
- Parallel trends test (pre-period)
- Alternative age bandwidths
- Event study specification

---

## Commands Executed

```python
# Initial data exploration
head -5 data/data.csv
# Columns identified: YEAR, SAMPLE, SERIAL, ..., UHRSWORK, INCTOT, FTOTINC, INCWAGE, POVERTY
```

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Use HISPAN=1 for Mexican ethnicity | Detailed version (HISPAND 100-107) too granular; general version captures Mexican |
| Use BPL=200 for Mexico birthplace | Direct coding for Mexico |
| Use CITIZEN=3 for non-citizen | Cannot distinguish documented/undocumented; assume non-citizens are undocumented per instructions |
| Exclude 2012 from analysis | Cannot distinguish pre/post DACA within 2012 (no month data) |
| Age definition based on BIRTHYR | Use birth year to calculate age as of June 15, 2012; adjust for birth quarter |
| Cluster standard errors by state | Account for within-state correlation of errors |
| Preferred model: DiD with demographics | Balance parsimony with control for observable differences |

---

## Analysis Execution

### Data Processing
```bash
# Run analysis script
cd "C:\Users\seraf\DACA Results Task 2\replication_32"
python analysis.py
```

### Sample Selection Results
- Total ACS observations (2006-2016): 33,851,424
- After Hispanic-Mexican filter (HISPAN=1): 2,945,521
- After Mexico birthplace (BPL=200): 991,261
- After non-citizen (CITIZEN=3): 701,347
- After age 26-35 as of June 2012: 181,229
- After arrived before age 16: 47,418
- After continuous residence since 2007: 47,418
- After excluding 2012: 43,238 (final sample)

### Group Sizes
- Treatment group (ages 26-30): 25,470
- Control group (ages 31-35): 17,768
- Pre-treatment (2006-2011): 28,377
- Post-treatment (2013-2016): 14,861

---

## Main Results

### Raw DiD Calculation
| Group | Pre (2006-2011) | Post (2013-2016) | Change |
|-------|-----------------|------------------|--------|
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| **DiD** | | | **+0.059** |

### Regression Results (Preferred Model: DiD with Demographics)

**Main Estimate:**
- DiD coefficient: **0.0466** (4.66 percentage points)
- Standard error: 0.0091 (clustered by state)
- 95% CI: [0.0287, 0.0645]
- P-value: < 0.001

**Interpretation:** DACA eligibility is associated with a 4.7 percentage point increase in the probability of full-time employment among eligible individuals.

### Robustness Checks
1. **Narrower bandwidth (27-29 vs 32-34):** DiD = 0.038 (SE: 0.019)
2. **Placebo test (pre-period only):** DiD = -0.003 (SE: 0.010, p = 0.79) - supports parallel trends
3. **By gender:** Males = 0.036, Females = 0.050

### Event Study
- Pre-treatment coefficients (2006-2010): All small and not significantly different from zero (except 2007 marginally negative)
- Post-treatment coefficients: Positive and increasing over time (2016 = 0.065, significant)
- Supports parallel trends assumption

---

## Output Files Generated

1. **replication_report_32.tex** - LaTeX source for report
2. **replication_report_32.pdf** - Compiled 18-page report
3. **run_log_32.md** - This run log
4. **analysis.py** - Python analysis script
5. **results_summary.json** - Key results in JSON format
6. **regression_tables.txt** - Full regression output
7. **yearly_fulltime_rates.csv** - Trends data
8. **figure_trends.png** - Employment trends figure
9. **figure_event_study.png** - Event study figure

---

## Session Complete

All deliverables have been produced:
- [x] replication_report_32.tex
- [x] replication_report_32.pdf (18 pages)
- [x] run_log_32.md

---
