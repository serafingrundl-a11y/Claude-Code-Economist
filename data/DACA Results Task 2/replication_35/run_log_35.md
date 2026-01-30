# Run Log - DACA Replication Study (ID: 35)

## Session Start
Date: 2026-01-26

## Task Overview
Replicate analysis of DACA's effect on full-time employment among Hispanic-Mexican, Mexican-born individuals in the US.

### Research Design:
- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at DACA implementation
- **Outcome**: Full-time employment (usually working 35+ hours/week)
- **Method**: Difference-in-Differences
- **Post-treatment Years**: 2013-2016

---

## Step 1: Data Exploration

### Files in data folder:
- `data.csv` - Main ACS data (33,851,424 rows)
- `acs_data_dict.txt` - Data dictionary
- `state_demo_policy.csv` - Optional state-level data (not used in main analysis)

### Key Variables Identified:
1. **Identification**:
   - YEAR: Survey year (2006-2016)
   - BIRTHYR: Birth year
   - BIRTHQTR: Birth quarter (for precise age calculation)

2. **Eligibility Criteria**:
   - HISPAN: Hispanic origin (1 = Mexican)
   - BPL: Birthplace (200 = Mexico)
   - CITIZEN: Citizenship status (3 = Not a citizen)
   - YRIMMIG: Year of immigration

3. **Outcome**:
   - UHRSWORK: Usual hours worked per week (>=35 = full-time)

4. **Controls**:
   - SEX, EDUC, MARST, STATEFIP, PERWT

---

## Step 2: Data Cleaning and Sample Construction

### Sample Selection Process:
| Step | Criterion | Observations |
|------|-----------|-------------|
| 1 | Full ACS 2006-2016 | 33,851,424 |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 |
| 3 | Born in Mexico (BPL=200) | 991,261 |
| 4 | Non-citizen (CITIZEN=3) | 701,347 |
| 5 | Ages 26-35 at DACA | 181,229 |
| 6 | Arrived before age 16 | 47,418 |
| 7 | Arrived by 2007 | 47,418 |
| 8 | Excluding 2012 | 43,238 |

### Key Decisions:
1. **Age calculation**: Used BIRTHYR and BIRTHQTR to calculate age at June 15, 2012. Individuals born in Q3-Q4 assumed not to have had birthday yet.
2. **Undocumented proxy**: Used CITIZEN=3 (not a citizen) as proxy since ACS cannot distinguish documented from undocumented.
3. **Exclusion of 2012**: Removed because DACA implemented mid-year and ACS doesn't report interview month.

---

## Step 3: Variable Construction

### Outcome Variable:
```
fulltime = 1 if UHRSWORK >= 35, else 0
```

### Treatment Variables:
```
treated = 1 if age_at_daca between 26 and 30
post = 1 if YEAR >= 2013
treated_post = treated * post (DiD interaction)
```

### Control Variables:
- female = 1 if SEX == 2
- married = 1 if MARST in [1, 2]
- educ_hs = 1 if EDUC >= 6
- educ_college = 1 if EDUC >= 10

---

## Step 4: Analysis

### Summary Statistics (Full-time employment rates):
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.6147 | 0.6339 | +0.0192 |
| Control (31-35) | 0.6461 | 0.6136 | -0.0324 |
| **DiD** | | | **+0.0516** |

### Main Regression Results:

#### Model Specifications:
1. Basic OLS (unweighted): 0.0516 (SE: 0.0100)
2. WLS with survey weights: 0.0590 (SE: 0.0098)
3. WLS with controls: 0.0466 (SE: 0.0090)
4. WLS with year FE: 0.0449 (SE: 0.0090)
5. WLS with clustered SE (preferred): **0.0466 (SE: 0.0090)**

### Preferred Estimate:
- **Effect size**: 0.0466 (4.66 percentage points)
- **Standard error**: 0.0090 (clustered by state)
- **95% CI**: [0.0289, 0.0643]
- **p-value**: < 0.001
- **Sample size**: 43,238

---

## Step 5: Robustness Checks

### 1. Logit Model (Marginal Effects)
- Estimate: 0.0514
- SE: 0.0105
- Consistent with LPM estimate

### 2. Narrower Age Bandwidth (27-29 vs 32-34)
- Estimate: 0.0374
- SE: 0.0187
- p-value: 0.045
- N: 25,606

### 3. Placebo Test (2009 as fake treatment)
- Estimate: -0.0028
- SE: 0.0103
- p-value: 0.785
- Supports parallel trends assumption

### 4. Effects by Gender
- Males: 0.0367 (SE: 0.0096), N = 24,243
- Females: 0.0505 (SE: 0.0157), N = 18,995

---

## Step 6: Event Study Analysis

### Event Study Coefficients (Reference: 2011)
| Year | Coefficient | SE | Sig |
|------|------------|-----|-----|
| 2006 | 0.0069 | 0.0272 | |
| 2007 | -0.0314 | 0.0160 | ** |
| 2008 | 0.0080 | 0.0207 | |
| 2009 | -0.0084 | 0.0221 | |
| 2010 | -0.0134 | 0.0246 | |
| 2013 | 0.0348 | 0.0227 | |
| 2014 | 0.0356 | 0.0170 | ** |
| 2015 | 0.0206 | 0.0177 | |
| 2016 | 0.0652 | 0.0208 | *** |

**Interpretation**: Most pre-treatment coefficients are small and insignificant, supporting parallel trends. Post-treatment effects are positive and grow over time, with the largest effect in 2016.

---

## Step 7: Outputs Generated

### Analysis Outputs:
1. `regression_results.csv` - Model comparison table
2. `summary_stats.csv` - Summary statistics by group/period
3. `event_study.csv` - Event study coefficients

### Final Deliverables:
1. `replication_report_35.tex` - LaTeX source (720 lines)
2. `replication_report_35.pdf` - Compiled report (25 pages)
3. `run_log_35.md` - This file
4. `analysis.py` - Python analysis script

---

## Summary of Key Findings

### Main Result:
DACA eligibility increased the probability of full-time employment by **4.66 percentage points** (95% CI: 2.89 to 6.43 pp) among Hispanic-Mexican, Mexican-born, non-citizen individuals who arrived before age 16 and by 2007.

### Interpretation:
- Represents a ~7.5% increase relative to pre-DACA baseline of 61.5%
- Effect is statistically significant (p < 0.001)
- Robust to alternative specifications and placebo tests
- Effects appear to strengthen over time (largest in 2016)
- Positive for both men and women

### Limitations:
1. Cannot directly observe undocumented status
2. Some pre-trend concern (2007 coefficient significant)
3. Intent-to-treat effect (not all eligible received DACA)
4. Cannot separate intensive vs. extensive margin effects

---

## Session End
Analysis completed: 2026-01-26
