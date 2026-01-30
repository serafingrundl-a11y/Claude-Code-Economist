# Replication Run Log - Replication 100

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

## Research Design
- **Treatment group**: DACA-eligible individuals aged 26-30 as of June 15, 2012 (born 1982-1986)
- **Control group**: Individuals aged 31-35 as of June 15, 2012 (born 1977-1981), who would have been eligible if not for their age
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

## Key Decisions Log

### Decision 1: Sample Selection Criteria
**Date**: 2026-01-26

Eligibility criteria for DACA (based on instructions):
1. Hispanic-Mexican ethnicity: HISPAN = 1
2. Born in Mexico: BPL = 200
3. Not a citizen: CITIZEN = 3
4. Immigrated before age 16: YRIMMIG - BIRTHYR < 16
5. Lived in US since June 15, 2007: YRIMMIG <= 2007

**Rationale**: These criteria follow the DACA eligibility requirements. We use non-citizen status as a proxy for undocumented status since the ACS does not distinguish documented from undocumented non-citizens.

### Decision 2: Treatment and Control Group Definition
- Treatment group: Birth years 1982-1986 (ages 26-30 on June 15, 2012)
- Control group: Birth years 1977-1981 (ages 31-35 on June 15, 2012)

**Rationale**: The control group would have been eligible for DACA except for the age requirement (must not have had 31st birthday as of June 15, 2012). This provides a comparison group with similar characteristics but ineligible due to age.

### Decision 3: Outcome Variable
Full-time employment defined as UHRSWORK >= 35 hours per week

**Rationale**: This follows the standard BLS definition of full-time work as 35 or more hours per week.

### Decision 4: Handling 2012 Data
Excluding 2012 from analysis.

**Rationale**: ACS does not list month of data collection, so we cannot distinguish observations from before vs. after DACA implementation (June 15, 2012). Including 2012 would contaminate both pre and post periods.

### Decision 5: Fixed Effects
Included year and state fixed effects in preferred specification.

**Rationale**: Year fixed effects control for common time trends; state fixed effects control for time-invariant state-level factors affecting employment.

### Decision 6: Standard Errors
Used heteroskedasticity-robust (HC1) standard errors.

**Rationale**: Employment outcomes are binary, so heteroskedasticity is expected. Robust standard errors provide valid inference.

## Commands and Analysis Steps

### Step 1: Data Loading
```python
df = pd.read_csv('data/data.csv')
```
- Total observations in raw data: 33,851,424
- Years available: 2006-2016

### Step 2: Sample Selection
Applied sequential filters:
1. Hispanic-Mexican (HISPAN == 1): 2,945,521 observations
2. Born in Mexico (BPL == 200): 991,261 observations
3. Non-citizen (CITIZEN == 3): 701,347 observations
4. Arrived before age 16: 205,327 observations
5. In US since 2007 (YRIMMIG <= 2007): 195,023 observations

### Step 3: Define Treatment/Control Cohorts
Kept only birth years 1977-1986:
- After cohort selection: 49,019 observations
- Treatment (born 1982-1986): 26,591
- Control (born 1977-1981): 18,134

### Step 4: Exclude 2012
- Final sample: 44,725 observations
- Pre-period (2006-2011): 29,326
- Post-period (2013-2016): 15,399

### Step 5: Create Outcome Variable
- UHRSWORK >= 35 coded as fulltime = 1
- Overall full-time employment rate: 62.43%

### Step 6: Regression Analysis

#### Model 1: Basic DiD
```
fulltime ~ treated + post + did
```
- DiD estimate: 0.0551 (SE: 0.0098)
- p-value < 0.001

#### Model 2: DiD with Demographics
```
fulltime ~ treated + post + did + female + married + educ_hs + educ_somecoll + educ_coll
```
- DiD estimate: 0.0486 (SE: 0.0091)
- p-value < 0.001

#### Model 3: DiD with Year FE
- DiD estimate: 0.0486 (SE: 0.0091)
- p-value < 0.001

#### Model 4: DiD with State and Year FE (PREFERRED)
- DiD estimate: 0.0477 (SE: 0.0091)
- 95% CI: [0.0299, 0.0654]
- p-value < 0.001

#### Model 5: Weighted DiD
- DiD estimate: 0.0491 (SE: 0.0106)
- p-value < 0.001

### Step 7: Robustness Checks

#### Placebo Test (Fake Treatment 2009)
- Placebo DiD coefficient: 0.0140
- p-value: 0.189 (not significant)
- Interpretation: No pre-trend detected

#### Event Study
Coefficients relative to 2011:
- 2006: -0.030 (p=0.102)
- 2007: -0.025 (p=0.169)
- 2008: 0.003 (p=0.870)
- 2009: -0.004 (p=0.817)
- 2010: -0.008 (p=0.676)
- 2013: 0.032 (p=0.103)
- 2014: 0.029 (p=0.142)
- 2015: 0.037 (p=0.065)
- 2016: 0.052 (p=0.011)

Pre-trend coefficients not significantly different from zero, supporting parallel trends assumption.

## 2x2 Difference-in-Differences Table

| Period | Treatment (26-30) | Control (31-35) | Difference |
|--------|-------------------|-----------------|------------|
| Pre-DACA (2006-2011) | 0.6111 (n=17,410) | 0.6431 (n=11,916) | -0.0320 |
| Post-DACA (2013-2016) | 0.6339 (n=9,181) | 0.6108 (n=6,218) | +0.0231 |
| Change | +0.0228 | -0.0323 | |
| **DiD Estimate** | | | **0.0551** |

## Final Results Summary

**Preferred Estimate (Model 4 with State/Year FE):**
- Effect size: 4.77 percentage points
- Standard error: 0.91 percentage points
- 95% CI: [2.99, 6.54] percentage points
- Sample size: 44,725
- Treatment group: 26,591
- Control group: 18,134

**Interpretation**: DACA eligibility is associated with a 4.77 percentage point increase in the probability of full-time employment among eligible Mexican-born Hispanic individuals, relative to the slightly-too-old control group. This effect is statistically significant at conventional levels.

## Output Files Generated
1. `results_summary.csv` - Summary statistics and regression results
2. `event_study_results.csv` - Year-by-year event study coefficients
3. `model_summaries.txt` - Full regression output
4. `figure1_event_study.png` - Event study plot
5. `figure2_parallel_trends.png` - Parallel trends visualization
6. `figure3_did_bars.png` - DiD bar chart
7. `replication_report_100.tex` - LaTeX report
8. `replication_report_100.pdf` - PDF report

## Software Used
- Python 3.x
- pandas
- numpy
- statsmodels
- matplotlib
- scipy
