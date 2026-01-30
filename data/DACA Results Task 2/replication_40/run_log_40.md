# Run Log - DACA Replication Study (Replication 40)

## Overview
This log documents the replication analysis examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

## Session Start
Date: 2026-01-26

---

## Step 1: Understanding the Research Task

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

### Study Design
- **Treatment Group**: Eligible individuals aged 26-30 at time of policy implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at time of policy implementation (otherwise eligible except for age)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment period**: 2006-2011
- **Post-treatment period**: 2013-2016 (excluding 2012 due to implementation timing)

### DACA Eligibility Criteria
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

---

## Step 2: Data Exploration

### Data Files
- **data.csv**: Main ACS data file (~33.85 million observations)
- **acs_data_dict.txt**: IPUMS variable codebook
- **state_demo_policy.csv**: Optional state-level supplementary data

### Key Variables for Analysis
From the data dictionary:

**Identification Variables:**
- YEAR: Census year (2006-2016)
- PERWT: Person weight for survey design

**DACA Eligibility Variables:**
- HISPAN/HISPAND: Hispanic origin (HISPAN=1 for Mexican)
- BPL/BPLD: Birthplace (BPL=200 for Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- AGE: Age at survey

**Outcome Variable:**
- UHRSWORK: Usual hours worked per week (>=35 = full-time)
- EMPSTAT: Employment status (1 = Employed)

**Covariates:**
- SEX: Gender
- EDUC/EDUCD: Education level
- MARST: Marital status
- STATEFIP: State FIPS code
- METRO: Metropolitan status

---

## Step 3: Sample Selection Strategy

### Step 3.1: Target Population
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Born in Mexico (BPL = 200)
- Not a citizen (CITIZEN = 3)
- Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
- Continuous US residence since June 15, 2007 (YRIMMIG <= 2007)

### Step 3.2: Age Group Definition
DACA was implemented June 15, 2012. The age threshold was "not yet 31st birthday as of June 15, 2012."

**Treatment Group (Ages 26-30 on June 15, 2012):**
- Birth years: July 1981 - June 1986
- These individuals were potentially eligible for DACA

**Control Group (Ages 31-35 on June 15, 2012):**
- Birth years: July 1976 - June 1981
- These individuals were NOT eligible due to age but otherwise similar

### Step 3.3: Treatment and Control in Survey Years
For each survey year, I calculated the age an individual would have been on June 15, 2012, based on their birth year and birth quarter, then assigned treatment/control status.

---

## Step 4: Analytical Approach

### Primary Specification: Difference-in-Differences
Model: Y_it = α + β₁(Treat_i) + β₂(Post_t) + β₃(Treat_i × Post_t) + X_it'γ + ε_it

Where:
- Y_it = 1 if full-time employed (UHRSWORK >= 35 & EMPSTAT = 1), 0 otherwise
- Treat_i = 1 if treatment group (ages 26-30 on June 15, 2012)
- Post_t = 1 if year >= 2013
- β₃ = Difference-in-differences estimate (main coefficient of interest)
- X_it = Covariates (sex, education, marital status, state, metro area)

### Robustness Checks
1. Linear probability model with and without covariates
2. Event study specification to examine pre-trends
3. Subgroup analysis by sex

---

## Step 5: Python Analysis Code Execution

### Sample Selection Results
Starting from 33,851,424 total ACS observations (2006-2016):
1. After Hispanic-Mexican filter (HISPAN=1): 2,945,521
2. After Mexico birthplace filter (BPL=200): 991,261
3. After non-citizen filter (CITIZEN=3): 701,347
4. After immigration year filter (<=2007): 654,693
5. After arrival before age 16 filter: 195,023
6. After treatment/control group filter (ages 26-35): 47,418
7. After excluding 2012: **43,238 final observations**

### Sample Breakdown
| Group | Period | Unweighted N | Weighted N |
|-------|--------|--------------|------------|
| Control (31-35) | Pre (2006-2011) | 11,683 | 1,631,151 |
| Control (31-35) | Post (2013-2016) | 6,085 | 845,134 |
| Treatment (26-30) | Pre (2006-2011) | 16,694 | 2,280,009 |
| Treatment (26-30) | Post (2013-2016) | 8,776 | 1,244,124 |

---

## Step 6: Results

### Simple 2x2 Difference-in-Differences
|  | Pre (2006-2011) | Post (2013-2016) | Difference |
|--|-----------------|------------------|------------|
| Control (31-35) | 0.6135 | 0.6037 | -0.0099 |
| Treatment (26-30) | 0.5655 | 0.6198 | +0.0543 |
| **Diff-in-Diff** | | | **0.0642** |

### Regression Results Summary

| Model | DiD Estimate | SE | 95% CI | p-value |
|-------|--------------|-----|--------|---------|
| Basic DiD | 0.0642 | 0.0121 | [0.040, 0.088] | <0.001 |
| + Demographics | 0.0462 | 0.0113 | [0.024, 0.068] | <0.001 |
| + Full Covariates | 0.0436 | 0.0112 | [0.022, 0.066] | <0.001 |
| + Year FE | 0.0426 | 0.0112 | [0.021, 0.065] | <0.001 |
| **+ State & Year FE (Preferred)** | **0.0417** | **0.0112** | **[0.020, 0.064]** | **0.0002** |

### Event Study (Pre-trends Check)
All pre-treatment coefficients are close to zero and not statistically significant, supporting the parallel trends assumption:
- 2006: -0.0071 (SE: 0.0234)
- 2007: -0.0319 (SE: 0.0231)
- 2008: 0.0030 (SE: 0.0235)
- 2009: -0.0105 (SE: 0.0243)
- 2010: -0.0101 (SE: 0.0238)
- 2011: Reference year

### Subgroup Analysis
- Male: DiD = 0.0362 (SE = 0.0139)
- Female: DiD = 0.0426 (SE = 0.0180)

---

## Step 7: Key Decisions and Justifications

### Decision 1: Using Non-Citizens as Proxy for Undocumented
**Rationale**: ACS does not directly identify undocumented status. Non-citizenship (CITIZEN=3) is the closest proxy available. This may include some legal permanent residents, potentially biasing estimates toward zero.

### Decision 2: Excluding 2012
**Rationale**: DACA was implemented on June 15, 2012. Since ACS does not record month of survey, 2012 observations cannot be cleanly assigned to pre or post periods.

### Decision 3: Age Calculation Using Birth Quarter
**Rationale**: Used birth quarter to calculate exact age on June 15, 2012. Q1-Q2 births assumed to have had birthday by June 15; Q3-Q4 assumed not yet.

### Decision 4: Immigration Year Cutoff at 2007
**Rationale**: DACA required continuous residence since June 15, 2007. Using YRIMMIG <= 2007 ensures this criterion is approximately met.

### Decision 5: Arrival Before Age 16 Filter
**Rationale**: DACA required arrival before 16th birthday. Calculated as YRIMMIG - BIRTHYR < 16.

### Decision 6: Preferred Specification Includes State and Year FE
**Rationale**: State fixed effects control for time-invariant state characteristics; year fixed effects control for common shocks. Combined with demographic covariates, this provides the most rigorous estimate.

---

## Step 8: Deliverables Created

1. **analysis.py** - Main Python analysis script
2. **create_figures.py** - Script to generate figures
3. **results_summary.csv** - Summary of regression results
4. **event_study_results.csv** - Event study coefficients
5. **summary_statistics.csv** - Summary statistics by group/period
6. **figure1_event_study.png/pdf** - Event study plot
7. **figure2_trends.png/pdf** - Parallel trends visualization
8. **figure3_model_comparison.png/pdf** - Model comparison forest plot
9. **figure4_subgroups.png/pdf** - Subgroup analysis plot
10. **replication_report_40.tex** - LaTeX source for report
11. **replication_report_40.pdf** - Final PDF report (16 pages)
12. **run_log_40.md** - This run log

---

## Final Summary

### Preferred Estimate
- **Effect**: 0.0417 (4.17 percentage points)
- **Standard Error**: 0.0112
- **95% Confidence Interval**: [0.0197, 0.0636]
- **Sample Size**: 43,238

### Interpretation
DACA eligibility is associated with a 4.17 percentage point increase in the probability of full-time employment among Hispanic-Mexican Mexican-born non-citizens who arrived in the US before age 16. This effect is statistically significant at conventional levels (p = 0.0002) and represents approximately a 7.3% increase relative to the pre-treatment mean full-time employment rate of 56.6%.

### Validity
Event study analysis shows no evidence of differential pre-trends, supporting the causal interpretation of the difference-in-differences estimate.

---

## Session End
Date: 2026-01-26
