# Run Log - DACA Replication Study 06

## Date: 2026-01-25

## Overview
Independent replication study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Step 1: Understanding the Research Question

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (defined as usually working 35+ hours per week)?

**DACA Eligibility Criteria:**
- Arrived unlawfully in the US before their 16th birthday
- Had not yet had their 31st birthday as of June 15, 2012
- Lived continuously in the US since June 15, 2007
- Present in the US on June 15, 2012 without lawful status

**Key Dates:**
- DACA enacted: June 15, 2012
- Applications started: August 15, 2012
- Study period: 2013-2016 (post-treatment)

---

## Step 2: Data Exploration

**Data Files:**
1. `data.csv` - ACS data from 2006-2016 (33,851,425 observations)
2. `acs_data_dict.txt` - IPUMS variable definitions
3. `state_demo_policy.csv` - State-level policy data (optional)

**Key Variables Identified:**
- `YEAR` - Survey year (2006-2016)
- `HISPAN` / `HISPAND` - Hispanic origin (1=Mexican for HISPAN; 100-107 for Mexican in HISPAND)
- `BPL` / `BPLD` - Birthplace (200=Mexico)
- `CITIZEN` - Citizenship status (3=Not a citizen)
- `YRIMMIG` - Year of immigration
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (for more precise age calculation)
- `AGE` - Age at survey
- `UHRSWORK` - Usual hours worked per week
- `EMPSTAT` - Employment status
- `PERWT` - Person weight for representative estimates
- `EDUC` / `EDUCD` - Educational attainment
- `SEX` - Gender
- `MARST` - Marital status
- `STATEFIP` - State FIPS code

---

## Step 3: Analysis Design

### Identification Strategy: Difference-in-Differences

**Treatment Group:** Hispanic-Mexican, Mexico-born non-citizens who meet DACA eligibility criteria
- Arrived before age 16
- Born after June 15, 1981 (under 31 as of June 15, 2012)
- Immigration year <= 2007 (present since at least June 2007)

**Control Group:** Hispanic-Mexican, Mexico-born non-citizens who do NOT meet DACA eligibility criteria
- Similar demographics but ineligible due to:
  - Arrived at age 16 or older, OR
  - Too old (born on or before June 15, 1981)

**Time Periods:**
- Pre-treatment: 2006-2011 (excluding 2012 due to implementation timing)
- Post-treatment: 2013-2016

**Outcome Variable:**
- Full-time employment: UHRSWORK >= 35 AND EMPSTAT == 1

### Key Methodological Decisions:

1. **2012 Exclusion:** Excluded 2012 due to inability to distinguish pre/post DACA within that year (DACA enacted June 15, 2012)
2. **Age restrictions:** Focus on working-age population (16-64)
3. **Sample restriction:** Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200) AND non-citizen (CITIZEN=3)
4. **Weighting:** Use PERWT for population-representative estimates
5. **Standard errors:** Cluster by state to account for within-state correlation
6. **Fixed effects:** Include both state and year fixed effects in preferred specification

---

## Step 4: Implementation

### Analysis Script: `analysis.py`

Created Python script to perform:
1. Data loading (chunked reading due to large file size)
2. Sample construction and filtering
3. DACA eligibility coding
4. Variable creation (outcome, controls)
5. Descriptive statistics
6. Difference-in-differences regression (4 specifications)
7. Robustness checks (6 alternative specifications)
8. Placebo test
9. Event study analysis

---

## Step 5: Execution Log

### [2026-01-25]

#### Data Loading and Filtering
- Loaded ACS data in chunks to manage memory (~34M rows)
- Filtered to target population: Hispanic-Mexican, Mexico-born, non-citizens
- Target population: 701,347 observations
- After excluding 2012: 636,722 observations
- After restricting to ages 16-64: 561,470 observations

#### Sample Composition
- DACA-eligible: 83,611 observations
- DACA-ineligible: 477,859 observations

#### DACA Eligibility Coding
- Arrived before age 16: 186,357
- Born after June 15, 1981: 208,596
- In US since 2007 or earlier: 595,366
- All criteria met (DACA-eligible): 83,611

#### Descriptive Statistics
| Group | Period | N | Full-time (%) |
|-------|--------|---|---------------|
| Ineligible | Pre (2006-11) | 298,978 | 57.5% |
| Ineligible | Post (2013-16) | 178,881 | 56.8% |
| Eligible | Pre (2006-11) | 46,814 | 39.9% |
| Eligible | Post (2013-16) | 36,797 | 48.0% |

#### Main Results

**Preferred Estimate (State + Year FE):**
- Treatment Effect: **0.0237** (2.37 percentage points)
- Standard Error: 0.0041 (clustered by state)
- 95% CI: [0.016, 0.032]
- p-value: < 0.001
- Sample Size: 561,470
- R-squared: 0.216

#### Robustness Results Summary

| Specification | Coefficient | SE | p-value |
|--------------|-------------|-----|---------|
| Basic DiD | 0.088 | 0.004 | <0.001 |
| With Controls | 0.031 | 0.004 | <0.001 |
| Year FE | 0.024 | 0.004 | <0.001 |
| **State+Year FE** | **0.024** | **0.004** | **<0.001** |
| Employment outcome | 0.040 | 0.007 | <0.001 |
| Ages 18-35 | 0.007 | 0.005 | 0.145 |
| Males only | 0.020 | 0.005 | <0.001 |
| Females only | 0.019 | 0.007 | 0.005 |
| Unweighted | 0.027 | 0.006 | <0.001 |
| Placebo (2009) | 0.019 | 0.004 | <0.001 |

#### Event Study Results

| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | -0.020 | 0.010 | [-0.039, -0.001] |
| 2007 | -0.011 | 0.006 | [-0.023, 0.000] |
| 2008 | -0.001 | 0.010 | [-0.021, 0.019] |
| 2009 | 0.011 | 0.011 | [-0.011, 0.033] |
| 2010 | 0.012 | 0.011 | [-0.010, 0.034] |
| 2011 | 0 (ref) | -- | -- |
| 2013 | 0.012 | 0.009 | [-0.007, 0.031] |
| 2014 | 0.017 | 0.014 | [-0.010, 0.044] |
| 2015 | 0.032 | 0.011 | [0.011, 0.053] |
| 2016 | 0.032 | 0.010 | [0.012, 0.053] |

**Note:** Pre-trends visible in event study, with upward trend from 2006-2011. Placebo test also significant, suggesting caution in causal interpretation.

---

## Step 6: Report Generation

### LaTeX Report
- Created comprehensive ~20-page report in LaTeX
- Includes: Abstract, Introduction, Background, Data, Empirical Strategy, Results, Discussion, Conclusion, Appendix
- Tables: Sample construction, descriptive statistics, main results, robustness checks, event study
- Figure: Event study plot using TikZ/PGFPlots

### PDF Compilation
- Compiled using pdflatex (2 passes for cross-references)
- Output: 19 pages, 337KB

---

## Step 7: Key Decisions and Justifications

1. **Sample Definition:** Restricted to Hispanic-Mexican, Mexico-born, non-citizens to match the research question's target population.

2. **2012 Exclusion:** DACA implemented June 15, 2012; ACS does not identify interview month, so 2012 contains mixed pre/post observations.

3. **Control Group:** Used DACA-ineligible individuals from the same population (arrived too late, too old) rather than a different ethnicity, to minimize unobserved heterogeneity.

4. **Outcome Definition:** Full-time = 35+ hours AND employed (EMPSTAT=1), following standard labor economics conventions.

5. **Standard Errors:** Clustered by state (51 clusters) to address within-state correlation and policy variation.

6. **Weighting:** Used PERWT for population representativeness; results robust to unweighted estimation.

7. **Fixed Effects:** Included state and year FE to control for time-invariant state characteristics and national trends.

---

## Step 8: Caveats and Limitations

1. **Pre-trends:** Event study reveals upward pre-trend in relative employment among eligible group, violating parallel trends assumption.

2. **Placebo Test Failure:** Significant effect using 2009 as fake treatment suggests differential trends existed before DACA.

3. **Measurement Error:** DACA eligibility imperfectly measured (no monthly immigration timing, cannot verify continuous residence).

4. **Selection:** Treatment is eligibility, not actual DACA receipt; take-up may be selective.

5. **Interpretation:** Results should be interpreted as suggestive of positive DACA effect, but causal claims require caution given pre-trend concerns.

---

## Files Generated

1. `analysis.py` - Main analysis script (Python)
2. `descriptive_stats.csv` - Descriptive statistics by group
3. `regression_results.csv` - All regression coefficients and standard errors
4. `event_study_results.csv` - Event study coefficients
5. `key_results.txt` - Key statistics for report
6. `replication_report_06.tex` - LaTeX source
7. `replication_report_06.pdf` - Final report (19 pages)
8. `run_log_06.md` - This run log

---

## Final Summary

**Preferred Estimate:** DACA eligibility is associated with a 2.37 percentage point increase in full-time employment (SE = 0.0041, p < 0.001). However, event study analysis reveals pre-existing differential trends that warrant cautious interpretation of the causal effect.

---

*End of Run Log*
