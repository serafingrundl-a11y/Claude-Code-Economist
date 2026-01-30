# DACA Replication Study - Run Log

## Study Overview
**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Design**: Difference-in-differences comparing:
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation (would have been eligible if not for age)

**Data Source**: American Community Survey (ACS) 2006-2016

---

## Session Log

### Step 1: Environment Setup
- Working directory: C:\Users\seraf\DACA Results Task 2\replication_50
- Data files identified:
  - data/data.csv (main ACS data file, ~6GB)
  - data/acs_data_dict.txt (data dictionary)
  - data/state_demo_policy.csv (optional state-level data)

### Step 2: Data Dictionary Review
Key variables identified:
- YEAR: Census year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1-4)
- HISPAN/HISPAND: Hispanic origin (1=Mexican for HISPAN)
- BPL/BPLD: Birthplace (200=Mexico for BPL)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight

### Step 3: Sample Selection Criteria
DACA eligibility requirements (per instructions):
1. Hispanic-Mexican ethnicity (HISPAN=1)
2. Born in Mexico (BPL=200)
3. Not a citizen (CITIZEN=3, assume undocumented)
4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
5. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
6. Present in US on June 15, 2012

Age groups for analysis:
- Treatment: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
- Control: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)

Time periods:
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016 (excluding 2012 due to mid-year implementation)

### Step 4: Analysis Approach
Difference-in-differences regression:
- Outcome: Full-time employment (UHRSWORK >= 35)
- Treatment indicator: 1 if ages 26-30 as of June 2012
- Post indicator: 1 if year >= 2013
- DiD coefficient: Treatment × Post interaction
- Use person weights (PERWT) for population estimates
- Cluster standard errors at state level (STATEFIP)

---

## Analysis Execution

### Step 5: Data Processing
- Python script: analysis.py
- Data loaded in chunks (500,000 rows) to manage memory
- Total ACS observations: 33,851,424
- After Hispanic-Mexican & Mexico-born filter: 991,261
- After non-citizen filter: 701,347
- After arrived before age 16: 205,327
- After in US since 2007: 195,023
- After age group filter (26-35 at June 2012): 47,418
- After excluding 2012: 43,238
- Final weighted population: 6,000,418

### Step 6: Age Calculation
Age as of June 15, 2012 calculated using:
- BIRTHYR (year of birth)
- BIRTHQTR (quarter of birth)
- Adjustment: For Q3/Q4 births, subtract 1 from (2012 - BIRTHYR) since birthday hasn't occurred by June 15

### Step 7: Model Specifications

**Model 1: Basic DiD**
- fulltime ~ treat + post + treat_post
- DiD coefficient: 0.0590 (SE: 0.0069)

**Model 2: DiD with Demographics**
- fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_somecoll
- DiD coefficient: 0.0472 (SE: 0.0094)

**Model 3: DiD with Year Fixed Effects**
- Added year dummies (reference: 2006)
- DiD coefficient: 0.0456 (SE: 0.0097)

**Model 4: DiD with State and Year Fixed Effects (PREFERRED)**
- Added state dummies (50 states)
- DiD coefficient: 0.0449 (SE: 0.0101)
- 95% CI: [0.0251, 0.0646]
- p-value < 0.001

### Step 8: Robustness Checks

| Check | Coefficient | SE | N |
|-------|------------|-----|------|
| Narrow age bands (27-29 vs 32-34) | 0.0378 | 0.0194 | 25,606 |
| Males only | 0.0351 | 0.0098 | 24,243 |
| Females only | 0.0510 | 0.0160 | 18,995 |
| Any employment outcome | 0.0431 | 0.0058 | 43,238 |
| Donut (excl. ages 30-31) | 0.0526 | 0.0141 | 35,245 |

### Step 9: Event Study Analysis
Reference year: 2011

| Year | Coefficient | SE | P-value |
|------|------------|-----|---------|
| 2006 | 0.0062 | 0.0276 | 0.822 |
| 2007 | -0.0319 | 0.0160 | 0.046 |
| 2008 | 0.0087 | 0.0206 | 0.674 |
| 2009 | -0.0076 | 0.0223 | 0.733 |
| 2010 | -0.0136 | 0.0240 | 0.572 |
| 2013 | 0.0344 | 0.0226 | 0.128 |
| 2014 | 0.0370 | 0.0167 | 0.027 |
| 2015 | 0.0220 | 0.0182 | 0.226 |
| 2016 | 0.0656 | 0.0207 | 0.002 |

Pre-treatment coefficients are generally small and insignificant, supporting parallel trends assumption.

---

## Key Decisions and Justifications

1. **Age calculation with birth quarter adjustment**: Accounts for whether individual had birthday by June 15, 2012. Those born Q3/Q4 hadn't turned older yet.

2. **Exclusion of 2012**: DACA implemented June 15, 2012. ACS doesn't record interview month, so cannot distinguish pre/post within 2012.

3. **Non-citizen = undocumented assumption**: ACS cannot distinguish documented vs undocumented non-citizens. Per instructions, assume non-citizens without papers are undocumented. This introduces measurement error (attenuation bias).

4. **Clustering at state level**: State-level clustering accounts for within-state correlation and state-level policy variation (e.g., driver's license access for DACA recipients).

5. **Person weights (PERWT)**: Used throughout for population-representative estimates.

6. **Linear probability model**: Chosen for ease of interpretation and compatibility with fixed effects. With outcomes in 60-70% range, LPM approximates nonlinear models well.

7. **State and year fixed effects in preferred specification**: Controls for time-invariant state characteristics and common shocks across years.

---

## Output Files Generated

### Analysis Files
- analysis.py (main analysis script)
- create_figures.py (figure generation script)
- results_summary.csv
- event_study_results.csv
- descriptive_stats.csv
- pre_trends.csv

### Figures
- figure1_event_study.png/pdf
- figure2_trends.png/pdf
- figure3_specifications.png/pdf
- figure4_robustness.png/pdf
- figure5_pretrends.png/pdf

### Report
- replication_report_50.tex (LaTeX source)
- replication_report_50.pdf (23 pages)

---

## Final Results Summary

**Preferred Estimate (Model 4 with State and Year FE):**
- DiD Coefficient: **0.0449** (4.49 percentage points)
- Standard Error: 0.0101
- 95% Confidence Interval: [0.0251, 0.0646]
- P-value: < 0.001

**Interpretation**: DACA eligibility is associated with a 4.49 percentage point increase in the probability of full-time employment among eligible individuals. This represents a ~7% relative increase from the pre-DACA treatment group mean of 63%.

---

## Session Completed
All deliverables generated:
- [x] replication_report_50.tex
- [x] replication_report_50.pdf
- [x] run_log_50.md
