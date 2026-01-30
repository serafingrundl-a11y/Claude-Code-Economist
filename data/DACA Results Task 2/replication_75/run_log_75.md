# Replication Run Log - Task 75

## Project Overview
Independent replication of DACA eligibility impact on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (35+ hours/week) among eligible Hispanic-Mexican Mexican-born individuals?

## Design
- **Treatment Group**: Ages 26-30 at policy implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at policy implementation (otherwise eligible but for age)
- **Method**: Difference-in-Differences
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 due to mid-year implementation)

---

## Session Log

### Step 1: Data Dictionary Review
- Reviewed acs_data_dict.txt for variable definitions
- Key variables identified:
  - YEAR: Survey year (2006-2016)
  - BIRTHYR: Birth year for age calculation
  - BIRTHQTR: Quarter of birth
  - HISPAN/HISPAND: Hispanic origin (1 = Mexican, 100-107 = Mexican detailed)
  - BPL/BPLD: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration (for continuous residence check)
  - UHRSWORK: Usual hours worked per week (35+ = full-time)
  - EMPSTAT: Employment status
  - PERWT: Person weight for population estimates
  - AGE: Age at survey time

### Step 2: DACA Eligibility Criteria
Per the instructions, eligibility requires:
1. Born in Mexico (BPL = 200)
2. Hispanic-Mexican ethnicity (HISPAN = 1)
3. Not a citizen and no immigration papers (CITIZEN = 3)
4. Arrived before 16th birthday
5. Arrived by June 15, 2007 (lived continuously since then)
6. Present in US on June 15, 2012

Treatment group: Would be ages 26-30 on June 15, 2012
Control group: Would be ages 31-35 on June 15, 2012

### Step 3: Data Exploration
- Total ACS observations (2006-2016): 33,851,424
- Data file size: ~6.3 GB
- Columns available: YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, REGION, STATEFIP, PUMA, METRO, STRATA, GQ, FOODSTMP, PERNUM, PERWT, FAMSIZE, NCHILD, RELATE, RELATED, SEX, AGE, BIRTHQTR, MARST, BIRTHYR, RACE, RACED, HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG, YRSUSA1, YRSUSA2, HCOVANY, HINSEMP, HINSCAID, HINSCARE, EDUC, EDUCD, EMPSTAT, EMPSTATD, LABFORCE, CLASSWKR, CLASSWKRD, OCC, IND, WKSWORK1, WKSWORK2, UHRSWORK, INCTOT, FTOTINC, INCWAGE, POVERTY

### Step 4: Sample Construction
Applied filters sequentially:
1. Hispanic-Mexican, Mexico-born, non-citizen: 701,347 observations
2. Arrived before age 16 and by 2007: 195,023 observations
3. Ages 26-35 at DACA implementation: 47,418 observations
4. Excluding 2012: 43,238 observations (final analysis sample)

Final sample breakdown:
- Treatment group (ages 26-30): 25,470 observations
- Control group (ages 31-35): 17,768 observations
- Pre-period (2006-2011): 28,377 observations
- Post-period (2013-2016): 14,861 observations

### Step 5: Statistical Analysis
Ran Python analysis script (analysis.py) using:
- pandas for data manipulation
- statsmodels for regression analysis
- scipy for statistics

Models estimated:
1. Basic DiD (unweighted OLS)
2. Weighted DiD (WLS with PERWT)
3. DiD with covariates (sex, education, marital status)
4. DiD with year fixed effects
5. DiD with state and year fixed effects (preferred specification)

### Step 6: Main Results

**Preferred Estimate (Model 5):**
- DiD Coefficient: 0.0435 (4.35 percentage points)
- Clustered Standard Error: 0.0099
- 95% CI: [0.024, 0.063]
- p-value: < 0.001
- Sample Size: 43,238
- R-squared: 0.161

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 4.35 percentage points, representing a ~7% increase relative to baseline.

### Step 7: Robustness Checks
1. **Gender heterogeneity:**
   - Males: DiD = 0.032 (SE = 0.011)
   - Females: DiD = 0.048 (SE = 0.015)

2. **Placebo test (fake treatment in 2009):**
   - Placebo DiD = 0.006 (SE = 0.011, p = 0.61)
   - No significant effect, supporting parallel trends assumption

3. **Event study:**
   - Pre-treatment coefficients (2006-2010): All close to zero and insignificant
   - Post-treatment coefficients grow over time
   - 2013: 0.035 (p = 0.07)
   - 2016: 0.065 (p = 0.001)

### Step 8: Report Generation
- Created LaTeX replication report (replication_report_75.tex)
- Compiled to PDF (replication_report_75.pdf)
- Report is 19 pages covering:
  - Introduction and background
  - Data and methodology
  - Results and robustness checks
  - Discussion and conclusion
  - Appendices with variable definitions and analytical decisions

---

## Key Analytical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Outcome variable | Full-time (35+ hrs/week) | Per research question |
| Treatment group | Ages 26-30 at DACA | DACA eligible |
| Control group | Ages 31-35 at DACA | Just over age cutoff |
| Pre-period | 2006-2011 | Pre-DACA data |
| Post-period | 2013-2016 | Post-DACA data |
| Exclude 2012 | Yes | Mid-year implementation |
| Undocumented proxy | CITIZEN = 3 | Best available in ACS |
| Age at immigration | < 16 | DACA requirement |
| Immigration year | <= 2007 | Continuous residence |
| Weighting | PERWT | Population estimates |
| Standard errors | State-clustered | Within-state correlation |
| Fixed effects | Year + State | Control for trends |

---

## Output Files
- `analysis.py` - Main analysis script
- `results_summary.csv` - Key results
- `yearly_effects.csv` - Year-by-year effects
- `event_study.csv` - Event study coefficients
- `model_coefficients.csv` - Full model output
- `replication_report_75.tex` - LaTeX report
- `replication_report_75.pdf` - Final PDF report

---

## Summary Statistics

### Full-Time Employment Rates (Weighted)

| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| Treatment (26-30) | 63.1% | 66.0% |
| Control (31-35) | 67.3% | 64.3% |

### DiD Calculation
- Treatment change: 66.0% - 63.1% = +2.9 pp
- Control change: 64.3% - 67.3% = -3.0 pp
- DiD: 2.9 - (-3.0) = 5.9 pp (simple calculation)
- Regression-adjusted DiD: 4.35 pp

---

## Conclusion
DACA eligibility had a statistically significant positive effect on full-time employment among Hispanic-Mexican Mexican-born individuals. The preferred estimate of 4.35 percentage points (95% CI: 2.40-6.29 pp) is robust to various specifications and supported by event study analysis showing no pre-trends and growing post-treatment effects.

---

*Log completed: January 26, 2026*
