# DACA Replication Study Run Log - Replication 48

## Project Overview
**Research Question:** What was the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US?

**Design:** Difference-in-Differences (DiD)
- Treatment group: Ages 26-30 on June 15, 2012 (birth years 1982-1986)
- Control group: Ages 31-35 on June 15, 2012 (birth years 1977-1981)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 due to mid-year policy implementation)

---

## Key Decisions Log

### Decision 1: Sample Selection Criteria
**Date:** Analysis start
**Decision:** Include individuals who meet ALL of the following:
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a US citizen and no first papers (CITIZEN == 3)
4. Arrived in US before age 16 (YRIMMIG - BIRTHYR < 16)
5. Arrived in US by June 2007 (YRIMMIG <= 2007)
6. In relevant age groups based on birth year

**Rationale:** These criteria approximate DACA eligibility. We cannot distinguish documented from undocumented immigrants, so we include all non-citizens without papers as the instructions specify.

### Decision 2: Age Group Assignment
**Date:** Analysis start
**Decision:** Use birth year to assign treatment/control groups:
- Treatment (ages 26-30 on 6/15/2012): Born 1982-1986
- Control (ages 31-35 on 6/15/2012): Born 1977-1981

**Rationale:** DACA eligibility cutoff was based on age at implementation (must be under 31). This creates a natural comparison group of those just above the age cutoff.

### Decision 3: Outcome Variable
**Date:** Analysis start
**Decision:** Full-time employment = UHRSWORK >= 35
- Binary indicator where 1 = usually works 35+ hours per week
- Includes all individuals regardless of current employment status

**Rationale:** Instructions define full-time as "usually working 35 hours per week or more."

### Decision 4: Exclusion of 2012
**Date:** Analysis start
**Decision:** Exclude 2012 from analysis entirely.

**Rationale:** DACA was implemented on June 15, 2012, mid-year. ACS does not indicate survey month, so we cannot distinguish pre- and post-treatment observations within 2012.

### Decision 5: Survey Weights
**Date:** Analysis start
**Decision:** Use PERWT (person weights) for all analyses.

**Rationale:** ACS is a complex survey design; weights ensure nationally representative estimates.

### Decision 6: Standard Errors
**Date:** Analysis start
**Decision:** Use heteroskedasticity-robust standard errors, clustered at the state level.

**Rationale:** State-level clustering accounts for correlation within states and is standard in DiD analyses with geographic variation.

---

## Commands and Code Execution Log

### Step 1: Data Exploration
```
Examined data dictionary (acs_data_dict.txt)
Previewed data.csv structure
Years available: 2006-2016
Columns: YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, REGION, STATEFIP,
         PUMA, METRO, STRATA, GQ, FOODSTMP, PERNUM, PERWT, FAMSIZE, NCHILD,
         RELATE, RELATED, SEX, AGE, BIRTHQTR, MARST, BIRTHYR, RACE, RACED,
         HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG, YRSUSA1, YRSUSA2,
         HCOVANY, HINSEMP, HINSCAID, HINSCARE, EDUC, EDUCD, EMPSTAT, EMPSTATD,
         LABFORCE, CLASSWKR, CLASSWKRD, OCC, IND, WKSWORK1, WKSWORK2, UHRSWORK,
         INCTOT, FTOTINC, INCWAGE, POVERTY
```

### Step 2: Sample Construction
```python
# Filter: HISPAN == 1 (Mexican Hispanic)
# Filter: BPL == 200 (Born in Mexico)
# Filter: CITIZEN == 3 (Not a citizen)
# Filter: Calculate age at immigration, require < 16
# Filter: YRIMMIG <= 2007 (continuous US residence since June 2007)
# Filter: Birth years 1977-1986 (relevant age range)
# Exclude YEAR == 2012

Final sample size: 44,161 observations
- Treatment group (born 1982-1986): 26,294
- Control group (born 1977-1981): 17,867
- Pre-period (2006-2011): 28,968
- Post-period (2013-2016): 15,193
```

### Step 3: Variable Construction
- `post`: 1 if YEAR >= 2013, 0 if YEAR <= 2011
- `treat`: 1 if BIRTHYR >= 1982 (ages 26-30 on 6/15/2012)
- `treat_post`: interaction term (treat * post)
- `fulltime`: 1 if UHRSWORK >= 35
- `employed`: 1 if EMPSTAT == 1
- `female`: 1 if SEX == 2
- `married`: 1 if MARST in [1, 2]
- `educ_hs`: 1 if EDUC >= 6
- `educ_college`: 1 if EDUC >= 7

### Step 4: Main DiD Regression
Model: fulltime = beta0 + beta1*treat + beta2*post + beta3*(treat*post) + epsilon

beta3 is the DiD estimate of DACA's effect on full-time employment.

### Step 5: Robustness Checks
1. With demographic covariates (sex, education, marital status)
2. Year fixed effects instead of post dummy
3. State fixed effects
4. Placebo test using 2006-2011 data only
5. Alternative outcome: Any employment
6. Heterogeneity by sex
7. Event study with year-by-treatment interactions

---

## Analysis Output Summary

### Main Results
**Preferred Estimate (Basic Weighted DiD with State-Clustered SE):**
- Effect size: 0.0624 (6.24 percentage points)
- Standard error: 0.0094
- 95% CI: [0.044, 0.081]
- t-statistic: 6.66
- p-value: < 0.0001
- Sample size: 44,161

### Robustness Results
| Model | Coefficient | Std. Error | p-value |
|-------|-------------|------------|---------|
| (1) Basic OLS | 0.0541 | 0.007 | <0.001 |
| (2) Weighted | 0.0624 | 0.009 | <0.001 |
| (3) Demographics | 0.0505 | 0.011 | <0.001 |
| (4) Year FE | 0.0614 | 0.010 | <0.001 |
| (5) Year+State FE | 0.0600 | 0.012 | <0.001 |
| (6) Full | 0.0486 | 0.011 | <0.001 |

### Placebo Test (Pre-Period Only)
- Coefficient: 0.0123
- p-value: 0.177
- Interpretation: No significant pre-trend difference

### Event Study Results
| Year | Coefficient | Std. Error |
|------|-------------|------------|
| 2006 | -0.0015 | 0.020 |
| 2007 | -0.0113 | 0.018 |
| 2008 | 0.0235 | 0.019 |
| 2009 | 0.0219 | 0.021 |
| 2010 | 0.0256 | 0.020 |
| 2011 | 0 (ref) | - |
| 2013 | 0.0629 | 0.024 |
| 2014 | 0.0724 | 0.019 |
| 2015 | 0.0488 | 0.020 |
| 2016 | 0.0994 | 0.018 |

---

## Interpretation
DACA eligibility is associated with a statistically significant increase in the probability of full-time employment by approximately 6.2 percentage points. The effect is robust across multiple specifications including controls for demographics, year fixed effects, and state fixed effects. The placebo test shows no evidence of differential pre-trends between the treatment and control groups, supporting the parallel trends assumption. The event study shows that the treatment effects emerge after 2012 and persist through 2016.

---

## File Manifest
- `data/data.csv`: Main ACS data file (2006-2016)
- `data/acs_data_dict.txt`: Variable codebook
- `data/state_demo_policy.csv`: Optional state-level data (not used)
- `analysis_48.py`: Main analysis script
- `results_48.csv`: Numerical results for report
- `descriptive_stats_48.csv`: Descriptive statistics
- `regression_results_48.csv`: Regression coefficients
- `event_study_48.csv`: Event study coefficients
- `replication_report_48.tex`: LaTeX report
- `replication_report_48.pdf`: Final PDF report
- `run_log_48.md`: This log file
