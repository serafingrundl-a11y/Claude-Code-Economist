# DACA Replication Study - Run Log

## Study Information
- **Research Question**: What was the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States?
- **Method**: Difference-in-Differences (DiD)
- **Data Source**: American Community Survey (ACS) 2006-2016 via IPUMS
- **Date**: January 2026

---

## Data Loading and Preparation

### Step 1: Examine Data Structure
- Read data dictionary (`acs_data_dict.txt`) to understand variable definitions
- Key variables identified:
  - `HISPAN`: Hispanic origin (1 = Mexican)
  - `BPL`: Birthplace (200 = Mexico)
  - `CITIZEN`: Citizenship status (3 = Non-citizen)
  - `YRIMMIG`: Year of immigration
  - `BIRTHYR`: Birth year
  - `EMPSTAT`: Employment status (1 = Employed)
  - `UHRSWORK`: Usual hours worked per week

### Step 2: Load Data
- Total observations in raw data: 33,851,424
- Used chunked reading to manage memory constraints
- Applied initial filters during loading (Hispanic-Mexican & born in Mexico)
- After initial filter: 991,261 observations

---

## Sample Construction

### DACA Eligibility Criteria Applied:
1. Hispanic-Mexican ethnicity (HISPAN = 1): 991,261 obs
2. Born in Mexico (BPL = 200): Included in step 1
3. Non-citizen (CITIZEN = 3): 701,347 obs
4. Arrived before age 16: 205,327 obs
5. In US since 2007 or earlier (YRIMMIG <= 2007): 195,023 obs

### Treatment and Control Groups:
- **Treatment group**: Ages 26-30 in June 2012 (birth years 1982-1986)
  - Observations: 29,093
- **Control group**: Ages 31-35 in June 2012 (birth years 1977-1981)
  - Observations: 19,926

### Time Periods:
- **Pre-treatment**: 2006-2011 (29,326 observations)
- **Post-treatment**: 2013-2016 (15,399 observations)
- **Transition year excluded**: 2012 (excluded from main analysis)

### Final Analytic Sample: 44,725 observations

---

## Variable Definitions

### Outcome Variable:
- **Full-time employment**: Binary indicator
  - Coded as 1 if EMPSTAT = 1 (employed) AND UHRSWORK >= 35 (usual hours ≥ 35/week)
  - Overall rate in sample: 56.28%

### Treatment Indicator:
- **treatment**: 1 if age 26-30 in 2012, 0 if age 31-35 in 2012

### Post-Treatment Indicator:
- **post**: 1 if year >= 2013, 0 if year <= 2011

### Covariates:
- **female**: 1 if SEX = 2
- **married**: 1 if MARST in [1, 2]
- **educ_hs_plus**: 1 if EDUC >= 6 (high school or more)
- **age**: AGE at time of survey
- **age_sq**: age squared
- **has_children**: 1 if NCHILD > 0

---

## Key Analytical Decisions

### Decision 1: Exclusion of 2012
- **Rationale**: DACA was implemented on June 15, 2012 (mid-year)
- ACS does not record interview month, so 2012 observations cannot be classified as pre- or post-treatment
- Excluded 2012 from main analysis; included in robustness check

### Decision 2: Age Bandwidth
- **Main specification**: 26-30 vs 31-35 (5-year bandwidth on each side)
- **Robustness check**: 27-29 vs 32-34 (narrower 3-year bandwidth)

### Decision 3: Outcome Definition
- Defined full-time as 35+ usual hours per week (standard BLS definition)
- Required employment status to be "employed" (EMPSTAT = 1)

### Decision 4: Sample Restrictions
- Used YRIMMIG <= 2007 as proxy for "continuously in US since June 2007"
- Used non-citizen status (CITIZEN = 3) as proxy for undocumented status
- Cannot directly identify DACA applicants or undocumented individuals

### Decision 5: Standard Errors
- Used heteroskedasticity-robust (HC1) standard errors throughout
- Did not cluster (ACS is repeated cross-section, not panel)

---

## Model Specifications

### Model 1: Basic DiD
```
fulltime_emp ~ treatment + post + treatment*post
```

### Model 2: DiD with Covariates
```
fulltime_emp ~ treatment + post + treatment*post + female + married + educ_hs_plus + age + age_sq + has_children
```

### Model 3: DiD with Fixed Effects (Preferred)
```
fulltime_emp ~ treatment + treatment*post + female + married + educ_hs_plus + age + age_sq + has_children + year_FE + state_FE
```

### Model 4: Weighted DiD with Fixed Effects
```
Same as Model 3, weighted by PERWT (person weights)
```

---

## Results Summary

### Main Results:

| Model | Coefficient | Std. Error | p-value | 95% CI |
|-------|------------|------------|---------|--------|
| Model 1 (Basic DiD) | 0.0592 | 0.0100 | <0.001 | [0.040, 0.079] |
| Model 2 (+ Covariates) | 0.0672 | 0.0127 | <0.001 | [0.042, 0.092] |
| Model 3 (+ FE) | 0.0199 | 0.0135 | 0.140 | [-0.007, 0.046] |
| Model 4 (Weighted + FE) | 0.0321 | 0.0163 | 0.049 | [0.000, 0.064] |

### Simple DiD Calculation:
- Treatment group change: 0.537 → 0.589 (+5.3 pp)
- Control group change: 0.577 → 0.570 (-0.7 pp)
- DiD = 5.3 - (-0.7) = 5.9 percentage points

### Event Study (reference year: 2011):
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.0232 | 0.0190 | 0.221 |
| 2007 | -0.0080 | 0.0191 | 0.674 |
| 2008 | 0.0150 | 0.0194 | 0.440 |
| 2009 | -0.0045 | 0.0199 | 0.822 |
| 2010 | 0.0038 | 0.0197 | 0.845 |
| 2013 | 0.0470 | 0.0203 | 0.021 |
| 2014 | 0.0445 | 0.0204 | 0.029 |
| 2015 | 0.0488 | 0.0207 | 0.019 |
| 2016 | 0.0600 | 0.0209 | 0.004 |

### Robustness Checks:
| Specification | Coefficient | SE | p-value |
|---------------|-------------|-----|---------|
| Placebo (fake 2009) | 0.0066 | 0.0111 | 0.552 |
| Narrow bandwidth | 0.0468 | 0.0121 | <0.001 |
| Include 2012 | 0.0482 | 0.0087 | <0.001 |

### Heterogeneity:
| Subgroup | Coefficient | SE | p-value |
|----------|-------------|-----|---------|
| Males | 0.0568 | 0.0120 | <0.001 |
| Females | 0.0387 | 0.0145 | 0.008 |
| Less than HS | 0.0266 | 0.0142 | 0.061 |
| HS or more | 0.0759 | 0.0125 | <0.001 |

---

## Interpretation

### Preferred Estimate (Model 3):
- **Point estimate**: 0.0199 (2.0 percentage points)
- **95% CI**: [-0.007, 0.046]
- **p-value**: 0.140
- **Sample size**: 44,725

### Conclusion:
The baseline DiD estimates suggest DACA eligibility increased full-time employment by approximately 5.9-6.7 percentage points. However, after including year and state fixed effects, the estimate is reduced to 2.0 percentage points and is no longer statistically significant at conventional levels. The event study shows no clear pre-trends and consistently positive post-treatment effects. Effects appear larger for males and more educated individuals.

---

## Files Generated

### Analysis Files:
- `analysis.py` - Main Python analysis script
- `summary_statistics.csv` - Summary statistics by group/period
- `regression_results.csv` - Main regression coefficients
- `event_study_results.csv` - Event study coefficients
- `heterogeneity_results.csv` - Subgroup analysis
- `robustness_results.csv` - Robustness check results
- `covariate_results.csv` - Full Model 3 coefficients
- `sample_by_year.csv` - Sample sizes by year

### Report Files:
- `replication_report_03.tex` - LaTeX source
- `replication_report_03.pdf` - Final report (20 pages)

### Log Files:
- `run_log_03.md` - This run log

---

## Computational Environment

- **Language**: Python 3.x
- **Key Packages**:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis)
- **LaTeX**: pdfTeX via MiKTeX

---

## Commands Executed

```python
# Data loading (chunked to manage memory)
for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes, chunksize=500000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)

# Sample construction
df_sample = df[df['CITIZEN'] == 3]
df_sample = df_sample[df_sample['age_at_immig'] < 16]
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]

# Treatment assignment
df['treatment'] = ((df['age_in_2012'] >= 26) & (df['age_in_2012'] <= 30)).astype(int)

# Outcome definition
df['fulltime_emp'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

# DiD regressions
model1 = smf.ols('fulltime_emp ~ treatment + post + treat_post', data=df_main).fit(cov_type='HC1')
model2 = smf.ols('fulltime_emp ~ treatment + post + treat_post + female + married + educ_hs_plus + age + age_sq + has_children', data=df_main).fit(cov_type='HC1')
model3 = smf.ols('fulltime_emp ~ treatment + treat_post + female + married + educ_hs_plus + age + age_sq + has_children + C(year_str) + C(state_str)', data=df_main).fit(cov_type='HC1')
```

---

## Notes

1. The ACS is a repeated cross-section, not panel data, so we observe different individuals in each year.

2. The significant reduction in the DiD estimate when adding fixed effects (from 6.7 to 2.0 pp) suggests important confounding from time-varying factors that differentially affected treatment and control groups.

3. The placebo test passing (insignificant coefficient with fake 2009 treatment) provides support for the parallel trends assumption.

4. The event study shows no clear pre-trend, with pre-treatment coefficients small and insignificant, supporting the identification strategy.

5. Post-treatment effects are consistently positive and significant in the event study (4.5-6.0 pp), suggesting a real treatment effect despite the insignificant pooled estimate in Model 3.
