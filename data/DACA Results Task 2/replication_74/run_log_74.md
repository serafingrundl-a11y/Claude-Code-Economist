# Replication Run Log - Task 74

## Overview

This log documents all commands, key decisions, and analytical steps taken during the independent replication of the DACA employment effects study.

## Research Question

Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design

- **Treatment Group**: Ages 26-30 as of June 15, 2012
- **Control Group**: Ages 31-35 as of June 15, 2012
- **Method**: Difference-in-Differences
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

---

## Session Log

### Step 1: Data Exploration

**Files in data folder**:
- `data.csv` - Main ACS data file (~6.3GB, 33,851,424 observations)
- `acs_data_dict.txt` - Data dictionary
- `state_demo_policy.csv` - Optional state-level data (not used)

**Data source**: American Community Survey (ACS) via IPUMS USA, 2006-2016

**Key Variables Identified**:

| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Census year (2006-2016) | Time period |
| HISPAN | Hispanic origin | Sample restriction (=1 for Mexican) |
| BPL | Birthplace | Sample restriction (=200 for Mexico) |
| CITIZEN | Citizenship status | DACA eligibility (=3 for non-citizen) |
| YRIMMIG | Year of immigration | DACA eligibility |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Age refinement |
| UHRSWORK | Usual hours worked per week | Outcome variable |
| PERWT | Person weight | Survey weights |
| STATEFIP | State FIPS code | State fixed effects |
| SEX | Sex | Control variable |
| MARST | Marital status | Control variable |
| EDUCD | Education detail | Control variable |

---

### Step 2: Sample Construction Decisions

**Sample Restrictions Applied**:

1. **Hispanic-Mexican ethnicity**: HISPAN == 1
   - Rationale: Focus on Mexican-origin population per research question

2. **Born in Mexico**: BPL == 200
   - Rationale: DACA-eligible population is predominantly Mexican-born

3. **Non-citizen**: CITIZEN == 3
   - Rationale: Proxy for undocumented status; cannot distinguish documented vs undocumented in ACS

4. **Year exclusion**: Exclude 2012
   - Rationale: Cannot distinguish pre/post DACA within 2012 (no interview month available)

5. **Arrived before age 16**: (YRIMMIG - BIRTHYR) < 16
   - Rationale: DACA eligibility criterion

6. **Continuous residence**: YRIMMIG <= 2007
   - Rationale: Proxy for continuous residence since June 15, 2007

7. **Age restriction**: Ages 26-35 as of June 15, 2012
   - Rationale: Treatment (26-30) and control (31-35) groups

**Age Calculation**:
```
age_june_2012 = 2012 - BIRTHYR
if BIRTHQTR in [3, 4]:  # July-Dec births
    age_june_2012 -= 1  # Haven't had birthday by June 15
```

---

### Step 3: Variable Definitions

**Outcome Variable**:
```
fulltime = 1 if UHRSWORK >= 35
         = 0 otherwise
```

**Treatment Variable**:
```
treated = 1 if age_june_2012 in [26, 27, 28, 29, 30]
        = 0 if age_june_2012 in [31, 32, 33, 34, 35]
```

**Post-Treatment Indicator**:
```
post = 1 if YEAR >= 2013
     = 0 if YEAR <= 2011
```

**Control Variables**:
- male: SEX == 1
- married: MARST <= 2
- educ_hs: EDUCD >= 62

---

### Step 4: Sample Statistics

**Sample Flow**:
| Restriction | N |
|-------------|---|
| Full ACS 2006-2016 | 33,851,424 |
| HISPAN == 1 | 2,945,521 |
| BPL == 200 | 991,261 |
| CITIZEN == 3 | 701,347 |
| Exclude 2012 | 636,722 |
| Arrived < age 16 | 186,357 |
| YRIMMIG <= 2007 | 177,294 |
| Ages 26-35 | **43,238** |

**Final Sample Breakdown**:
- Treatment group (26-30): 25,470
- Control group (31-35): 17,768
- Pre-period (2006-2011): 28,377
- Post-period (2013-2016): 14,861

---

### Step 5: Analysis Commands

**Python Script**: `analysis_script.py`

```python
# Key commands executed:

# 1. Load data with chunked reading (memory efficient)
cols_needed = ['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR',
               'BIRTHQTR', 'AGE', 'SEX', 'MARST', 'EDUCD', 'UHRSWORK',
               'EMPSTAT', 'PERWT', 'STATEFIP']

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=1000000):
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    chunk = chunk[chunk['YEAR'] != 2012]
    chunks.append(chunk)

# 2. Basic DiD regression (unweighted)
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()

# 3. Weighted DiD
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT']).fit()

# 4. DiD with controls
model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + educ_hs',
                  data=df, weights=df['PERWT']).fit()

# 5. DiD with year FE
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + male + married + educ_hs',
                  data=df, weights=df['PERWT']).fit()

# 6. Preferred: DiD with year + state FE, robust SE
model_preferred = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + male + married + educ_hs',
                           data=df, weights=df['PERWT']).fit(cov_type='HC1')

# 7. Event study
model_event = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treat_2006 + treat_2007 +
                       treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 +
                       treat_2015 + treat_2016 + male + married + educ_hs',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

---

### Step 6: Results Summary

#### Raw Difference-in-Differences

| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Treatment (26-30) | 0.615 | 0.634 | +0.019 |
| Control (31-35) | 0.646 | 0.614 | -0.032 |
| **DiD** | | | **0.052** |

#### Regression Results

| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| 1. Basic (unweighted) | 0.0516 | 0.0100 | [0.032, 0.071] | <0.001 |
| 2. Weighted | 0.0590 | 0.0098 | [0.040, 0.078] | <0.001 |
| 3. + Demographics | 0.0480 | 0.0090 | [0.030, 0.066] | <0.001 |
| 4. + Year FE | 0.0463 | 0.0090 | [0.029, 0.064] | <0.001 |
| 5. + State FE | 0.0456 | 0.0090 | [0.028, 0.063] | <0.001 |
| **6. + Robust SE** | **0.0456** | **0.0107** | **[0.025, 0.067]** | **<0.001** |

#### Preferred Estimate

- **Effect Size**: 0.0456 (4.56 percentage points)
- **Robust SE**: 0.0107
- **95% CI**: [0.0247, 0.0665]
- **T-statistic**: 4.28
- **P-value**: 0.000019
- **Sample Size**: 43,238

#### Event Study Results (Reference: 2011)

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | 0.007 | 0.023 | 0.764 |
| 2007 | -0.029 | 0.022 | 0.195 |
| 2008 | 0.008 | 0.023 | 0.710 |
| 2009 | -0.007 | 0.023 | 0.774 |
| 2010 | -0.013 | 0.023 | 0.562 |
| 2013 | 0.036 | 0.024 | 0.141 |
| 2014 | 0.036 | 0.025 | 0.139 |
| 2015 | 0.022 | 0.025 | 0.365 |
| 2016 | 0.068 | 0.025 | 0.006 |

**Interpretation**: Pre-treatment coefficients are not significantly different from zero, supporting the parallel trends assumption. Post-treatment coefficients are uniformly positive.

#### Heterogeneity by Gender

| Subgroup | Coefficient | SE | N |
|----------|-------------|-----|---|
| Males | 0.0330 | 0.0124 | 24,243 |
| Females | 0.0473 | 0.0181 | 18,995 |

---

### Step 7: Key Analytical Decisions

1. **Non-citizen proxy**: Used CITIZEN == 3 as proxy for undocumented status since ACS cannot distinguish documented from undocumented non-citizens.

2. **Age bandwidth**: Used 5-year bandwidth on each side of cutoff (ages 26-30 vs 31-35) to balance similarity and sample size.

3. **Continuous residence**: Used YRIMMIG <= 2007 as proxy for continuous residence since June 15, 2007.

4. **Full-time threshold**: Used UHRSWORK >= 35 following standard labor economics definitions.

5. **2012 exclusion**: Excluded 2012 entirely since ACS doesn't record interview month.

6. **Survey weights**: Used PERWT in all preferred specifications.

7. **Robust SE**: Used HC1 heteroskedasticity-robust standard errors.

8. **Fixed effects**: Included year and state fixed effects to control for common shocks and state-level differences.

---

### Step 8: Output Files Generated

1. `analysis_script.py` - Main analysis script
2. `regression_results.csv` - All model coefficients
3. `event_study_results.csv` - Event study estimates
4. `heterogeneity_results.csv` - Gender subgroup results
5. `summary_statistics.csv` - Descriptive statistics
6. `key_results.csv` - Preferred estimate summary
7. `replication_report_74.tex` - LaTeX report source
8. `replication_report_74.pdf` - Final report (25 pages)
9. `run_log_74.md` - This file

---

### Step 9: LaTeX Compilation

```bash
pdflatex -interaction=nonstopmode replication_report_74.tex
pdflatex -interaction=nonstopmode replication_report_74.tex  # Second pass for references
```

Output: 25 pages

---

## Final Summary

**Research Question**: Effect of DACA eligibility on full-time employment

**Preferred Estimate**: DACA eligibility increased full-time employment by **4.56 percentage points** (SE = 0.0107, p < 0.001)

**Interpretation**: This represents a 7.4% increase relative to the treatment group's pre-period mean. The effect is positive for both men and women. Event study analysis supports the parallel trends assumption.

---

*Log completed: January 2026*
