# Run Log - DACA Replication Study (ID: 69)

## Date: January 26, 2026

---

## 1. Project Overview

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (defined as usually working 35 hours per week or more)?

**Design:** Difference-in-Differences comparing:
- **Treatment Group:** Ages 26-30 as of June 15, 2012 (DACA eligible)
- **Control Group:** Ages 31-35 as of June 15, 2012 (too old for DACA)

---

## 2. Data Source

- **Source:** American Community Survey (ACS) via IPUMS USA
- **Years:** 2006-2016 (1-year samples), excluding 2012
- **Data File:** data.csv (6.26 GB, 33,851,424 observations)
- **Data Dictionary:** acs_data_dict.txt

---

## 3. Key Decisions and Rationale

### 3.1 Sample Selection

| Step | Criterion | N Remaining | Rationale |
|------|-----------|-------------|-----------|
| 1 | HISPAN == 1 | 2,945,521 | Mexican Hispanic ethnicity |
| 2 | BPL == 200 | 991,261 | Born in Mexico |
| 3 | CITIZEN == 3 | 701,347 | Non-citizen (proxy for undocumented) |
| 4 | Age 26-35 at DACA | 181,229 | Treatment and control groups |
| 5 | YRIMMIG - BIRTHYR < 16 | 47,418 | Arrived before age 16 (DACA requirement) |
| 6 | YRIMMIG <= 2007 | 47,418 | Continuous US presence since June 2007 |
| 7 | YEAR != 2012 | 43,238 | Exclude ambiguous year |

### 3.2 Treatment Definition

- **Treatment:** Age 26-30 as of June 15, 2012
- **Control:** Age 31-35 as of June 15, 2012
- **Age Calculation:** Used BIRTHYR and BIRTHQTR; adjusted for birth quarter (Q3/Q4 births had not yet had 2012 birthday by June 15)

### 3.3 Outcome Variable

- **Full-time employment:** UHRSWORK >= 35
- Binary indicator (1 = full-time, 0 = not full-time)

### 3.4 Time Periods

- **Pre-DACA:** 2006-2011
- **Post-DACA:** 2013-2016
- **Excluded:** 2012 (DACA implemented mid-year)

### 3.5 Model Specification Choice

Selected **Model 3 (Year FE + Covariates)** as preferred specification because:
1. Year fixed effects control for common time trends
2. Covariates (sex, education, marital status) improve precision
3. State fixed effects (Model 4/5) may absorb variation relevant to policy effects
4. Weighted estimates (Model 6) similar, suggesting representativeness

---

## 4. Commands Executed

### 4.1 Data Loading and Initial Exploration

```python
# Load data
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424
# Years: 2006-2016
```

### 4.2 Sample Selection

```python
# Hispanic-Mexican
df_sample = df[df['HISPAN'] == 1].copy()

# Born in Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()

# Non-citizen
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()

# Age at DACA implementation
df_sample['age_at_daca'] = 2012 - df_sample['BIRTHYR']
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] -= 1

# Treatment and control groups
df_sample['treatment'] = ((df_sample['age_at_daca'] >= 26) &
                          (df_sample['age_at_daca'] <= 30)).astype(int)
df_analysis = df_sample[(df_sample['treatment'] == 1) |
                        ((df_sample['age_at_daca'] >= 31) &
                         (df_sample['age_at_daca'] <= 35))].copy()

# DACA eligibility criteria
df_analysis['age_at_immigration'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
df_analysis = df_analysis[df_analysis['age_at_immigration'] < 16].copy()
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()

# Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
```

### 4.3 Variable Construction

```python
# Outcome
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Post indicator
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# DiD interaction
df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

# Covariates
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'],
                                  bins=[-1, 2, 6, 10, 11],
                                  labels=['less_hs', 'hs', 'some_college', 'college_plus'])
```

### 4.4 Regression Analysis

```python
import statsmodels.formula.api as smf

# Model 1: Basic DiD
model1 = smf.ols('fulltime ~ treatment + post + treat_post',
                 data=df_analysis).fit(cov_type='HC1')

# Model 2: Year FE
model2 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post',
                 data=df_analysis).fit(cov_type='HC1')

# Model 3: Year FE + Covariates (PREFERRED)
model3 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post + female + C(educ_cat) + married',
                 data=df_analysis).fit(cov_type='HC1')

# Model 4: Year + State FE
model4 = smf.ols('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post',
                 data=df_analysis).fit(cov_type='HC1')

# Model 5: Full Model
model5 = smf.ols('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + female + C(educ_cat) + married',
                 data=df_analysis).fit(cov_type='HC1')

# Model 6: Weighted
model_weighted = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + female + C(educ_cat) + married',
                         data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
```

### 4.5 Robustness Checks

```python
# Gender-specific estimates
model_male = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post',
                     data=df_analysis[df_analysis['female']==0]).fit(cov_type='HC1')
model_female = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post',
                       data=df_analysis[df_analysis['female']==1]).fit(cov_type='HC1')

# Placebo test
df_placebo = df_analysis[df_analysis['post'] == 0].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['fake_treat_post'] = df_placebo['treatment'] * df_placebo['fake_post']
model_placebo = smf.ols('fulltime ~ treatment + C(YEAR) + fake_treat_post',
                        data=df_placebo).fit(cov_type='HC1')
```

### 4.6 Event Study

```python
# Year-specific treatment effects
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'year_{year}'] = ((df_analysis['YEAR'] == year) &
                                    (df_analysis['treatment'] == 1)).astype(int)

model_event = smf.ols('fulltime ~ treatment + C(YEAR) + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016',
                      data=df_analysis).fit(cov_type='HC1')
```

---

## 5. Results Summary

### 5.1 Sample Sizes by Group and Period

| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Treatment (26-30) | 16,694 | 8,776 | 25,470 |
| Control (31-35) | 11,683 | 6,085 | 17,768 |
| **Total** | **28,377** | **14,861** | **43,238** |

### 5.2 Mean Full-time Employment Rates

| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment | 0.615 | 0.634 | +0.019 |
| Control | 0.646 | 0.614 | -0.032 |
| **Simple DiD** | | | **0.052** |

### 5.3 Regression Results

| Model | Coefficient | Std. Error | 95% CI | P-value | N |
|-------|-------------|------------|--------|---------|---|
| Basic DiD | 0.0516 | 0.0100 | [0.032, 0.071] | <0.001 | 43,238 |
| Year FE | 0.0515 | 0.0099 | [0.032, 0.071] | <0.001 | 43,238 |
| **Year FE + Covariates** | **0.0451** | **0.0092** | **[0.027, 0.063]** | **<0.001** | **43,238** |
| Year + State FE | 0.0503 | 0.0099 | [0.031, 0.070] | <0.001 | 43,238 |
| Full Model | 0.0442 | 0.0092 | [0.026, 0.062] | <0.001 | 43,238 |
| Weighted | 0.0457 | 0.0107 | [0.025, 0.067] | <0.001 | 43,238 |

### 5.4 Preferred Estimate (Model 3)

- **Effect Size:** 0.0451 (4.51 percentage points)
- **Standard Error:** 0.0092
- **95% Confidence Interval:** [0.027, 0.063]
- **P-value:** <0.001
- **Sample Size:** 43,238

### 5.5 Robustness Checks

| Check | Coefficient | Std. Error | P-value |
|-------|-------------|------------|---------|
| Male only | 0.046 | 0.011 | <0.001 |
| Female only | 0.046 | 0.015 | 0.003 |
| Placebo (2009) | 0.007 | 0.012 | 0.564 |

### 5.6 Event Study Coefficients (relative to 2011)

| Year | Coefficient | Std. Error | P-value |
|------|-------------|------------|---------|
| 2006 | -0.022 | 0.020 | 0.264 |
| 2007 | -0.038 | 0.020 | 0.059 |
| 2008 | -0.008 | 0.021 | 0.701 |
| 2009 | -0.020 | 0.021 | 0.337 |
| 2010 | -0.029 | 0.021 | 0.173 |
| 2011 | 0.000 | --- | (ref) |
| 2013 | 0.024 | 0.022 | 0.274 |
| 2014 | 0.030 | 0.022 | 0.176 |
| 2015 | 0.027 | 0.022 | 0.233 |
| 2016 | 0.049 | 0.022 | 0.028 |

---

## 6. Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `analysis_data.csv` | Analysis dataset |
| `regression_results.csv` | Summary of regression results |
| `key_results.txt` | Key results for reporting |
| `figure1_parallel_trends.png/pdf` | Parallel trends plot |
| `figure2_event_study.png/pdf` | Event study plot |
| `figure3_did_visualization.png/pdf` | DiD visualization |
| `figure4_age_distribution.png/pdf` | Age distribution |
| `figure5_employment_education.png/pdf` | Employment by education |
| `figure6_model_comparison.png/pdf` | Model comparison |
| `replication_report_69.tex` | LaTeX report |
| `replication_report_69.pdf` | Final PDF report (25 pages) |
| `run_log_69.md` | This log file |

---

## 7. Interpretation

DACA eligibility is associated with an increase of approximately 4.5 percentage points in the probability of full-time employment among Hispanic-Mexican immigrants born in Mexico. This effect is:

1. **Statistically significant:** p < 0.001
2. **Robust:** Consistent across multiple model specifications (range: 4.4-5.2 pp)
3. **Supports parallel trends:** Pre-treatment coefficients are small and insignificant
4. **Consistent by gender:** Similar effects for men and women
5. **Passes placebo test:** No significant effect with fake 2009 treatment date

The findings suggest that DACA's work authorization provisions had meaningful positive effects on full-time employment for eligible immigrants.

---

## 8. Software Environment

- **Python:** 3.x
- **Key packages:**
  - pandas (data manipulation)
  - statsmodels (regression analysis)
  - matplotlib (visualization)
- **LaTeX:** pdfLaTeX (MiKTeX distribution)

---

## 9. Replication Instructions

1. Ensure Python 3.x is installed with pandas, numpy, statsmodels, matplotlib
2. Ensure LaTeX (pdflatex) is installed
3. Place `data.csv` in the `data/` folder
4. Run: `python analysis.py`
5. Run: `python create_figures.py`
6. Compile: `pdflatex replication_report_69.tex` (run 3 times for references)

---

*End of Run Log*
