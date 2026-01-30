# Run Log - DACA Replication Study (Run 24)

## Overview

This document logs all commands and key decisions made during the independent replication of the DACA eligibility effect on full-time employment.

**Date**: January 27, 2026
**Data Source**: American Community Survey (ACS) via IPUMS USA
**Analysis Tool**: Python with pandas, numpy, statsmodels, matplotlib

---

## 1. Data Exploration

### 1.1 Initial Data Load

```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```

**Findings**:
- Total observations: 17,382
- Total columns: 105
- Years included: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)

### 1.2 Key Variable Distributions

**ELIGIBLE variable**:
- 1 (ages 26-30 in June 2012): 11,382 observations
- 0 (ages 31-35 in June 2012): 6,000 observations

**AFTER variable**:
- 0 (pre-DACA, 2008-2011): 9,527 observations
- 1 (post-DACA, 2013-2016): 7,855 observations

**FT (Full-Time Employment)**:
- 1 (employed full-time): 11,283 observations
- 0 (not full-time): 6,099 observations

### 1.3 Cross-tabulation: ELIGIBLE x AFTER

| ELIGIBLE | AFTER=0 | AFTER=1 | Total |
|----------|---------|---------|-------|
| 0        | 3,294   | 2,706   | 6,000 |
| 1        | 6,233   | 5,149   | 11,382|
| Total    | 9,527   | 7,855   | 17,382|

---

## 2. Key Analytical Decisions

### 2.1 Outcome Variable

**Decision**: Use FT variable as provided in dataset
- FT = 1 if usually works 35+ hours per week
- FT = 0 otherwise (including not in labor force)
- Individuals not in labor force are retained in analysis as FT=0

**Rationale**: Instructions explicitly stated to keep those not in labor force in the analysis.

### 2.2 Treatment and Comparison Groups

**Decision**: Use ELIGIBLE variable as provided
- Treatment (ELIGIBLE=1): Ages 26-30 at June 2012
- Comparison (ELIGIBLE=0): Ages 31-35 at June 2012

**Rationale**: Instructions stated to use the provided ELIGIBLE variable and not create our own.

### 2.3 Weighting

**Decision**: Use person weights (PERWT) for all main analyses

**Rationale**: ACS is a complex survey; weighting produces population-representative estimates.

### 2.4 Standard Errors

**Decision**: Use heteroskedasticity-robust (HC1) standard errors as primary; report state-clustered SEs as robustness check

**Rationale**: Robust SEs account for heteroskedasticity; clustering accounts for within-state correlation.

### 2.5 Control Variables

**Decision**: Include age, sex, education (EDUC_RECODE), marital status (MARST), year FE, and state FE in preferred specification

**Rationale**: These are standard demographic controls that may affect employment outcomes and differ between groups.

### 2.6 Preferred Specification

**Decision**: Select Model 4 (demographics + year FE + state FE, weighted) as preferred specification

**Rationale**: Balances controlling for observable confounders while maintaining parsimony; weighting ensures representative estimates.

---

## 3. Analysis Commands

### 3.1 Basic Difference-in-Differences

```python
# Unweighted means
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()

# Weighted means
def wmean(group, col='FT', wt='PERWT'):
    return np.average(group[col], weights=group[wt])
weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(lambda x: wmean(x))
```

**Results**:
- Unweighted DiD estimate: 0.0643
- Weighted DiD estimate: 0.0748

### 3.2 Regression-Based DiD

```python
import statsmodels.api as sm

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic weighted OLS
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')
```

**Results** (Basic DiD):
- ELIGIBLE_AFTER coefficient: 0.0748
- Robust SE: 0.0181
- 95% CI: [0.0393, 0.1102]
- p-value: <0.001

### 3.3 With Year Fixed Effects

```python
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True).astype(float)
X_year = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER']], year_dummies], axis=1)
model_year = sm.WLS(df['FT'], X_year, weights=df['PERWT']).fit(cov_type='HC1')
```

**Results**:
- ELIGIBLE_AFTER: 0.0721 (SE: 0.0181)

### 3.4 With Year + State Fixed Effects

```python
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True).astype(float)
X_both = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER']], year_dummies, state_dummies], axis=1)
model_both = sm.WLS(df['FT'], X_both, weights=df['PERWT']).fit(cov_type='HC1')
```

**Results**:
- ELIGIBLE_AFTER: 0.0710 (SE: 0.0180)

### 3.5 Full Model with Demographics (PREFERRED)

```python
df['FEMALE'] = (df['SEX'] == 2).astype(float)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True).astype(float)
marst_dummies = pd.get_dummies(df['MARST'], prefix='MARST', drop_first=True).astype(float)

X_full = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE']],
                    educ_dummies, marst_dummies, year_dummies, state_dummies], axis=1)
model_full = sm.WLS(df['FT'], X_full, weights=df['PERWT']).fit(cov_type='HC1')
```

**PREFERRED ESTIMATE**:
- ELIGIBLE_AFTER: **0.0594**
- Robust SE: **0.0166**
- 95% CI: **[0.0268, 0.0919]**
- t-statistic: 3.5721
- p-value: **0.000354**
- N: **17,382**
- R-squared: 0.1395

---

## 4. Robustness Checks

### 4.1 Heterogeneous Effects by Sex

```python
# Males only
df_male = df[df['SEX'] == 1]
# [run regression on subset]
```

**Results**:
- Males: DiD = 0.0610 (SE: 0.0196), N = 9,075
- Females: DiD = 0.0413 (SE: 0.0272), N = 8,307

### 4.2 Event Study (Year-Specific Effects)

```python
# Create year-eligible interactions
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(float)
```

**Results** (reference year: 2011):
| Year | Coefficient | SE | p-value |
|------|-------------|------|---------|
| 2008 | -0.0663 | 0.0320 | 0.038 |
| 2009 | -0.0466 | 0.0329 | 0.157 |
| 2010 | -0.0750 | 0.0328 | 0.022 |
| 2011 | [ref] | -- | -- |
| 2013 | 0.0180 | 0.0339 | 0.595 |
| 2014 | -0.0155 | 0.0349 | 0.657 |
| 2015 | -0.0089 | 0.0348 | 0.799 |
| 2016 | 0.0572 | 0.0351 | 0.103 |

### 4.3 Placebo Test (Pre-Period Split at 2010)

```python
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['PLACEBO_AFTER'] = (df_pre['YEAR'] >= 2010).astype(float)
df_pre['ELIGIBLE_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_AFTER']
```

**Results**:
- Placebo DiD (2008-2009 vs 2010-2011): 0.0170 (SE: 0.0223, p = 0.446)

### 4.4 Narrower Bandwidth (27-29 vs 32-34)

**Results**:
- DiD: 0.0522 (SE: 0.0232), N = 8,362

### 4.5 Heterogeneity by Education

| Education | Coefficient | SE | N |
|-----------|-------------|------|-------|
| High School | 0.0454 | 0.0193 | 12,444 |
| Some College | 0.0560 | 0.0421 | 2,877 |
| Two-Year Degree | 0.1645 | 0.0754 | 991 |
| BA+ | 0.1607 | 0.0675 | 1,058 |

### 4.6 Clustered Standard Errors

**Results** (preferred specification):
- Robust (HC1) SE: 0.0166
- State-clustered SE: 0.0208
- Year-clustered SE: 0.0214

---

## 5. Figures Generated

1. **figure1_parallel_trends.png**: FT rates by eligibility status, 2008-2016
2. **figure2_did_bars.png**: Pre/post comparison bar chart
3. **figure3_event_study.png**: Event study coefficients plot
4. **figure4_heterogeneity_sex.png**: FT trends by sex

---

## 6. Output Files

| Filename | Description |
|----------|-------------|
| replication_report_24.tex | LaTeX source file (~20 pages) |
| replication_report_24.pdf | Compiled PDF report |
| run_log_24.md | This run log |
| figure1_parallel_trends.png | Parallel trends figure |
| figure2_did_bars.png | DiD bar chart |
| figure3_event_study.png | Event study figure |
| figure4_heterogeneity_sex.png | Heterogeneity by sex |

---

## 7. Summary of Main Findings

**Preferred Estimate**: DACA eligibility increased full-time employment by **5.94 percentage points** (95% CI: 2.68 to 9.19 pp, p < 0.001).

**Key findings**:
1. Effect is robust across specifications (5.2-7.5 pp range)
2. Effect is larger for males than females
3. Effect is larger for more educated individuals
4. Some evidence of pre-trends warrants caution
5. Placebo test does not detect spurious pre-period effects

---

## 8. Software Environment

- Python 3.14
- pandas
- numpy
- statsmodels
- matplotlib
- LaTeX (MiKTeX)

---

*End of Run Log*
