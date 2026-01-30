# DACA Replication Study - Run Log

## Session Information
- Date: January 27, 2026
- Replication ID: 14

## Data Files Used
- `data/prepared_data_numeric_version.csv` - Main analysis dataset (17,382 observations)
- `data/prepared_data_labelled_version.csv` - Dataset with labeled categorical variables
- `data/acs_data_dict.txt` - Data dictionary

## Key Analytic Decisions

### 1. Sample Definition
- Used the provided ELIGIBLE variable to identify treatment (ages 26-30 at June 15, 2012) and control (ages 31-35) groups
- Did not create own eligibility variable as instructed
- Kept all observations without additional sample restrictions
- Total N = 17,382 (11,382 treatment, 6,000 control)

### 2. Outcome Variable
- Used FT variable as provided (1 = usually works 35+ hours/week, 0 = otherwise)
- Included those not in the labor force as zeros, as instructed
- This captures both extensive margin (labor force participation) and intensive margin (hours) effects

### 3. Model Specification
- Estimated OLS linear probability model rather than logit/probit
- Rationale: Ease of interpretation, straightforward handling of interaction terms, common in DiD literature

### 4. Control Variables
- Demographics: AGE, SEX (coded as FEMALE dummy), MARST (marital status dummies)
- Education: EDUC category dummies
- Fixed effects: State (STATEFIP), Year (YEAR)
- Did not include state policy variables in preferred specification (absorbed by state FE)

### 5. Standard Errors
- Primary: Heteroskedasticity-robust (HC1) standard errors
- Robustness check: State-clustered standard errors

### 6. Sample Weights
- Primary analysis: Unweighted OLS
- Robustness check: Weighted regression using PERWT

## Commands Executed

### Data Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
df.shape  # (17382, 105)
df['YEAR'].value_counts().sort_index()
df['ELIGIBLE'].value_counts()
df['FT'].value_counts()
```

### Simple Difference-in-Differences Calculation
```python
# Pre-post for treatment (ELIGIBLE=1)
ft_treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()   # 0.6263
ft_treated_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()  # 0.6658

# Pre-post for control (ELIGIBLE=0)
ft_control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()   # 0.6697
ft_control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()  # 0.6449

# DiD = (0.6658 - 0.6263) - (0.6449 - 0.6697) = 0.0643
```

### Regression Models
```python
import statsmodels.formula.api as smf

# Create variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Model 1: Basic DiD
m1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
# DiD = 0.0643, SE = 0.0153, p < 0.001

# Model 2: + Demographics
m2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + C(MARST)',
             data=df).fit(cov_type='HC1')
# DiD = 0.0548, SE = 0.0142, p < 0.001

# Model 3: + Education
m3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + C(MARST) + C(EDUC)',
             data=df).fit(cov_type='HC1')
# DiD = 0.0530, SE = 0.0141, p < 0.001

# Model 4: + Year FE
m4 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + C(MARST) + C(EDUC) + C(YEAR)',
             data=df).fit(cov_type='HC1')
# DiD = 0.0515, SE = 0.0141, p < 0.001

# Model 5: + State FE (PREFERRED)
m5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + C(MARST) + C(EDUC) + C(STATEFIP) + C(YEAR)',
             data=df).fit(cov_type='HC1')
# DiD = 0.0515, SE = 0.0141, p < 0.001
```

### Robustness Checks
```python
# State-clustered SE
df_clean = df.dropna(subset=['FT', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE', 'MARST', 'EDUC', 'YEAR', 'STATEFIP'])
m_cluster = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + FEMALE + C(MARST) + C(EDUC) + C(YEAR) + C(STATEFIP)',
                    data=df_clean).fit(cov_type='cluster', cov_kwds={'groups': df_clean['STATEFIP']})
# DiD = 0.0515, Clustered SE = 0.0148, p < 0.001

# Placebo test (fake treatment year = 2010)
df_pre = df_clean[df_clean['YEAR'].isin([2008, 2009, 2010, 2011])].copy()
df_pre['AFTER_PLACEBO'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_AFTER_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['AFTER_PLACEBO']
m_placebo = smf.ols('FT ~ ELIGIBLE + AFTER_PLACEBO + ELIGIBLE_AFTER_PLACEBO + AGE + FEMALE + C(MARST) + C(EDUC) + C(STATEFIP) + C(YEAR)',
                    data=df_pre).fit(cov_type='HC1')
# Placebo DiD = 0.014, SE = 0.019, p = 0.48 (not significant)
```

### Event Study
```python
# Create year interaction dummies
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

event_formula = 'FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_Y2008 + ELIGIBLE_Y2009 + ELIGIBLE_Y2010 + ELIGIBLE_Y2013 + ELIGIBLE_Y2014 + ELIGIBLE_Y2015 + ELIGIBLE_Y2016 + AGE + FEMALE + C(MARST) + C(EDUC) + C(STATEFIP)'
m_event = smf.ols(event_formula, data=df).fit(cov_type='HC1')

# Results (relative to 2011):
# 2008: -0.052 (SE = 0.027)
# 2009: -0.039 (SE = 0.028)
# 2010: -0.057 (SE = 0.028)
# 2013:  0.019 (SE = 0.028)
# 2014: -0.017 (SE = 0.029)
# 2015:  0.023 (SE = 0.029)
# 2016:  0.034 (SE = 0.029)
```

### Subgroup Analysis
```python
# By sex
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    m = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + AGE + C(MARST) + C(EDUC) + C(STATEFIP) + C(YEAR)',
                data=df_sub).fit(cov_type='HC1')

# Male: DiD = 0.0504, SE = 0.0169, p = 0.003
# Female: DiD = 0.0379, SE = 0.0228, p = 0.096
```

### Visualizations
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parallel trends plot
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
# Saved as: parallel_trends.png

# Event study plot
# Saved as: event_study.png
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_14.tex
pdflatex -interaction=nonstopmode replication_report_14.tex  # Second pass for references
```

## Results Summary

### Preferred Estimate (Model 5: Full Controls + State and Year FE)
| Metric | Value |
|--------|-------|
| Effect Size | 0.0515 (5.15 percentage points) |
| Standard Error | 0.0141 |
| 95% Confidence Interval | [0.024, 0.079] |
| p-value | < 0.001 |
| Sample Size | 17,379 |
| R-squared | 0.138 |

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 5.15 percentage points among Mexican-born Hispanic individuals aged 26-30 (compared to those aged 31-35). This represents about an 8% increase relative to the treatment group's pre-DACA baseline of 62.6%.

## Output Files Created
1. `replication_report_14.tex` - LaTeX source file (21 pages)
2. `replication_report_14.pdf` - Compiled PDF report
3. `parallel_trends.png` - Visualization of FT employment trends
4. `event_study.png` - Event study coefficient plot
5. `run_log_14.md` - This log file

## Session Complete
All deliverables created successfully.
