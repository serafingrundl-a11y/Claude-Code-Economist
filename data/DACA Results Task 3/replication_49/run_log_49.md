# DACA Replication Study - Run Log (Replication #49)

## Overview
This document logs all commands executed and key decisions made during the replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

---

## 1. Data Exploration

### Initial Data Load
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")
# Output: Total observations: 17,382
```

### Key Variables Identified
- **ELIGIBLE**: 1 = Treatment group (ages 26-30 at June 2012), 0 = Control group (ages 31-35)
- **AFTER**: 1 = Post-DACA period (2013-2016), 0 = Pre-DACA period (2008-2011)
- **FT**: 1 = Full-time employment (35+ hours/week), 0 = Not full-time
- **YEAR**: Survey year (2008-2011, 2013-2016; 2012 excluded)

### Sample Composition
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-DACA period (AFTER=0): 9,527 observations
- Post-DACA period (AFTER=1): 7,855 observations

---

## 2. Analytical Decisions

### Decision 1: Use of Provided Variables
**Decision**: Used the pre-constructed ELIGIBLE, AFTER, and FT variables as provided in the dataset.
**Rationale**: Instructions explicitly stated to use the ELIGIBLE variable and not create own eligibility variable. The FT variable was already constructed per the research design.

### Decision 2: Inclusion of All Observations
**Decision**: Retained all observations in the analytic sample, including those not in the labor force.
**Rationale**: Instructions stated "do not further limit the sample by dropping individuals on the basis of their characteristics" and "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis."

### Decision 3: Model Specification
**Decision**: Implemented a progressive set of models from simple DiD to full specification with controls and fixed effects.
**Rationale**: This approach allows assessment of robustness and transparency about how controls affect estimates.

Models estimated:
1. Basic DiD (no controls)
2. Basic DiD with robust standard errors (HC1)
3. DiD with demographic controls (sex, age, marital status)
4. DiD with demographic + education controls
5. DiD with controls + state and year fixed effects (robust SE)
6. **PREFERRED**: DiD with controls + fixed effects + state-clustered SE

### Decision 4: Standard Error Clustering
**Decision**: Clustered standard errors at the state level for the preferred specification.
**Rationale**: DiD designs with policy variation often have within-cluster correlation. State-level clustering accounts for common shocks and serial correlation within states.

### Decision 5: Covariate Selection
**Decision**: Included SEX, AGE, MARST (marital status), and EDUC_RECODE as control variables.
**Rationale**: These are standard demographic predictors of employment that may differ between treatment and control groups, improving precision and reducing omitted variable bias.

### Decision 6: Fixed Effects
**Decision**: Included state (STATEFIP) and year (YEAR) fixed effects in the preferred specification.
**Rationale**: State fixed effects control for time-invariant state characteristics. Year fixed effects (which absorb the AFTER indicator) control for common time trends affecting all groups.

---

## 3. Analysis Commands

### Simple DiD Calculation
```python
treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()   # 0.6263
treated_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()  # 0.6658
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()   # 0.6697
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()  # 0.6449

simple_did = (treated_post - treated_pre) - (control_post - control_pre)
# Result: 0.0643 (6.43 percentage points)
```

### Regression Analysis
```python
import statsmodels.formula.api as smf

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
# DiD coefficient: 0.0643, SE: 0.0153, p < 0.001

# Model 6 (Preferred): Full specification with clustered SE
model6 = smf.ols(formula5, data=df_model5).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model5['STATEFIP']}
)
# DiD coefficient: 0.0523, Clustered SE: 0.0150, p = 0.0005
```

### Event Study
```python
# Year-specific treatment effects (2011 = reference)
# Pre-treatment coefficients:
#   2008: -0.0523 (SE: 0.0269)
#   2009: -0.0380 (SE: 0.0277)
#   2010: -0.0592 (SE: 0.0275) [significant]
# Post-treatment coefficients:
#   2013: 0.0202 (SE: 0.0282)
#   2014: -0.0161 (SE: 0.0285)
#   2015: 0.0218 (SE: 0.0290)
#   2016: 0.0349 (SE: 0.0291)
```

---

## 4. Key Results Summary

### Preferred Estimate (Model 6)
- **Effect size**: 5.23 percentage points
- **Standard error**: 0.0150 (state-clustered)
- **95% Confidence interval**: [2.29, 8.18] percentage points
- **p-value**: 0.0005
- **Sample size**: 17,382 observations

### Interpretation
DACA eligibility is associated with a statistically significant 5.23 percentage point increase in the probability of full-time employment. Given the baseline full-time employment rate of 62.6% for the treatment group in the pre-period, this represents approximately an 8.4% relative increase.

### Robustness
- Results stable across specifications (range: 5.2-6.4 pp)
- Weighted analysis: 7.5 pp (SE: 0.015)
- Subgroup by sex: Males 5.1 pp, Females 3.8 pp
- Event study: Some pre-trends concerns (2010 coefficient significant)

---

## 5. Output Files Generated

### Data Analysis
- `analysis.py` - Main analysis script
- `key_results.csv` - Summary of key results
- `descriptive_statistics.csv` - Descriptive statistics by group
- `model1_summary.txt` - Full output of basic DiD model
- `model6_summary.txt` - Full output of preferred model
- `event_study_results.txt` - Event study regression output

### Figures
- `figure1_ft_trends.png` - Full-time employment trends by eligibility status
- `figure2_event_study.png` - Event study coefficient plot
- `figure3_ft_distribution.png` - FT distribution by group and period
- `figure4_model_comparison.png` - Coefficient comparison across models

### Report
- `replication_report_49.tex` - LaTeX source file (26 pages)
- `replication_report_49.pdf` - Compiled PDF report

---

## 6. Software Environment

- **Python version**: 3.x
- **Key packages**: pandas, numpy, statsmodels, matplotlib, scipy
- **LaTeX distribution**: MiKTeX (pdfTeX 3.141592653-2.6-1.40.28)

---

## 7. Caveats and Limitations

1. **Parallel trends**: Event study shows some pre-treatment coefficients that are negative, with 2010 being statistically significant. This raises potential concerns about the parallel trends assumption.

2. **Intent-to-treat**: The analysis identifies the effect of DACA eligibility, not actual DACA receipt. Some eligible individuals may not have applied for or received DACA.

3. **Repeated cross-section**: The ACS is not a panel, so we cannot track individuals over time. DiD relies on comparing group means across periods.

4. **Outcome measure**: Full-time employment is a relatively coarse measure. Effects on wages, job quality, or specific sectors are not captured.

---

## 8. Execution Timeline

1. Read replication instructions and data dictionary
2. Load and explore the data
3. Implement difference-in-differences analysis with multiple specifications
4. Generate figures and tables
5. Write LaTeX replication report
6. Compile to PDF
7. Document run log

---

**Analysis completed successfully.**
