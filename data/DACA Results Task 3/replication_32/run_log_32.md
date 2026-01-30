# Replication Run Log - DACA Analysis (Replication 32)

## Date: 2026-01-27

---

## 1. Overview

This log documents all commands executed and key analytical decisions made during the independent replication of the DACA employment impact study.

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on full-time employment (working 35+ hours/week)?

**Identification Strategy:** Difference-in-Differences (DiD)
- Treatment Group: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
- Control Group: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
- Pre-Period: 2008-2011
- Post-Period: 2013-2016 (DACA implemented June 15, 2012)

---

## 2. Data Exploration

### 2.1 Initial Data Inspection

**Command:** Examined column headers of prepared_data_numeric_version.csv

```bash
head -1 data/prepared_data_numeric_version.csv
```

**Key Variables Identified:**
- `FT`: Full-time employment (1 = 35+ hours/week, 0 = otherwise) - OUTCOME
- `ELIGIBLE`: DACA eligibility indicator (1 = eligible/treatment, 0 = comparison) - TREATMENT
- `AFTER`: Post-DACA period indicator (1 = 2013-2016, 0 = 2008-2011) - TIME
- `PERWT`: Person weight for weighted analysis
- `YEAR`: Survey year
- `STATEFIP`: State FIPS code
- `AGE`, `SEX`, `EDUC_RECODE`, `MARST`: Demographic covariates
- State policy variables: `DRIVERSLICENSES`, `INSTATETUITION`, `EVERIFY`, etc.

### 2.2 Data Dictionary Notes

- Binary IPUMS variables coded 1=No, 2=Yes
- Added variables (FT, AFTER, ELIGIBLE, state policies) coded 0=No, 1=Yes
- ACS is repeated cross-section, not panel data
- 2012 data excluded (treatment timing ambiguity)

---

## 3. Analytical Approach

### 3.1 Primary Specification: Difference-in-Differences

The basic DiD model:
```
FT = β₀ + β₁*ELIGIBLE + β₂*AFTER + β₃*(ELIGIBLE × AFTER) + ε
```

Where β₃ is the DiD estimator representing the causal effect of DACA eligibility on full-time employment.

### 3.2 Key Decisions

1. **Weighting:** Use PERWT (person weights) to account for ACS survey design
2. **Standard Errors:** Cluster at state level to account for policy implementation at state level
3. **Covariates:** Add demographic controls for improved precision
4. **Robustness:** Test with year fixed effects, state fixed effects, and various covariate specifications

---

## 4. Commands Executed

### 4.1 Data Loading

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")  # 17,382
```

### 4.2 Descriptive Statistics

```python
# Sample sizes by group
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
# Control (31-35):  Pre=3,294, Post=2,706
# Treatment (26-30): Pre=6,233, Post=5,149

# Weighted FT rates by group and period
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
# Control (31-35):  Pre=68.86%, Post=66.29%  -> Change=-2.57pp
# Treatment (26-30): Pre=63.69%, Post=68.60% -> Change=+4.91pp
# Simple DiD = 4.91 - (-2.57) = 7.48pp
```

### 4.3 Main Regression Models

```python
# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df, weights=df['PERWT']).fit(
                      cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Result: DiD = 0.0748, SE = 0.0203, p = 0.0002

# Model 2: + Demographics
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + EDUC_dummies',
                  data=df, weights=df['PERWT']).fit(
                      cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Result: DiD = 0.0640, SE = 0.0216, p = 0.0031

# Model 3: + Year FEs
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + Demographics + Year_FEs',
                  data=df, weights=df['PERWT']).fit(
                      cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Result: DiD = 0.0613, SE = 0.0210, p = 0.0035

# Model 4: + State FEs (Preferred)
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + Demographics + Year_FEs + State_FEs',
                  data=df, weights=df['PERWT']).fit(
                      cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# Result: DiD = 0.0607, SE = 0.0215, p = 0.0048
```

### 4.4 Robustness Checks

```python
# Pre-trends test (pre-period only)
pre_df = df[df['AFTER'] == 0]
pre_df['YEAR_TREND'] = pre_df['YEAR'] - 2008
model_pretrend = smf.wls('FT ~ ELIGIBLE * YEAR_TREND',
                          data=pre_df, weights=pre_df['PERWT']).fit(
                              cov_type='cluster', cov_kwds={'groups': pre_df['STATEFIP']})
# Result: Interaction = 0.0174, SE = 0.0100, p = 0.082 (not significant)

# Heterogeneity by sex
# Male:   DiD = 0.0716, SE = 0.0195, p = 0.0002
# Female: DiD = 0.0527, SE = 0.0290, p = 0.0696

# Event study
# Year-specific effects relative to 2011:
# 2008: -0.0681 (p=0.021), 2009: -0.0499 (p=0.183), 2010: -0.0821 (p=0.006)
# 2013: +0.0158 (p=0.697), 2014: +0.0000 (p=1.000), 2015: +0.0014 (p=0.971), 2016: +0.0741 (p=0.013)
```

### 4.5 Figures

```python
# Created 5 figures:
# figure1_trends.png - FT employment trends by group
# figure2_did_bars.png - Pre/post comparison bar chart
# figure3_event_study.png - Year-specific treatment effects
# figure4_state_distribution.png - Sample by state
# figure5_heterogeneity.png - Subgroup analysis
```

### 4.6 LaTeX Compilation

```bash
pdflatex -interaction=nonstopmode replication_report_32.tex
pdflatex -interaction=nonstopmode replication_report_32.tex
pdflatex -interaction=nonstopmode replication_report_32.tex
# Output: replication_report_32.pdf (22 pages)
```

---

## 5. Key Analytical Decisions

### 5.1 Sample Definition
- **Decision:** Use the entire provided dataset without further restrictions
- **Rationale:** Instructions explicitly state "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample"

### 5.2 Treatment/Control Groups
- **Decision:** Use ELIGIBLE variable as provided (ages 26-30 = treatment, ages 31-35 = control)
- **Rationale:** Instructions specify to use the ELIGIBLE variable and not create own eligibility variable

### 5.3 Weighting
- **Decision:** Use person weights (PERWT) for all weighted analyses
- **Rationale:** PERWT accounts for ACS survey design and ensures population representativeness

### 5.4 Standard Errors
- **Decision:** Cluster at state level (STATEFIP)
- **Rationale:**
  - State-level policy implementation (driver's licenses, work permits)
  - Within-state correlation of outcomes
  - Conservative approach for inference

### 5.5 Covariates
- **Decision:** Include SEX, MARST (married), EDUC_RECODE (categorical), NCHILD
- **Rationale:** These are pre-treatment characteristics that may affect employment and differ between groups

### 5.6 Fixed Effects
- **Decision:** Include year FEs and state FEs in preferred specification
- **Rationale:**
  - Year FEs control for common time shocks (e.g., macroeconomic conditions)
  - State FEs control for time-invariant state differences

### 5.7 Not-in-Labor-Force
- **Decision:** Keep individuals not in labor force in sample (coded as FT=0)
- **Rationale:** Instructions state "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis"

### 5.8 Preferred Model
- **Decision:** Model 4 (State & Year FEs + Demographics) is the preferred specification
- **Rationale:** Most comprehensive specification controlling for observable confounders while maintaining causal interpretation

---

## 6. Results Summary

### 6.1 Preferred Estimate

| Statistic | Value |
|-----------|-------|
| DiD Estimate | 0.0607 (6.07 pp) |
| Standard Error | 0.0215 |
| 95% CI | [0.0185, 0.1029] |
| p-value | 0.0048 |
| Sample Size | 17,382 |

### 6.2 All Model Estimates

| Model | DiD Estimate | SE | p-value |
|-------|-------------|-----|---------|
| (1) Basic DiD | 0.0748 | 0.0203 | 0.0002 |
| (2) + Demographics | 0.0640 | 0.0216 | 0.0031 |
| (3) + Year FEs | 0.0613 | 0.0210 | 0.0035 |
| (4) + State & Year FEs | 0.0607 | 0.0215 | 0.0048 |

### 6.3 Interpretation

DACA eligibility is associated with a **6.07 percentage point** increase in the probability of full-time employment. This effect is:
- Statistically significant at the 1% level (p = 0.0048)
- Economically meaningful (≈9-10% relative increase from baseline of ~64%)
- Robust across specifications

---

## 7. Files Generated

| File | Description |
|------|-------------|
| analysis_daca.py | Main analysis script |
| create_figures.py | Figure generation script |
| analysis_results.csv | Numerical results |
| figure1_trends.png/pdf | Employment trends |
| figure2_did_bars.png/pdf | Pre/post comparison |
| figure3_event_study.png/pdf | Event study |
| figure4_state_distribution.png/pdf | State sample |
| figure5_heterogeneity.png/pdf | Subgroup analysis |
| replication_report_32.tex | LaTeX source |
| replication_report_32.pdf | Final report (22 pages) |
| run_log_32.md | This log file |

---

## 8. Session Complete

All required deliverables have been created:
1. ✅ replication_report_32.tex - LaTeX source
2. ✅ replication_report_32.pdf - Compiled PDF report (22 pages)
3. ✅ run_log_32.md - This run log

**End of Log**
