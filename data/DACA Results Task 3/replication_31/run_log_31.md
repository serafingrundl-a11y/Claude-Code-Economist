# Replication Run Log - Replication 31

## Date: January 27, 2026

## Overview
This document logs all commands, key decisions, and analytical steps taken during the independent replication of the DACA impact study on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

---

## 1. Initial Setup and Data Review

### Understanding the Research Question
- **Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability of full-time employment (35+ hours/week)?

### Study Design
- **Treatment Group**: Individuals ages 26-30 at the time DACA went into place (June 15, 2012), who were eligible for DACA
- **Control Group**: Individuals ages 31-35 at the time DACA went into place, who would have been eligible except for their age
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment Period**: 2008-2011
- **Post-treatment Period**: 2013-2016 (2012 excluded as treatment timing unclear)

### Data Files Reviewed
- `prepared_data_labelled_version.csv`: Full labelled dataset
- `prepared_data_numeric_version.csv`: Numeric version of dataset
- `acs_data_dict.txt`: IPUMS ACS data dictionary

### Key Variables Identified
| Variable | Description |
|----------|-------------|
| `ELIGIBLE` | 1 = Treatment group (ages 26-30 in June 2012), 0 = Control group (ages 31-35) |
| `AFTER` | 1 = Post-DACA period (2013-2016), 0 = Pre-DACA period (2008-2011) |
| `FT` | 1 = Full-time employment (35+ hrs/week), 0 = Not full-time |
| `PERWT` | Person weight for survey weighting |
| `YEAR` | Survey year |
| `SEX` | 1 = Male, 2 = Female |
| `EDUC_RECODE` | Education level (recoded) |
| `MARST` | Marital status |
| `AGE` | Current age at survey |
| `STATEFIP` | State FIPS code |

### Sample Size
- Total observations: 17,382
- Treatment group: 11,382
- Control group: 6,000

---

## 2. Data Exploration Commands

```python
# Load and explore data
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(df.shape)  # (17382, 105)

# Check key variable distributions
print(df['ELIGIBLE'].value_counts())  # 1: 11382, 0: 6000
print(df['AFTER'].value_counts())     # 0: 9527, 1: 7855
print(df['FT'].value_counts())        # 1: 11283, 0: 6099
```

---

## 3. Analytical Decisions

### Decision 1: Use Survey Weights (PERWT)
- **Rationale**: ACS data uses complex survey design; PERWT ensures nationally representative estimates
- **Implementation**: All descriptive statistics and regression analyses weighted by PERWT

### Decision 2: Basic DiD Specification
- **Model**: FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE × AFTER) + ε
- **Key Parameter**: β₃ represents the DiD estimate (ATT)

### Decision 3: Extended Model with Covariates
- Include demographic controls: sex, education, marital status, age
- Include state fixed effects
- Include year fixed effects
- **Rationale**: Improve precision and control for observable differences between groups

### Decision 4: Cluster Standard Errors
- Cluster at state level to account for within-state correlation
- **Rationale**: DACA effects may vary by state due to complementary policies (driver's licenses, in-state tuition)

### Decision 5: Handle Missing Education Values
- 3 observations had missing `EDUC_RECODE` values
- **Solution**: Filled with 'Unknown' category and included as a separate level in regression
- **Rationale**: Preserve full sample size and avoid potential bias from dropping observations

---

## 4. Analysis Commands

### Basic DiD Model
```python
import statsmodels.formula.api as smf

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic DiD
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['STATEFIP']})
```

### Extended Model with Covariates (PREFERRED)
```python
# Create female indicator
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Model with state and year fixed effects
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + C(MARST) + C(EDUC_NUM) + C(STATEFIP) + C(YEAR)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='cluster',
                                           cov_kwds={'groups': df['STATEFIP']})
```

### Event Study
```python
# Create year-specific treatment effects
for year in years:
    df[f'ELIGIBLE_YEAR_{year}'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]
```

---

## 5. Results Summary

### Simple DiD (Weighted Means)
| Group | Period | FT Rate |
|-------|--------|---------|
| Treatment (26-30) | Pre-DACA | 63.69% |
| Treatment (26-30) | Post-DACA | 68.60% |
| Control (31-35) | Pre-DACA | 68.86% |
| Control (31-35) | Post-DACA | 66.29% |

- Treatment group change: +4.91 pp
- Control group change: -2.57 pp
- **Simple DiD**: +7.48 pp

### Regression Results
| Model | Estimate | SE | 95% CI | p-value |
|-------|----------|-----|--------|---------|
| 1. Basic DiD | 0.0748 | 0.0203 | [0.035, 0.114] | 0.0002 |
| 2. + Demographics | 0.0628 | 0.0209 | [0.022, 0.104] | 0.0027 |
| **3. + State/Year FE (Preferred)** | **0.0594** | **0.0208** | **[0.019, 0.100]** | **0.0043** |
| 4. + State Policies | 0.0585 | 0.0207 | [0.018, 0.099] | 0.0048 |
| 5. Males Only | 0.0612 | 0.0184 | [0.025, 0.097] | 0.0009 |
| 6. Females Only | 0.0413 | 0.0278 | [-0.013, 0.096] | 0.1373 |

### PREFERRED ESTIMATE (Model 3)
- **Effect Size**: 5.94 percentage points
- **Standard Error**: 0.0208
- **95% CI**: [1.87, 10.01] percentage points
- **p-value**: 0.0043

### Interpretation
DACA eligibility is associated with a statistically significant **increase** of approximately **5.94 percentage points** in the probability of full-time employment among eligible Hispanic-Mexican Mexican-born individuals.

---

## 6. Robustness Checks

### 1. Event Study (Pre-Trends Analysis)
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2008 | -0.0663 | 0.0261 | [-0.118, -0.015] |
| 2009 | -0.0466 | 0.0274 | [-0.100, 0.007] |
| 2010 | -0.0751 | 0.0313 | [-0.136, -0.014] |
| 2011 | 0 (ref) | --- | --- |
| 2013 | 0.0180 | 0.0363 | [-0.053, 0.089] |
| 2014 | -0.0154 | 0.0211 | [-0.057, 0.026] |
| 2015 | -0.0089 | 0.0331 | [-0.074, 0.056] |
| 2016 | 0.0573 | 0.0287 | [0.001, 0.114] |

**Note**: Pre-treatment coefficients suggest some concern about parallel trends, with 2008 and 2010 showing significant differences from 2011.

### 2. Heterogeneity by Sex
- **Males**: 6.12 pp (SE=0.0184), p=0.0009 - Statistically significant
- **Females**: 4.13 pp (SE=0.0278), p=0.1373 - Not statistically significant

### 3. Logit Model
- Log-odds coefficient: 0.283 (SE=0.061, p<0.001)
- Average Marginal Effect: 6.43 pp
- Qualitatively consistent with linear probability model results

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `analysis_script.py` | Main analysis Python script |
| `create_figures.py` | Figure generation script |
| `replication_report_31.tex` | LaTeX report source |
| `replication_report_31.pdf` | Compiled PDF report (21 pages) |
| `run_log_31.md` | This log file |
| `figures/ft_rates_by_group.png` | FT rates by group bar chart |
| `figures/event_study.png` | Event study plot |
| `figures/model_comparison.png` | Model comparison forest plot |
| `figures/ft_trends.png` | Time trends in FT employment |
| `figures/did_visualization.png` | DiD conceptual diagram |
| `tables/model_results.csv` | Regression results |
| `tables/event_study_results.csv` | Event study coefficients |
| `tables/summary_statistics.csv` | Summary statistics |
| `tables/key_results.pkl` | Python pickle with all results |

---

## 8. Software and Packages

- **Python**: 3.x
- **Key Packages**:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis, WLS, clustered SEs)
  - matplotlib (visualization)
  - scipy (statistical functions)

---

## 9. Key Methodological Notes

1. **Survey Weights**: All analyses use PERWT to ensure nationally representative estimates
2. **Standard Errors**: Clustered at state level (STATEFIP) throughout
3. **Fixed Effects**: State and year fixed effects included in preferred specification
4. **Outcome Variable**: FT is binary (0/1), coded as 1 if UHRSWORK >= 35
5. **Treatment Definition**: Based on pre-constructed ELIGIBLE variable in dataset
6. **Sample Restriction**: No additional sample restrictions beyond provided data

---

## 10. Limitations and Caveats

1. **Pre-Trends**: Event study shows some evidence of differential pre-trends, suggesting caution in causal interpretation
2. **Intent-to-Treat**: Estimates reflect eligibility effects, not actual DACA receipt
3. **Age-Based Identification**: Treatment/control groups differ by age, which may correlate with other employment determinants
4. **Repeated Cross-Section**: ACS is not panel data; we compare different individuals over time

---

## 11. Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment of approximately 5.94 percentage points. The effect is robust to inclusion of demographic controls, state and year fixed effects, and state policy controls. Heterogeneity analysis suggests larger effects for males compared to females. However, some concerns about parallel trends in the pre-treatment period suggest cautious interpretation of the results.

---

*Log completed: January 27, 2026*
