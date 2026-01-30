# Replication Run Log - Study 45

## Overview
- **Date**: January 27, 2026
- **Research Question**: Effect of DACA eligibility on full-time employment among ethnically Hispanic-Mexican individuals born in Mexico
- **Method**: Difference-in-Differences (DiD)
- **Treatment Group**: Ages 26-30 at June 15, 2012 (ELIGIBLE = 1)
- **Control Group**: Ages 31-35 at June 15, 2012 (ELIGIBLE = 0)
- **Outcome**: Full-time employment (FT = 1 if working 35+ hours/week)

## Data Processing Steps

### 1. Data Loading
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```
- **Dataset shape**: 17,382 observations, 105 variables
- **Years included**: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **Treatment group (ELIGIBLE=1)**: 11,382 observations
- **Control group (ELIGIBLE=0)**: 6,000 observations
- **Pre-period (AFTER=0)**: 9,527 observations
- **Post-period (AFTER=1)**: 7,855 observations

### 2. Variable Construction
```python
# Interaction term for DiD
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Female indicator (IPUMS: SEX=1 Male, SEX=2 Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Married indicator (IPUMS: MARST=1 or 2 = married)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
```

### 3. Key Decisions

1. **Used pre-constructed ELIGIBLE variable**: As instructed, did not create own eligibility variable
2. **Kept non-labor force participants**: As instructed, included those not in labor force with FT=0
3. **Applied survey weights (PERWT)**: Used weighted least squares (WLS) for population-representative estimates
4. **Clustered standard errors by state**: To account for within-state correlation of errors

### 4. Model Specifications

#### Model Progression:
| Model | Specification | DiD Estimate | SE |
|-------|--------------|--------------|-----|
| 1 | Basic DiD (weighted) | 0.0749 | 0.0152 |
| 2 | + Demographics (Age, Female, Married) | 0.0649 | 0.0142 |
| 3 | + Education FE | 0.0624 | 0.0142 |
| 4 | + Year FE | 0.0598 | 0.0142 |
| 5 | + State FE | 0.0591 | 0.0142 |
| 6 | State-clustered SE | 0.0591 | 0.0213 |

### 5. Preferred Specification
```python
model = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(STATEFIP)',
                data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP'].values})
```

**Key Results:**
- **DiD Effect (ELIGIBLE_AFTER)**: 0.0591
- **Clustered Standard Error**: 0.0213
- **95% Confidence Interval**: [0.0174, 0.1009]
- **p-value**: 0.0055
- **R-squared**: 0.138
- **N**: 17,379

## Main Findings

### Primary Result
DACA eligibility increased full-time employment by **5.91 percentage points** (95% CI: 1.74 to 10.09 pp), statistically significant at p < 0.01.

### Simple DiD Calculation (weighted)
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treated (26-30) | 63.7% | 68.6% | +4.9 pp |
| Control (31-35) | 68.9% | 66.3% | -2.6 pp |
| **DiD** | | | **+7.5 pp** |

### Heterogeneity Analysis
| Subgroup | DiD Estimate | SE | p-value |
|----------|--------------|-----|---------|
| Male | 0.0615 | 0.0170 | <0.001 |
| Female | 0.0419 | 0.0229 | 0.067 |
| Not Married | 0.0948 | 0.0213 | <0.001 |
| Married | 0.0014 | 0.0186 | 0.940 |
| HS or Less | 0.0455 | 0.0164 | 0.006 |
| Some College+ | 0.0984 | 0.0281 | <0.001 |

### Robustness Checks
| Specification | DiD Estimate | SE |
|--------------|--------------|-----|
| Unweighted | 0.0523 | 0.0143 |
| Without State FE | 0.0598 | 0.0142 |
| Region FE (not State) | 0.0592 | 0.0142 |
| With children control | 0.0584 | 0.0142 |

### Placebo Test
- Fake treatment in 2010 (pre-period only)
- Placebo DiD: 0.0188 (SE: 0.0194, p = 0.332)
- Not statistically significant, supporting parallel trends assumption

## Output Files Generated

1. **replication_report_45.tex** - LaTeX source (23 pages)
2. **replication_report_45.pdf** - Compiled PDF report
3. **run_log_45.md** - This log file
4. **figure1_parallel_trends.png/pdf** - Time trends by treatment group
5. **figure2_did_visualization.png/pdf** - DiD design visualization
6. **figure3_covariate_balance.png/pdf** - Covariate distributions
7. **figure4_event_study.png/pdf** - Event study plot
8. **figure5_heterogeneity_sex.png/pdf** - Heterogeneity by sex

## Software Used
- Python 3.x
- pandas
- numpy
- statsmodels
- matplotlib
- LaTeX (pdflatex via MiKTeX)

## Key Analytic Decisions

1. **Weights**: Used PERWT survey weights for all main analyses to ensure population representativeness
2. **Standard Errors**: Clustered at state level (50 states) in preferred specification
3. **Fixed Effects**: Included both year FE (7 years) and state FE (50 states)
4. **Covariates**: Controlled for age, sex, marital status, and education category
5. **Sample**: Used full provided sample without additional restrictions, as instructed
6. **Non-labor force**: Kept in sample with FT=0, as instructed

## Summary

The analysis finds strong evidence that DACA eligibility increased full-time employment by approximately 6 percentage points. This effect is:
- Statistically significant (p < 0.01)
- Robust across specifications
- Larger for unmarried individuals and those with some college education
- Supported by placebo tests showing no pre-treatment effects

The main limitation is potential violations of the parallel trends assumption, though event study analysis and placebo tests provide some support for the identifying assumptions.
