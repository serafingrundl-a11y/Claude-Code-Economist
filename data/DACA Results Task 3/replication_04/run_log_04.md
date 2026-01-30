# DACA Replication Study - Run Log 04

## Overview
Independent replication of DACA impact on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (usually working 35+ hours per week)?

**Identification Strategy**: Difference-in-differences comparing:
- Treated group: Ages 26-30 at time of DACA (June 2012) - ELIGIBLE=1
- Control group: Ages 31-35 at time of DACA (June 2012) - ELIGIBLE=0

**Time Periods**:
- Pre-DACA: 2008-2011 (AFTER=0)
- Post-DACA: 2013-2016 (AFTER=1)
- 2012 excluded from data

---

## Session Log

### Step 1: Environment Setup and Data Exploration
**Date**: 2026-01-26

1. Read replication_instructions.docx
   - Confirmed research question focuses on full-time employment (FT variable)
   - Treatment: DACA eligibility for ages 26-30 as of June 15, 2012
   - Control: Ages 31-35 as of June 15, 2012 (would be eligible but for age)
   - Post-treatment period: 2013-2016
   - Pre-treatment period: 2008-2011

2. Examined data files:
   - `prepared_data_labelled_version.csv` (18.9 MB) - labeled version with string categories
   - `prepared_data_numeric_version.csv` (6.5 MB) - numeric coded version
   - `acs_data_dict.txt` - IPUMS data dictionary

3. Key variables identified:
   - `FT`: Full-time employment (1=yes, 0=no) - OUTCOME
   - `AFTER`: Post-DACA period (1=2013-2016, 0=2008-2011)
   - `ELIGIBLE`: Treatment group indicator (1=ages 26-30 at DACA, 0=ages 31-35)
   - `AGE_IN_JUNE_2012`: Age at time of DACA implementation
   - `PERWT`: Person weight for survey weighting
   - Various demographic and state-level policy controls available

### Step 2: Data Loading and Initial Exploration

**Command executed:**
```bash
python analysis.py
```

**Key findings from data exploration:**
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 (65.5%)
- Control group (ELIGIBLE=0): 6,000 (34.5%)
- Pre-period (AFTER=0): 9,527 (54.8%)
- Post-period (AFTER=1): 7,855 (45.2%)

**Weighted Population:**
- Total weighted population: 2,416,349
- Treatment group weighted: 1,596,317 (66.1%)
- Control group weighted: 820,032 (33.9%)

**Full-time Employment Rates (weighted):**
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control (ages 31-35) | 68.86% | 66.29% | -2.57 pp |
| Treatment (ages 26-30) | 63.69% | 68.60% | +4.91 pp |

### Step 3: Difference-in-Differences Analysis

**Simple 2x2 DiD (weighted):**
- DiD Estimate = (68.60 - 63.69) - (66.29 - 68.86) = 7.48 percentage points

**Regression-based DiD:**

Model specification:
```
FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE × AFTER) + ε
```

Key decision: Used weighted least squares (WLS) with survey weights (PERWT) and robust (HC1) standard errors.

### Step 4: Model Specifications Tested

| Model | Description | DiD Estimate | SE | 95% CI |
|-------|-------------|--------------|-----|--------|
| 1 | Basic DiD (unweighted) | 0.0643 | 0.0153 | [0.034, 0.094] |
| 2 | Basic DiD (weighted) | 0.0748 | 0.0181 | [0.039, 0.110] |
| 3 | Year FE only | 0.0721 | 0.0181 | [0.037, 0.108] |
| 4 | State FE only | 0.0737 | 0.0180 | [0.038, 0.109] |
| 5 | Two-way FE | 0.0710 | 0.0180 | [0.036, 0.106] |
| 6 | Demographics | 0.0642 | 0.0167 | [0.031, 0.097] |
| 7 | Full w/ Education | 0.0583 | 0.0166 | [0.026, 0.091] |
| 8 | **Preferred** | **0.0641** | **0.0167** | **[0.031, 0.097]** |

### Step 5: Preferred Specification

**Decision**: Selected Model 8 as preferred specification:
- Year fixed effects (absorbs common time trends)
- State fixed effects (absorbs time-invariant state differences)
- Demographic controls: FEMALE, MARST, NCHILD
- Weighted by PERWT
- HC1 robust standard errors

**Rationale**:
1. Two-way fixed effects control for unobserved state and time heterogeneity
2. Demographic controls improve precision and address potential composition differences
3. Did not include education as it may be endogenous (DACA could affect educational attainment)
4. Survey weights account for complex sampling design

**Preferred Results:**
- DiD Coefficient: 0.0641
- Standard Error: 0.0167
- t-statistic: 3.85
- p-value: 0.0001
- 95% CI: [0.0314, 0.0967]
- N: 17,382
- R-squared: 0.1367

### Step 6: Robustness Checks

**6a. State-Clustered Standard Errors:**
- HC1 SE: 0.0167
- Cluster SE: 0.0207
- Effect remains significant at 1% level with clustering

**6b. Heterogeneity by Gender:**
- Male subsample: 0.0613 (SE: 0.0196), N=9,075
- Female subsample: 0.0560 (SE: 0.0273), N=8,307
- Effects are similar across gender

**6c. Pre-Trends Analysis:**
Year-specific effects (reference: 2011):
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2008 | -0.0660 | 0.0321 | 0.040 |
| 2009 | -0.0496 | 0.0330 | 0.132 |
| 2010 | -0.0732 | 0.0329 | 0.026 |
| 2013 | 0.0208 | 0.0340 | 0.541 |
| 2014 | -0.0124 | 0.0350 | 0.723 |
| 2015 | -0.0052 | 0.0348 | 0.881 |
| 2016 | 0.0662 | 0.0354 | 0.062 |

**Note on pre-trends**: Some pre-period coefficients are statistically significant, suggesting potential differential trends between treatment and control groups before DACA. This is a limitation of the analysis.

### Step 7: Final Results Summary

**PREFERRED ESTIMATE:**
- Effect of DACA eligibility on full-time employment: **6.41 percentage points**
- Standard Error: 0.0167
- 95% Confidence Interval: [3.14, 9.67] percentage points
- Sample Size: 17,382
- Statistical significance: p < 0.001

**Interpretation**: DACA eligibility is associated with a 6.41 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born individuals aged 26-30 at the time of DACA implementation, compared to those aged 31-35 who were otherwise similar but ineligible due to age.

---

## Key Analytical Decisions

1. **Weighted analysis**: Used PERWT survey weights for population-representative estimates
2. **Standard errors**: HC1 robust standard errors for main specification; clustered by state for robustness
3. **Fixed effects**: Included both year and state fixed effects
4. **Controls**: Added SEX, MARST, NCHILD; excluded EDUC due to potential endogeneity
5. **Sample**: Used full provided sample without further restrictions (per instructions)
6. **Functional form**: Linear probability model for interpretability

## Files Produced

1. `analysis.py` - Main analysis script
2. `results_summary.json` - Key results in JSON format
3. `replication_report_04.tex` - LaTeX report
4. `replication_report_04.pdf` - Compiled PDF report
5. `run_log_04.md` - This log file
