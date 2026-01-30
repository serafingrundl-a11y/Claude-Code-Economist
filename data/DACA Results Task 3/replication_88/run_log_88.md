# Replication Run Log - Study 88

## Overview
Independent replication of DACA (Deferred Action for Childhood Arrivals) impact study on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design
- **Treatment group**: DACA-eligible individuals aged 26-30 at the time policy went into effect (June 15, 2012)
- **Control group**: Otherwise eligible individuals aged 31-35 at the time policy went into effect (ineligible due to age cutoff)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment period**: 2008-2011
- **Post-treatment period**: 2013-2016 (2012 excluded as treatment year)

## Key Variables Used
- `FT`: Full-time employment indicator (outcome, 1=employed full-time, 0=not)
- `ELIGIBLE`: Treatment group indicator (1=ages 26-30 at June 2012, 0=ages 31-35)
- `AFTER`: Post-treatment period indicator (1=2013-2016, 0=2008-2011)
- `PERWT`: Person weight for survey weighting
- `SEX`: Sex (Male/Female)
- `EDUC_RECODE`: Education level (recoded categories)
- `YEAR`: Survey year

---

## Session Log

### Step 1: Environment Setup and Data Exploration
**Date**: Analysis session started

**Commands executed**:
```bash
# Check available software
which python python3 R
python --version
pip show pandas statsmodels scipy numpy matplotlib
where pdflatex
```

**Findings**:
- Python 3.14.2 available with required packages (pandas, statsmodels, numpy, scipy, matplotlib)
- LaTeX available via MiKTeX

**Data exploration**:
- Loaded `prepared_data_labelled_version.csv`
- Total observations: 17,382
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations

### Step 2: Sample Verification
**Age distribution check**:
- Control group (ELIGIBLE=0): Mean age at June 2012 = 32.93 (range: 31-35)
- Treatment group (ELIGIBLE=1): Mean age at June 2012 = 28.11 (range: 26-30.75)

**Sample sizes by year and group**:
| Year | Control (31-35) | Treatment (26-30) |
|------|-----------------|-------------------|
| 2008 | 848             | 1,506             |
| 2009 | 816             | 1,563             |
| 2010 | 851             | 1,593             |
| 2011 | 779             | 1,571             |
| 2013 | 747             | 1,377             |
| 2014 | 707             | 1,349             |
| 2015 | 623             | 1,227             |
| 2016 | 629             | 1,196             |

### Step 3: Descriptive Statistics
**Demographic balance between groups**:
- Sex: Similar distribution (~52% male in both groups)
- Education: Similar distributions, majority have High School Degree
- Marital status: Treatment group has more "Never married" (47% vs 34%), reflecting younger age

### Step 4: Main DiD Analysis

**Key Decision**: Use weighted least squares (WLS) with person weights (PERWT) for main analysis to account for survey design.

**2x2 Table of Full-Time Employment Rates (Weighted)**:

|                    | Pre (2008-2011) | Post (2013-2016) | Change    |
|--------------------|-----------------|------------------|-----------|
| Control (31-35)    | 68.86%          | 66.29%           | -2.57 pp  |
| Treatment (26-30)  | 63.69%          | 68.60%           | +4.91 pp  |
| **DiD Estimate**   |                 |                  | **+7.48 pp** |

### Step 5: Regression Models Estimated

1. **Model 1**: Basic OLS DiD (unweighted)
   - Coefficient: 0.0643 (SE: 0.0153, p < 0.001)

2. **Model 2**: Weighted DiD (WLS with PERWT)
   - Coefficient: 0.0748 (SE: 0.0152, p < 0.001)

3. **Model 3**: Weighted DiD with Robust SE (HC1)
   - Coefficient: 0.0748 (SE: 0.0181, p < 0.001)

4. **Model 4**: Weighted DiD + Sex Control
   - Coefficient: 0.0634 (SE: 0.0168, p < 0.001)

5. **Model 5**: Weighted DiD + Year Fixed Effects (PREFERRED)
   - Coefficient: 0.0721 (SE: 0.0181, p < 0.001)

6. **Model 6**: Year FE + Sex Control
   - Coefficient: 0.0607 (SE: 0.0168, p < 0.001)

7. **Model 7**: Full Model (Year FE + Sex + Education)
   - Coefficient: 0.0583 (SE: 0.0167, p < 0.001)

### Step 6: Robustness Checks

**Parallel Trends Check**:
- Pre-treatment differences (Treatment - Control):
  - 2008: -6.70 pp
  - 2009: -4.88 pp
  - 2010: -8.10 pp
  - 2011: +0.11 pp
- Average pre-treatment difference: -4.90 pp
- Average post-treatment difference: +2.39 pp
- Change in difference (DiD): +7.28 pp

**Placebo Test** (2008-2009 vs 2010-2011 within pre-period):
- Coefficient: 0.0178 (SE: 0.0241, p = 0.461)
- Interpretation: No significant pre-trend detected, supports parallel trends assumption

**Heterogeneous Effects by Sex**:
- Female: 0.0527 (SE: 0.0281, p = 0.061) - marginally significant
- Male: 0.0716 (SE: 0.0199, p < 0.001) - strongly significant

**Heterogeneous Effects by Education**:
- High School Degree: 0.0608 (SE: 0.0214, p = 0.005)
- Some College: 0.0672 (SE: 0.0437, p = 0.124)
- BA+: 0.1619 (SE: 0.0714, p = 0.023)
- Two-Year Degree: 0.1816 (SE: 0.0765, p = 0.018)

### Step 7: Final Results and Interpretation

**PREFERRED ESTIMATE (Model 5: Weighted DiD with Year FE)**:
- **Effect size**: 0.0721 (7.21 percentage points)
- **Standard Error**: 0.0181
- **95% CI**: [0.037, 0.108]
- **P-value**: 0.0001
- **Sample Size**: 17,382

**Interpretation**: DACA eligibility is associated with a 7.21 percentage point increase in the probability of full-time employment among eligible Mexican-born individuals aged 26-30 compared to the control group aged 31-35. This effect is statistically significant at the 1% level.

---

## Key Analytical Decisions

1. **Weighting**: Used person weights (PERWT) to account for ACS survey design
2. **Standard Errors**: Used heteroskedasticity-robust standard errors (HC1)
3. **Preferred Specification**: Year fixed effects model chosen over basic DiD to control for aggregate time trends
4. **Sample**: Did not limit sample by any subgroup characteristics per instructions
5. **Covariates**: Tested models with sex and education controls; main results robust

## Files Generated
- `analysis.py`: Main Python analysis script
- `results_summary.csv`: Summary of all model coefficients
- `yearly_effects.csv`: Year-by-year treatment effects
- `replication_report_88.tex`: LaTeX replication report
- `replication_report_88.pdf`: Final PDF report

---

## Technical Notes
- Software: Python 3.14.2, pandas 2.3.3, statsmodels 0.14.6
- LaTeX: MiKTeX distribution
- All analysis code available in `analysis.py`
- Analysis is fully reproducible from the provided data file
