# Replication Run Log - ID 77

## Date: 2026-01-27

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design
- **Treatment group**: Individuals ages 26-30 at the time DACA went into effect (June 15, 2012)
- **Control group**: Individuals ages 31-35 at the time DACA went into effect
- **Design**: Difference-in-Differences (DiD)
- **Pre-treatment period**: 2008-2011
- **Post-treatment period**: 2013-2016 (2012 excluded as policy transition year)
- **Outcome**: Full-time employment (FT = 1 if working 35+ hours/week)

## Data Description
- Source: American Community Survey (ACS) via IPUMS USA
- Years: 2008-2011 (pre) and 2013-2016 (post), 2012 excluded
- Key variables provided:
  - ELIGIBLE: 1 = treatment group (ages 26-30 in June 2012), 0 = control group (ages 31-35)
  - AFTER: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
  - FT: 1 = full-time employment (35+ hours/week), 0 = not full-time

---

## Analysis Steps and Commands

### Step 1: Data Loading and Initial Exploration
**Timestamp**: Start of analysis

**Command**: Load CSV data files
- File: prepared_data_numeric_version.csv

**Observations**:
- Total observations: 17,382
- Data contains prepared variables including ELIGIBLE, AFTER, FT
- Sample already restricted to eligible population (Mexican-born, Hispanic ethnicity)
- Year distribution: 2008 (2,354), 2009 (2,379), 2010 (2,444), 2011 (2,350), 2013 (2,124), 2014 (2,056), 2015 (1,850), 2016 (1,825)
- 2012 excluded from data as transition year

### Step 2: Sample Composition
**Sample sizes by group:**
- Treatment (ages 26-30), Pre-DACA: 6,233 observations
- Treatment (ages 26-30), Post-DACA: 5,149 observations
- Control (ages 31-35), Pre-DACA: 3,294 observations
- Control (ages 31-35), Post-DACA: 2,706 observations

**Key Decision**: Used full sample without dropping any observations, as specified in instructions.

### Step 3: Difference-in-Differences Estimation

**Model Specification**:
```
FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE × AFTER) + Xγ + ε
```

Where:
- FT = full-time employment indicator
- ELIGIBLE = 1 for treatment group (ages 26-30 in June 2012)
- AFTER = 1 for post-DACA period (2013-2016)
- β₃ = DiD estimate (causal effect of DACA eligibility)
- X = vector of control variables

**Models Estimated**:
1. Basic DiD (OLS, no weights)
2. DiD with survey weights (WLS)
3. DiD with demographic controls (sex, marital status, children, education)
4. DiD with year fixed effects
5. DiD with state fixed effects
6. DiD with robust standard errors (HC1)

### Step 4: Key Decisions

1. **Weighting**: Used ACS person weights (PERWT) for population-representative estimates
2. **Standard Errors**: Used heteroskedasticity-robust (HC1) standard errors
3. **Fixed Effects**: Included year and state fixed effects to control for aggregate trends
4. **Controls**: Included sex, marital status, number of children, and education dummies
5. **Estimation Method**: Linear Probability Model (LPM) for ease of interpretation

### Step 5: Results Summary

**Preferred Estimate (Model 5 with robust SEs)**:
- DiD Coefficient: 0.0607
- Robust Standard Error: 0.0166
- 95% Confidence Interval: [0.0281, 0.0933]
- p-value: 0.0003
- Sample Size: 17,382

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 6.07 percentage points among the treatment group (ages 26-30) relative to the control group (ages 31-35).

### Step 6: Robustness Checks

1. **Event Study**: Year-by-year treatment effects show no significant pre-trends (2009, 2010 coefficients not significantly different from zero) but significant positive effects in post-DACA years (especially 2013, 2016).

2. **Subgroup Analysis**:
   - Males: DiD = 0.0606 (SE: 0.0196)
   - Females: DiD = 0.0490 (SE: 0.0271)

3. **Balance Check**: Pre-treatment differences exist in age, marital status, and children (by design), but education levels are similar.

### Step 7: Output Files Generated
- analysis.py - Main analysis script
- analysis_extended.py - Extended analysis with visualizations
- figure1_trends.png - Employment trends by group
- figure2_eventstudy.png - Event study plot
- summary_statistics.csv - Summary statistics table
- balance_table.csv - Pre-treatment balance table
- analysis_results.txt - Complete results summary
- replication_report_77.tex - LaTeX report
- replication_report_77.pdf - PDF report

