# Run Log - DACA Replication Study 43

## Overview
This log documents all commands executed and key decisions made during the independent replication of the DACA employment study.

**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours/week)?

**Method**: Difference-in-Differences comparing ages 26-30 (treatment) vs 31-35 (control) on June 15, 2012

---

## Step 1: Environment Setup and Data Exploration

### Date: January 26, 2026

### 1.1 Read Replication Instructions
- Read `replication_instructions.docx` using Python docx library
- Confirmed research design: DiD with age cutoff at 31

### 1.2 Explore Data Files
```
Directory: C:\Users\seraf\DACA Results Task 2\replication_43\data\
Files found:
- data.csv (33,851,425 rows including header)
- acs_data_dict.txt
- state_demo_policy.csv (not used)
- State Level Data Documentation.docx (not used)
```

### 1.3 Data Dictionary Review
Read `acs_data_dict.txt` to understand variable definitions:
- YEAR: 2006-2016 ACS samples
- HISPAN: Hispanic origin (1=Mexican)
- BPL: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1-4)
- UHRSWORK: Usual hours worked per week
- PERWT: Person weight

---

## Step 2: Sample Construction

### Key Decisions:

1. **Hispanic-Mexican ethnicity**: HISPAN = 1
   - Rationale: Instructions specify "ethnically Hispanic-Mexican"
   - Result: 2,945,521 observations (8.7% of total)

2. **Born in Mexico**: BPL = 200
   - Rationale: Instructions specify "Mexican-born"
   - Result: 991,261 observations

3. **Non-citizen status**: CITIZEN = 3
   - Rationale: Per instructions, assume non-citizens without papers are undocumented
   - Note: Cannot distinguish documented vs undocumented in ACS
   - Result: 701,347 observations

4. **Arrived before age 16**: (YRIMMIG - BIRTHYR) < 16
   - Rationale: DACA eligibility requirement
   - Result: 205,327 observations

5. **Continuous presence since 2007**: YRIMMIG <= 2007
   - Rationale: Proxy for continuous presence requirement (DACA requires presence since June 15, 2007)
   - Result: 195,023 observations

6. **Treatment group**: Ages 26-30 on June 15, 2012
   - Calculation: age_june_2012 = 2012 - BIRTHYR - 1[BIRTHQTR >= 3]
   - Adjustment subtracts 1 year if born in Q3 or Q4 (July-December)
   - Result: 27,903 observations (before excluding 2012)

7. **Control group**: Ages 31-35 on June 15, 2012
   - Same age calculation method
   - Result: 19,515 observations (before excluding 2012)

8. **Time periods**:
   - Pre-period: 2006-2011
   - Post-period: 2013-2016
   - Excluded: 2012 (cannot distinguish before/after June 15)

### Final Sample: 43,238 observations
- Treatment: 25,470
- Control: 17,768
- Pre-period: 28,377
- Post-period: 14,861

---

## Step 3: Variable Construction

### Outcome Variable
- **fulltime**: UHRSWORK >= 35
- Definition follows BLS full-time employment standard
- Mean: 62.7%

### Control Variables
- **female**: SEX == 2
- **married**: MARST == 1 (married, spouse present)
- **hs_or_more**: EDUC >= 6 (high school graduate or higher)
- **nchild**: NCHILD (number of children)
- **state**: STATEFIP (for fixed effects)

### DiD Variables
- **treat_group**: 1 if ages 26-30, 0 if ages 31-35
- **post**: 1 if YEAR >= 2013, 0 if YEAR <= 2011
- **treat_post**: treat_group * post (interaction term)

---

## Step 4: Estimation

### Model Specifications:

1. **Model 1 (Basic DiD)**:
   ```
   fulltime ~ treat_group + post + treat_post
   ```
   - DiD estimate: 0.0590 (SE: 0.0117)

2. **Model 2 (+ Controls)**:
   ```
   fulltime ~ treat_group + post + treat_post + female + married + hs_or_more + nchild
   ```
   - DiD estimate: 0.0487 (SE: 0.0107)

3. **Model 3 (+ Year FE)**:
   ```
   fulltime ~ treat_group + treat_post + controls + C(YEAR)
   ```
   - DiD estimate: 0.0468 (SE: 0.0107)

4. **Model 4 (+ State FE) - PREFERRED**:
   ```
   fulltime ~ treat_group + treat_post + controls + C(YEAR) + C(state)
   ```
   - DiD estimate: **0.0461** (SE: 0.0107)
   - 95% CI: [0.0252, 0.0671]
   - p-value: < 0.0001

### Estimation Details:
- Method: Weighted Least Squares (WLS)
- Weights: PERWT (person weights from IPUMS)
- Standard errors: Heteroskedasticity-robust (HC1)

---

## Step 5: Robustness Checks

### 5.1 Placebo Test
- Period: 2006-2011 only
- Artificial "post" = 2009-2011
- Result: DiD = -0.0026 (SE: 0.0125), p = 0.835
- Interpretation: No evidence of pre-trends

### 5.2 Narrower Age Bandwidth
- Treatment: ages 27-29
- Control: ages 32-34
- N = 25,606
- Result: DiD = 0.0356 (SE: 0.0137)
- Interpretation: Consistent positive effect

### 5.3 Heterogeneity by Sex
- Males: DiD = 0.0301 (SE: 0.0124)
- Females: DiD = 0.0609 (SE: 0.0180)
- Interpretation: Effect larger for females

### 5.4 Event Study
Year-by-year effects (reference: 2011):
| Year | Coefficient | SE |
|------|-------------|------|
| 2006 | 0.007 | 0.023 |
| 2007 | -0.029 | 0.022 |
| 2008 | 0.008 | 0.023 |
| 2009 | -0.008 | 0.023 |
| 2010 | -0.015 | 0.023 |
| 2013 | 0.035 | 0.024 |
| 2014 | 0.036 | 0.025 |
| 2015 | 0.022 | 0.025 |
| 2016 | 0.069 | 0.025 |

Interpretation: No pre-trends; positive post-DACA effects growing over time

---

## Step 6: Manual 2x2 DiD Verification

| Group | Pre-Period | Post-Period | Diff |
|-------|-----------|-------------|------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| **DiD** | | | **+0.059** |

Matches Model 1 regression result.

---

## Key Analytic Decisions and Rationale

1. **Age bandwidth (26-30 vs 31-35)**:
   - Chose 5-year bandwidth on each side of cutoff
   - Balances sample size vs comparability
   - Robustness check with narrower bandwidth confirms results

2. **Non-citizen proxy for undocumented**:
   - ACS cannot identify undocumented status directly
   - CITIZEN = 3 is the best available proxy
   - This likely attenuates the estimate (includes some documented non-citizens)

3. **Immigration year proxy for continuous presence**:
   - YRIMMIG <= 2007 proxies for "present since June 15, 2007"
   - Cannot verify continuous physical presence in ACS

4. **Excluded 2012**:
   - DACA announced June 15, 2012
   - ACS does not provide month of survey
   - Cannot distinguish pre/post observations in 2012

5. **Preferred specification includes state and year FE**:
   - Controls for state-specific factors
   - Controls for year-specific shocks
   - Most conservative specification

6. **Used person weights (PERWT)**:
   - Ensures nationally representative estimates
   - Standard practice for ACS analysis

---

## Output Files

1. **analysis.py**: Main analysis script
2. **results.json**: Saved results for reproducibility
3. **replication_report_43.tex**: LaTeX source for report
4. **replication_report_43.pdf**: Compiled report (~31 pages)
5. **run_log_43.md**: This file

---

## Preferred Estimate Summary

| Metric | Value |
|--------|-------|
| Point estimate | 0.0461 (4.61 percentage points) |
| Standard error | 0.0107 |
| 95% CI | [0.0252, 0.0671] |
| p-value | < 0.0001 |
| Sample size | 43,238 |
| Specification | DiD with state and year FE |

**Interpretation**: DACA eligibility is associated with a 4.6 percentage point increase in the probability of full-time employment among Hispanic-Mexican individuals born in Mexico who arrived in the US before age 16. This effect is statistically significant at the 1% level.

---

## Software Environment

- Python 3.14.2
- pandas 2.3.3
- numpy 2.3.5
- statsmodels 0.14.6
- scipy 1.16.3
- LaTeX: MiKTeX (for PDF compilation)

---

## Commands Executed

```bash
# Data exploration
cd "C:\Users\seraf\DACA Results Task 2\replication_43\data"
head -5 data.csv
wc -l data.csv

# Run analysis
cd "C:\Users\seraf\DACA Results Task 2\replication_43"
python analysis.py

# Compile report
pdflatex -interaction=nonstopmode replication_report_43.tex
pdflatex -interaction=nonstopmode replication_report_43.tex  # Second pass for references
```

---

## End of Log
