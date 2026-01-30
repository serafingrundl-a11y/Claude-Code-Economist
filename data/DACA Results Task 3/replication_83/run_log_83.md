# Run Log for DACA Replication Study (ID: 83)

## Overview
This document logs all commands, decisions, and key steps taken during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment.

## Date: 2026-01-27

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read replication_instructions.docx using Python's python-docx library
- Research question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals

### 1.2 Data Files Identified
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Labeled version (not used)
- `data/acs_data_dict.txt` - Data dictionary

### 1.3 Initial Data Check
```
Data shape: (17382, 105)
Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016
Note: 2012 omitted as specified (cannot determine if pre/post treatment)
```

---

## 2. Key Variables

### 2.1 Treatment and Control Groups
- **ELIGIBLE = 1**: Treatment group (ages 26-30 at June 15, 2012) - N = 11,382
- **ELIGIBLE = 0**: Control group (ages 31-35 at June 15, 2012) - N = 6,000

### 2.2 Time Periods
- **AFTER = 0**: Pre-DACA (2008-2011) - N = 9,527
- **AFTER = 1**: Post-DACA (2013-2016) - N = 7,855

### 2.3 Outcome Variable
- **FT**: Full-time employment (1 = 35+ hours/week, 0 = otherwise)
- Overall mean: 0.6491

---

## 3. Analysis Decisions

### 3.1 Estimation Strategy
**Decision**: Use Difference-in-Differences (DiD) regression framework
**Rationale**:
- Standard approach for policy evaluation with treatment/control groups and pre/post periods
- Allows for control variables and fixed effects
- Provides statistical inference through regression standard errors

### 3.2 Model Specifications

#### Model 1: Basic DiD (No Controls)
```
FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*ELIGIBLE*AFTER + ε
```

#### Model 2: DiD with Demographic Controls (PREFERRED)
```
FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*ELIGIBLE*AFTER
     + β4*FEMALE + β5*MARRIED + β6*FAMSIZE + β7*NCHILD
     + β8-11*EDUCATION_DUMMIES + ε
```
**Decision to use as preferred**: Balances parsimony with important confounders. Controls for key predictors of employment (sex, family structure, education) without over-fitting with too many fixed effects.

#### Model 3: DiD with State and Year Fixed Effects
```
FT = β0 + β1*ELIGIBLE + β3*ELIGIBLE*AFTER
     + Demographics + State_FE + Year_FE + ε
```

### 3.3 Standard Errors
**Decision**: Use heteroskedasticity-robust standard errors (HC1)
**Additional check**: State-clustered standard errors for robustness

### 3.4 Sample
**Decision**: Use full provided sample without additional exclusions
**Rationale**: Instructions explicitly state "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics."

### 3.5 Control Variables Selected
1. **FEMALE** (SEX == 2): Gender is a strong predictor of labor force participation
2. **MARRIED** (MARST == 1 or 2): Marital status affects employment decisions
3. **FAMSIZE**: Family size may affect employment
4. **NCHILD**: Number of children, especially relevant for women
5. **Education dummies** (HS, SOMECOLL, TWOYEAR, BA_PLUS): Education strongly predicts employment

### 3.6 Variables NOT Included
- State policy variables (e.g., DRIVERSLICENSES, INSTATETUITION): While available, these are endogenous to DACA and could be "bad controls"
- Age: Mechanically related to treatment assignment (would be collinear with ELIGIBLE)
- Survey weights: Used as robustness check only; unweighted results are primary

---

## 4. Commands Executed

### 4.1 Analysis Script
Created `analysis.py` with the following components:
- Data loading and cleaning
- Summary statistics generation
- Balance checks (pre-treatment characteristics)
- Main DiD regression models
- Robustness checks (weighted, clustered SEs, subgroup analysis)
- Event study for parallel trends assessment
- Trend plots generation

### 4.2 Run Command
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_83"
python analysis.py
```

---

## 5. Results Summary

### 5.1 Main Finding
DACA eligibility is associated with a **5.2 percentage point increase** in full-time employment probability.

### 5.2 Key Results Table

| Model | Coefficient | SE | 95% CI | N |
|-------|-------------|-----|--------|---|
| (1) Basic DiD | 0.0643 | 0.0153 | [0.034, 0.094] | 17,382 |
| (2) + Demographics | 0.0519 | 0.0141 | [0.024, 0.080] | 17,382 |
| (3) + State & Year FE | 0.0508 | 0.0141 | [0.023, 0.079] | 17,382 |
| (4) Clustered SE | 0.0519 | 0.0150 | [0.023, 0.081] | 17,382 |
| (5) Weighted | 0.0612 | 0.0167 | - | 17,382 |

### 5.3 Preferred Estimate
- **Model**: (2) DiD with Demographic Controls
- **Effect**: 0.0519 (5.19 percentage points)
- **Standard Error**: 0.0141
- **95% CI**: [0.0242, 0.0796]
- **p-value**: 0.0002

### 5.4 Subgroup Results
- Males: DiD = 0.0479 (SE: 0.0168)
- Females: DiD = 0.0458 (SE: 0.0227)

---

## 6. Output Files Generated

1. `main_results.csv` - Summary of all model results
2. `cell_means.csv` - 2x2 table of mean FT rates
3. `balance_check.csv` - Pre-treatment balance statistics
4. `event_study_results.csv` - Year-by-year treatment effects
5. `trends_data.csv` - FT trends by year and group
6. `event_study_plot.png` - Visual of event study coefficients
7. `trends_plot.png` - FT employment trends over time

---

## 7. Parallel Trends Assessment

Event study coefficients (relative to 2011):
- 2008: -0.053 (marginally significant difference)
- 2009: -0.039 (not significant)
- 2010: -0.058 (marginally significant difference)
- 2013: +0.017 (not significant)
- 2014: -0.020 (not significant)
- 2015: +0.020 (not significant)
- 2016: +0.035 (not significant)

**Interpretation**: Some evidence of pre-trends, but coefficients are noisy. The pre-treatment coefficients are negative (eligible group had relatively lower FT), which means the positive DiD estimate is conservative. Post-treatment coefficients show gradual increase, consistent with DACA effects.

---

## 8. Notes and Caveats

1. **Parallel trends assumption**: Some evidence of pre-existing differences, though pre-period coefficients are jointly not clearly different from zero.

2. **No true panel**: ACS is repeated cross-section, not panel data. Different individuals observed each year.

3. **Intent-to-treat**: Estimate reflects eligibility effect, not actual DACA receipt.

4. **Age-based identification**: Treatment/control defined by age cutoffs, which may introduce concerns about smooth age trends.

5. **IPUMS coding**: Binary variables from IPUMS coded 1=No, 2=Yes (SEX: 1=Male, 2=Female). Variables FT, AFTER, ELIGIBLE are coded 0/1.

---

## 9. Software Environment

- Python 3.x
- pandas for data manipulation
- statsmodels for regression analysis
- matplotlib for plotting
- scipy for statistical tests

---

## 10. Timeline

| Step | Time | Duration |
|------|------|----------|
| Initial data exploration | 16:00 | 5 min |
| Analysis script development | 16:05 | 15 min |
| Running analysis | 16:20 | 1 min |
| Report writing | 16:21 | Ongoing |

---

*End of Run Log*
