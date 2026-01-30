# Run Log - DACA Replication Study (Replication 16)

## Overview
This document logs all commands, key decisions, and analytical choices made during the independent replication study examining the effect of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Date: January 27, 2026

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Replication Instructions
- **Command**: Read `replication_instructions.docx` using Python `python-docx` library
- **Key Information Extracted**:
  - Research question: Effect of DACA eligibility on full-time employment
  - Treatment group: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
  - Control group: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
  - Pre-period: 2008-2011 (AFTER=0)
  - Post-period: 2013-2016 (AFTER=1)
  - Outcome: FT (Full-time employment, 35+ hours/week)
  - Year 2012 excluded from data

### 1.2 Data Files Identified
- `data/prepared_data_numeric_version.csv` - Main analysis file
- `data/prepared_data_labelled_version.csv` - Labelled version (not used)
- `data/acs_data_dict.txt` - Data dictionary

### 1.3 Initial Data Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Shape: {df.shape}")  # (17382, 105)
print(f"Columns: {list(df.columns)}")
```

**Key Findings**:
- Total observations: 17,382
- 105 variables including key variables: YEAR, ELIGIBLE, AFTER, FT, PERWT
- Years covered: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- AGE_IN_JUNE_2012 ranges from 26 to 35 (as specified)

---

## 2. Key Analytical Decisions

### 2.1 Identification Strategy
**Decision**: Use Difference-in-Differences (DiD) design
- **Rationale**: The research design explicitly specified comparing the treatment group (age 26-30) to the control group (age 31-35) before and after DACA implementation. DiD is the standard approach for this quasi-experimental setting with:
  - Clear treatment/control groups defined by age cutoff
  - Clear pre/post periods
  - Expectation of parallel trends in absence of treatment

### 2.2 Sample Definition
**Decision**: Use the entire provided dataset without additional restrictions
- **Rationale**: Instructions stated "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics"
- Used ELIGIBLE variable as provided (did not create own eligibility variable)
- Kept all observations including those not in labor force

### 2.3 Outcome Variable
**Decision**: Use FT (full-time employment) as binary outcome
- FT = 1 if usually works 35+ hours per week
- FT = 0 otherwise (including not in labor force)
- **Rationale**: Matches the research question definition

### 2.4 Weighting
**Decision**: Use person weights (PERWT) for preferred specification
- **Rationale**: ACS is a complex survey; using weights produces population-representative estimates
- Also estimated unweighted models for comparison/robustness

### 2.5 Standard Errors
**Decision**: Use heteroskedasticity-robust standard errors (HC1)
- **Rationale**: Binary outcome variable makes heteroskedasticity likely; robust SEs are conservative

### 2.6 Fixed Effects
**Decision**: Include state and year fixed effects in preferred specification
- State FE: Control for time-invariant state-level confounders
- Year FE: Control for common time shocks
- **Rationale**: Improves identification by controlling for unobserved heterogeneity

### 2.7 Control Variables
**Decision**: Include demographic controls:
- SEX (Male indicator)
- MARST (Married indicator)
- NCHILD (Number of children)
- YRSUSA1 (Years in USA)
- EDUC_RECODE (Education dummies)

**Not included**:
- AGE: Not included because age determines treatment status
- State-level policy variables: State FE absorb these

**Rationale**: Control for observable characteristics that predict employment and may differ between groups

### 2.8 Model Selection
**Preferred Model**: Weighted OLS with demographics, state FE, and year FE
- **Rationale**:
  - Survey weights provide population-representative estimates
  - Controls improve precision and reduce omitted variable bias
  - Fixed effects control for unobserved confounders
  - Results robust across specifications, giving confidence in findings

---

## 3. Analysis Code and Commands

### 3.1 Main Analysis Script
Created `analysis_code.py` with the following components:

```python
# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit(cov_type='HC1')

# Model 2: DiD with demographics
model2_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + MALE + MARRIED + NCHILD + YRSUSA1 + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA'

# Model 3: + State FE
model3_formula = model2_formula + ' + C(STATEFIP)'

# Model 4: + Year FE (AFTER absorbed)
model4_formula = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + MALE + MARRIED + NCHILD + YRSUSA1 + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA + C(STATEFIP) + C(YEAR)'

# Model 5-6: Weighted versions
model5 = smf.wls(..., weights=df['PERWT'])
model6 = smf.wls(model4_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')  # PREFERRED
```

### 3.2 Event Study Analysis
```python
# Create year interactions
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_X_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

event_formula = 'FT ~ ELIGIBLE + YEAR_2009 + ... + ELIGIBLE_X_2009 + ... + ELIGIBLE_X_2016'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

### 3.3 Parallel Trends Test
```python
df_pre = df[df['AFTER'] == 0].copy()
df_pre['TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_X_TREND'] = df_pre['ELIGIBLE'] * df_pre['TREND']
model_trend = smf.wls('FT ~ ELIGIBLE + TREND + ELIGIBLE_X_TREND', data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
```

### 3.4 Figure Generation
Created `create_figures.py`:
- Figure 1: Full-time employment rates by year and group
- Figure 2: Event study plot
- Figure 3: DiD visualization (2x2)
- Figure 4: Forest plot of regression coefficients
- Figure 5: Subgroup analysis by sex
- Figure 6: Variable distributions

---

## 4. Results Summary

### 4.1 Simple Difference-in-Differences
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.626 | 0.666 | +0.039 |
| Control (31-35) | 0.670 | 0.645 | -0.025 |
| **DiD** | | | **+0.064** |

### 4.2 Regression Results

| Model | DiD Estimate | SE | 95% CI | N |
|-------|--------------|-----|--------|---|
| Basic DiD (unweighted) | 0.064 | 0.015 | [0.034, 0.094] | 17,382 |
| + Demographics | 0.058 | 0.014 | [0.030, 0.086] | 17,382 |
| + State FE | 0.058 | 0.014 | [0.030, 0.086] | 17,382 |
| + Year FE | 0.056 | 0.014 | [0.029, 0.084] | 17,382 |
| Basic DiD (weighted) | 0.075 | 0.018 | [0.039, 0.110] | 17,382 |
| **Full (weighted) - PREFERRED** | **0.064** | **0.017** | **[0.031, 0.097]** | **17,382** |

### 4.3 Event Study Results
Pre-treatment coefficients (2009-2011): Not statistically significant, supports parallel trends
Post-treatment coefficients (2013-2016): Positive and significant, largest in 2016 (0.142)

### 4.4 Parallel Trends Test
- Differential trend coefficient: 0.017
- Standard error: 0.011
- P-value: 0.113
- **Conclusion**: Cannot reject parallel trends assumption

### 4.5 Subgroup Analysis
| Subgroup | DiD Estimate | SE |
|----------|--------------|-----|
| Male | 0.072 | 0.020 |
| Female | 0.053 | 0.028 |

---

## 5. Files Generated

### 5.1 Analysis Files
- `analysis_code.py` - Main analysis script
- `create_figures.py` - Figure generation script

### 5.2 Output Files
- `regression_results.csv` - Summary of regression estimates
- `event_study_results.csv` - Year-specific treatment effects
- `summary_statistics.csv` - Descriptive statistics by group

### 5.3 Figures
- `figure1_ft_rates_by_year.pdf` - Trends plot
- `figure2_event_study.pdf` - Event study
- `figure3_did_visualization.pdf` - DiD diagram
- `figure4_forest_plot.pdf` - Coefficient comparison
- `figure5_subgroup_sex.pdf` - Subgroup analysis
- `figure6_distributions.pdf` - Variable distributions

### 5.4 Report Files
- `replication_report_16.tex` - LaTeX source
- `replication_report_16.pdf` - Final report (21 pages)

---

## 6. Compilation Commands

```bash
# Run analysis
python analysis_code.py

# Generate figures
python create_figures.py

# Compile LaTeX (run 3 times for references)
pdflatex -interaction=nonstopmode replication_report_16.tex
pdflatex -interaction=nonstopmode replication_report_16.tex
pdflatex -interaction=nonstopmode replication_report_16.tex
```

---

## 7. Key Conclusions

### 7.1 Main Finding
DACA eligibility increased the probability of full-time employment by approximately **6.4 percentage points** (SE = 0.017, 95% CI: [0.031, 0.097], p < 0.001) among ethnically Hispanic-Mexican, Mexican-born individuals aged 26-30 at the time of implementation.

### 7.2 Interpretation
- Effect is economically meaningful (~10% increase relative to baseline)
- Statistically significant at p < 0.001
- Robust across model specifications
- Consistent with program's goal of improving labor market access

### 7.3 Validity Assessment
- Parallel trends assumption supported by:
  - Visual inspection of pre-treatment trends
  - Non-significant pre-treatment event study coefficients
  - Non-significant differential trend test (p = 0.113)
- Results robust to inclusion of controls and fixed effects

---

## 8. Software Environment

- **Python**: pandas, numpy, statsmodels, matplotlib
- **LaTeX**: pdflatex (MiKTeX)
- **Operating System**: Windows

---

## End of Run Log
