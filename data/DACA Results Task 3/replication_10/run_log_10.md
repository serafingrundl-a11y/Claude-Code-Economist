# Run Log - DACA Replication Study 10

## Overview
This document logs all commands executed and key decisions made during this independent replication study of the effect of DACA eligibility on full-time employment.

**Date:** January 27, 2026
**Study:** Independent replication of DACA employment effects

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Replication Instructions
- Read `replication_instructions.docx` using Python docx library
- Key requirements identified:
  - Research question: Effect of DACA eligibility on full-time employment
  - Treatment group: Ages 26-30 as of June 15, 2012 (ELIGIBLE=1)
  - Control group: Ages 31-35 as of June 15, 2012 (ELIGIBLE=0)
  - Outcome: Full-time employment (FT, 35+ hours/week)
  - Post-period: 2013-2016 (AFTER=1)
  - Pre-period: 2008-2011 (AFTER=0)
  - Year 2012 excluded from analysis

### 1.2 Data File Inspection
```bash
ls -la data/
```
Files found:
- `acs_data_dict.txt` - Variable dictionary (121,391 bytes)
- `prepared_data_labelled_version.csv` - Data with labels (18,988,640 bytes)
- `prepared_data_numeric_version.csv` - Numeric data (6,458,555 bytes)

### 1.3 Data Structure Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f'Shape: {df.shape}')
print(df.dtypes)
```
Results:
- Total observations: 17,382
- Total variables: 105
- Years covered: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)

### 1.4 Key Variable Verification
- **FT (outcome):** 11,283 full-time (64.9%), 6,099 not full-time (35.1%)
- **ELIGIBLE (treatment):** 11,382 treatment (65.5%), 6,000 control (34.5%)
- **AFTER (time):** 9,527 pre-DACA, 7,855 post-DACA

---

## 2. Analysis Decisions

### 2.1 Estimation Method
**Decision:** Linear Probability Model (OLS)

**Rationale:**
- Coefficients directly interpretable as marginal effects on probability
- Easy to incorporate fixed effects
- Widely used in applied econometrics for DID designs
- Robust standard errors address heteroskedasticity concerns

### 2.2 Standard Errors
**Decision:** Heteroskedasticity-robust standard errors (HC1)

**Rationale:**
- Linear probability models inherently have heteroskedastic errors
- HC1 (Huber-White) provides consistent standard errors
- Standard in the literature

### 2.3 Weighting
**Decision:** Primary analysis unweighted; weighted results as robustness check

**Rationale:**
- Unweighted estimates are more transparent for causal inference
- Weights can introduce additional variability
- Weighted estimates reported for comparison

### 2.4 Covariates
**Decision:** Include demographics (sex, marital status, children, age) and education

**Variables created:**
- `FEMALE` = 1 if SEX == 2
- `MARRIED` = 1 if MARST in [1, 2]
- `EDUC_HS`, `EDUC_SOMECOLL`, `EDUC_TWOYEAR`, `EDUC_BAPLUS` from EDUC_RECODE

**Rationale:**
- Improves precision by controlling for predictors of employment
- Addresses compositional differences between treatment and control groups
- Education is a key determinant of employment outcomes

### 2.5 Fixed Effects
**Decision:** Test year FE and state FE in robustness specifications

**Rationale:**
- Year FE allows flexible time trends
- State FE controls for time-invariant state differences
- Interaction term still identified by treatment-control differences within year/state

### 2.6 Preferred Specification
**Decision:** Model 5 - DID with demographic and education covariates (unweighted, robust SEs)

**Rationale:**
- Balances parsimony with comprehensiveness
- Controls for observable differences between groups
- Estimate is stable across specifications
- Standard errors appropriately account for heteroskedasticity

---

## 3. Analysis Execution

### 3.1 Main Analysis Script
Created: `analysis.py`

Key computations:
1. Data loading and verification
2. Descriptive statistics by group and period
3. Simple DID calculation (weighted and unweighted)
4. Regression models:
   - Model 1: Basic DID (unweighted)
   - Model 2: Basic DID (weighted)
   - Model 3: Basic DID (robust SEs)
   - Model 4: DID + demographics
   - Model 5: DID + demographics + education (PREFERRED)
   - Model 6: DID + covariates (weighted)
   - Model 7: DID + year FE
   - Model 8: DID + year FE + covariates
   - Model 9: DID + state FE
   - Model 10: Full model (year FE + state FE + covariates)
5. Parallel trends test
6. Covariate balance table
7. Heterogeneity by sex

### 3.2 Execution Command
```bash
python analysis.py
```

---

## 4. Results Summary

### 4.1 Main Findings

| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Basic DID | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| + Demographics | 0.0581 | 0.0142 | [0.030, 0.086] | <0.001 |
| + Education (PREFERRED) | 0.0559 | 0.0142 | [0.028, 0.084] | <0.001 |
| + Year FE | 0.0629 | 0.0152 | [0.033, 0.093] | <0.001 |
| + State FE | 0.0639 | 0.0153 | [0.034, 0.094] | <0.001 |
| Full Model | 0.0544 | 0.0142 | [0.027, 0.082] | <0.001 |

### 4.2 Preferred Estimate
- **Effect Size:** 0.0559 (5.59 percentage points)
- **Standard Error:** 0.0142
- **95% CI:** [0.028, 0.084]
- **p-value:** < 0.001
- **Sample Size:** 17,382
- **R-squared:** 0.131

### 4.3 Parallel Trends Test
- Pre-period trend interaction coefficient: 0.0151
- p-value: 0.098
- **Conclusion:** Parallel trends assumption appears satisfied

### 4.4 Heterogeneity by Sex
- Males: 0.0615 (SE: 0.0170, p < 0.001)
- Females: 0.0452 (SE: 0.0232, p = 0.051)

---

## 5. Visualization

### 5.1 Figures Created
Created: `create_figures.py`

```bash
python create_figures.py
```

Figures generated:
1. `figure1_parallel_trends.png` - FT employment trends by treatment status
2. `figure2_did_visualization.png` - DID bar chart
3. `figure3_ft_distribution.png` - FT distribution by group/period
4. `figure4_event_study.png` - Event study plot
5. `figure5_coefficient_plot.png` - Model comparison
6. `figure6_heterogeneity_sex.png` - Heterogeneity by sex
7. `figure7_sample_sizes.png` - Sample sizes by year

---

## 6. Report Compilation

### 6.1 LaTeX Report
Created: `replication_report_10.tex`

Sections:
1. Introduction (background, research question, identification strategy)
2. Data (source, sample construction, key variables, descriptive statistics)
3. Empirical Strategy (DID framework, extended specifications)
4. Results (main results, preferred specification, magnitude interpretation)
5. Validity and Robustness (parallel trends, event study, covariate balance, heterogeneity)
6. Discussion (summary, mechanisms, limitations)
7. Conclusion
8. Appendices (additional figures, variable definitions, analytical decisions)

### 6.2 PDF Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_10.tex
pdflatex -interaction=nonstopmode replication_report_10.tex
pdflatex -interaction=nonstopmode replication_report_10.tex
```
(Multiple passes to resolve cross-references)

Output: `replication_report_10.pdf` (21 pages)

---

## 7. Output Files

### 7.1 Required Deliverables
- [x] `replication_report_10.tex` - LaTeX source
- [x] `replication_report_10.pdf` - Final report (21 pages)
- [x] `run_log_10.md` - This run log

### 7.2 Additional Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `regression_results.csv` - Regression results table
- `annual_means.csv` - Annual means by group
- `figure1_parallel_trends.png` - Parallel trends figure
- `figure2_did_visualization.png` - DID visualization
- `figure3_ft_distribution.png` - FT distribution
- `figure4_event_study.png` - Event study
- `figure5_coefficient_plot.png` - Coefficient comparison
- `figure6_heterogeneity_sex.png` - Heterogeneity analysis
- `figure7_sample_sizes.png` - Sample sizes

---

## 8. Key Findings Summary

**Research Question:** What was the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals?

**Preferred Estimate:** DACA eligibility increased full-time employment by **5.59 percentage points** (SE = 0.014, 95% CI: [0.028, 0.084], p < 0.001)

**Interpretation:**
- This represents an **8.9% relative increase** from the pre-treatment mean of 62.6%
- The effect is **statistically significant** at all conventional levels
- Results are **robust** across multiple specifications
- **Parallel trends assumption appears satisfied** based on pre-period test
- Effect is **slightly larger for males** than females, though not significantly different

---

## 9. Reproducibility Notes

### Software Environment
- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (visualization)
- scipy (statistical tests)
- pdflatex/MiKTeX (LaTeX compilation)

### To Reproduce
1. Ensure required Python packages are installed
2. Run `python analysis.py` to generate results
3. Run `python create_figures.py` to generate figures
4. Run `pdflatex replication_report_10.tex` (3 times) to compile report

---

*End of Run Log*
