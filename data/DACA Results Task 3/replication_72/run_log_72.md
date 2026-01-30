# DACA Replication Study - Run Log

## Replication ID: 72
## Date: January 27, 2026

---

## Overview

This log documents all commands executed and key decisions made during the independent replication of the DACA eligibility effect on full-time employment study.

---

## Step 1: Initial Setup and Data Exploration

### 1.1 Read Replication Instructions
- Extracted text from `replication_instructions.docx` using Python `python-docx` library
- Research question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals
- Treatment group: Ages 26-30 as of June 2012
- Comparison group: Ages 31-35 as of June 2012
- Outcome: Full-time employment (FT = 1 if working 35+ hours/week)

### 1.2 Examine Data Files
**Files in data folder:**
- `prepared_data_numeric_version.csv` - Main analysis dataset
- `prepared_data_labelled_version.csv` - Labelled version of data
- `acs_data_dict.txt` - IPUMS data dictionary

**Data characteristics:**
- Total observations: 17,382
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Key pre-constructed variables: ELIGIBLE, AFTER, FT

### 1.3 Initial Data Summary
```
Sample sizes:
- Treatment (ELIGIBLE=1): 11,382 observations
- Comparison (ELIGIBLE=0): 6,000 observations
- Pre-DACA (AFTER=0): 9,527 observations
- Post-DACA (AFTER=1): 7,855 observations
```

---

## Step 2: Analysis Script Development

### 2.1 Analysis Approach Decisions

**Key methodological decisions:**
1. **Research design**: Difference-in-differences (DiD)
2. **Estimation method**: OLS linear probability model
3. **Preferred specification**: DiD with demographic controls (sex, marital status, children, education)
4. **Robustness checks**: Multiple specifications including state FE, year FE, robust SE, weighted estimates

**Rationale:**
- DiD is appropriate given the age-based eligibility cutoff and repeated cross-sectional data
- Linear probability model chosen for interpretability of coefficients as percentage point changes
- Demographic controls included to account for compositional differences between age groups
- Multiple specifications tested to assess robustness

### 2.2 Control Variables Selected
- FEMALE (derived from SEX)
- MARRIED (derived from MARST, codes 1-2 = married)
- NCHILD (number of children)
- Education dummies (High School, Some College, Two-Year, BA+; reference = Less than HS)

### 2.3 Analysis Script
Created `analysis_script.py` with the following components:
1. Data loading and preparation
2. Descriptive statistics
3. Basic DiD calculation
4. Regression-based DiD (8 specifications)
5. Robust standard errors
6. Subgroup analysis (by sex, education, marital status)
7. Parallel trends assessment
8. Event study analysis
9. Weighted analysis using PERWT
10. Results export

---

## Step 3: Analysis Execution

### 3.1 Run Analysis Script
**Command:**
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_72"
python analysis_script.py
```

### 3.2 Key Results

**Basic DiD Calculation:**
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.6263 | 0.6658 | +0.0394 |
| Comparison (31-35) | 0.6697 | 0.6449 | -0.0248 |
| **DiD Estimate** | | | **0.0643** |

**Regression Results Summary:**
| Model | Estimate | Std. Error | p-value |
|-------|----------|------------|---------|
| (1) Basic DiD | 0.0643 | 0.0153 | <0.001 |
| (2) + Demographics | 0.0556 | 0.0143 | <0.001 |
| (3) + Economic | 0.0519 | 0.0143 | <0.001 |
| (4) + State FE | 0.0556 | 0.0143 | <0.001 |
| (5) + Year FE | 0.0541 | 0.0143 | <0.001 |
| (6) + State + Year FE | 0.0542 | 0.0143 | <0.001 |
| (7) Robust SE | 0.0556 | 0.0142 | <0.001 |
| (8) Weighted | 0.0640 | 0.0142 | <0.001 |

**Preferred Estimate (Model 2):**
- Effect: 0.0556 (5.56 percentage points)
- SE: 0.0143
- 95% CI: [0.0276, 0.0836]
- p-value: <0.001
- Sample size: 17,382

**Parallel Trends Test:**
- Pre-trend interaction coefficient: 0.0151
- SE: 0.0093
- p-value: 0.103
- Result: No significant differential pre-trend (supports parallel trends assumption)

---

## Step 4: Report Generation

### 4.1 LaTeX Report Creation
Created `replication_report_72.tex` containing:
1. Abstract
2. Table of Contents
3. Introduction
4. Background
5. Data
6. Methodology
7. Results
8. Discussion
9. Conclusion
10. Appendix (Full model results, technical details)

### 4.2 PDF Compilation
**Commands:**
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_72"
pdflatex -interaction=nonstopmode replication_report_72.tex
pdflatex -interaction=nonstopmode replication_report_72.tex  # Second pass for cross-references
```

**Output:** `replication_report_72.pdf` (21 pages)

---

## Step 5: Output Files Generated

### Final Deliverables:
1. **replication_report_72.tex** - LaTeX source file
2. **replication_report_72.pdf** - Compiled PDF report (21 pages)
3. **run_log_72.md** - This run log

### Supporting Files:
4. `analysis_script.py` - Python analysis code
5. `results_summary.csv` - Exported results table
6. `model_summary.txt` - Full regression output

---

## Key Analytical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Research design | Difference-in-differences | Appropriate for age-based eligibility cutoff |
| Estimation | OLS (linear probability model) | Interpretable coefficients as pp changes |
| Controls | Demographics (sex, marital, children, education) | Account for compositional differences |
| Preferred model | Model 2 (with demographics) | Balance of parsimony and bias reduction |
| Standard errors | Conventional (with robust check) | Both yield similar inference |
| Weights | Primary unweighted (weighted as check) | Unweighted more common in DiD |

---

## Interpretation

DACA eligibility increased the probability of full-time employment by approximately 5.6 percentage points (SE = 0.014, p < 0.001) among ethnically Hispanic-Mexican Mexican-born individuals. This represents approximately an 8.9% relative increase from the baseline rate of 62.6% in the treatment group.

The effect is:
- Robust across multiple specifications
- Supported by parallel trends evidence
- Larger for males than females
- Particularly large for individuals with some college or two-year degrees

---

## Session Information

- Platform: Windows 11
- Python version: 3.x
- Key packages: pandas, numpy, statsmodels, scipy
- LaTeX: MiKTeX 25.12, pdfTeX 3.141592653-2.6-1.40.28

---

*End of run log*
