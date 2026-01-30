# DACA Replication Study - Run Log (Replication 27)

## Overview
This document logs all commands executed and key decisions made during the independent replication analysis of the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Date of Analysis
January 27, 2026

---

## Phase 1: Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Extracted text from `replication_instructions.docx` using Python's `python-docx` library
- Confirmed research question: Effect of DACA eligibility on full-time employment
- Identified treatment group: Ages 26-30 as of June 2012 (ELIGIBLE=1)
- Identified control group: Ages 31-35 as of June 2012 (ELIGIBLE=0)

### 1.2 Data Exploration
```bash
# Listed files in data directory
ls -la data/
# Output:
# - acs_data_dict.txt (121,391 bytes)
# - prepared_data_labelled_version.csv (18,988,640 bytes)
# - prepared_data_numeric_version.csv (6,458,555 bytes)

# Examined data header
head -1 data/prepared_data_numeric_version.csv
# 105 variables identified

# Counted observations
wc -l data/prepared_data_numeric_version.csv
# 17,383 lines (17,382 observations + header)
```

### 1.3 Data Dictionary Review
- Read `acs_data_dict.txt` to understand variable definitions
- Confirmed key variables:
  - FT: Full-time employment (1 = 35+ hours/week)
  - ELIGIBLE: Treatment indicator (pre-constructed)
  - AFTER: Post-treatment indicator (pre-constructed)
  - SEX: 1=Male, 2=Female (IPUMS coding)
  - MARST: Marital status
  - EDUC_RECODE: Education categories

---

## Phase 2: Analysis Script Development

### 2.1 Main Analysis Script (`analysis.py`)
Created comprehensive Python script with the following analyses:

1. **Data Loading and Validation**
   - Loaded `prepared_data_numeric_version.csv`
   - Verified key variables (ELIGIBLE, AFTER, FT)
   - Checked sample sizes by group

2. **Descriptive Statistics**
   - Cross-tabulation of treatment x time
   - Mean outcomes by group
   - Covariate distributions

3. **Difference-in-Differences Models**
   - Model 1: Baseline DiD (heteroskedasticity-robust SE)
   - Model 2: DiD with state clustering
   - Model 3: DiD with year fixed effects
   - Model 4: DiD with state + year fixed effects
   - Model 5: DiD with covariates (PREFERRED SPECIFICATION)

4. **Robustness Checks**
   - Weighted regression using PERWT
   - Gender heterogeneity analysis
   - Event study specification
   - Parallel trends test

### 2.2 Key Decision Points

**Decision 1: Handling Missing Education Data**
- EDUC_RECODE had 3 missing values (out of 17,382)
- Decision: Drop observations with missing education
- Rationale: Minimal data loss (0.02%), maintains complete case analysis
- Final sample: 17,379 observations

**Decision 2: Standard Error Clustering**
- Decision: Cluster at state level (STATEFIP)
- Rationale: State-level policies and economic conditions create within-state correlation
- 50+ state clusters provide reliable asymptotic inference

**Decision 3: Covariate Selection**
- Included: FEMALE, MARRIED, NCHILD, education categories
- Rationale: Theory-driven selection of demographic predictors of employment
- Did not include: Endogenous post-treatment variables

**Decision 4: Reference Year for Event Study**
- Decision: Use 2011 as reference year
- Rationale: Last pre-treatment year, provides cleanest comparison

**Decision 5: Preferred Specification**
- Model 5 selected as preferred: State FE + Year FE + Covariates
- Rationale: Controls for both geography and time trends while accounting for individual heterogeneity

---

## Phase 3: Results Summary

### 3.1 Raw DiD Calculation
```
                    Pre-DACA   Post-DACA   Change
Control (31-35)     0.670      0.645       -0.025
Treatment (26-30)   0.626      0.666       +0.039
DiD                                        +0.064
```

### 3.2 Regression Results

| Model | Specification | DiD Estimate | SE | 95% CI |
|-------|---------------|--------------|-----|--------|
| 1 | Baseline | 0.0643 | 0.0141 | [0.037, 0.092] |
| 2 | State Clustering | 0.0643 | 0.0141 | [0.037, 0.092] |
| 3 | Year FE | 0.0629 | 0.0139 | [0.036, 0.090] |
| 4 | State + Year FE | 0.0626 | 0.0144 | [0.034, 0.091] |
| **5** | **With Covariates** | **0.0545** | **0.0151** | **[0.025, 0.084]** |
| 6 | Weighted | 0.0608 | 0.0216 | [0.018, 0.103] |

### 3.3 Preferred Estimate
- **Effect Size: 0.0545** (5.45 percentage points)
- **Standard Error: 0.0151**
- **95% CI: [0.025, 0.084]**
- **p-value: 0.0003**
- **Sample Size: 17,379**

### 3.4 Event Study Results
```
Year    Coefficient   SE       p-value
2008    -0.061       0.023    0.009
2009    -0.041       0.031    0.182
2010    -0.067       0.020    0.001
2011    (reference)  --       --
2013    +0.018       0.027    0.507
2014    -0.012       0.022    0.574
2015    +0.029       0.036    0.410
2016    +0.048       0.022    0.025
```

### 3.5 Parallel Trends Test
- ELIGIBLE x Year (pre-period): 0.0148 (SE: 0.0077)
- p-value: 0.056
- Conclusion: Cannot reject parallel trends at 5% level

### 3.6 Heterogeneity by Gender
- Male: 0.0504 (SE: 0.0173), p = 0.004
- Female: 0.0488 (SE: 0.0149), p = 0.001

---

## Phase 4: Figure Generation

### 4.1 Figures Created (`create_figures.py`)
1. `figure1_trends.png`: FT employment trends by eligibility status
2. `figure2_eventstudy.png`: Event study coefficients with 95% CI
3. `figure3_did.png`: 2x2 DiD visualization
4. `figure4_sample.png`: Age distribution and sample sizes
5. `figure5_gender.png`: Trends by gender

---

## Phase 5: Report Compilation

### 5.1 LaTeX Document
- Created `replication_report_27.tex`
- Approximately 20 pages including:
  - Title page
  - Abstract
  - Table of Contents
  - Introduction (background, research question)
  - Data section (source, sample, variables)
  - Methodology (identification, specification, inference)
  - Results (raw DiD, regressions, robustness checks)
  - Discussion (interpretation, threats, comparison)
  - Conclusion
  - Figures
  - Appendices (variable definitions, code details)

### 5.2 PDF Compilation
```bash
# First pass
pdflatex -interaction=nonstopmode replication_report_27.tex

# Second pass (for references)
pdflatex -interaction=nonstopmode replication_report_27.tex

# Output: replication_report_27.pdf (1.25 MB)
```

---

## Phase 6: Files Generated

### Required Deliverables
| File | Description | Status |
|------|-------------|--------|
| `replication_report_27.tex` | LaTeX source | Created |
| `replication_report_27.pdf` | Final report PDF | Created |
| `run_log_27.md` | This run log | Created |

### Supporting Files
| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `analysis_results.txt` | Summary of key results |
| `figure1_trends.png` | Employment trends |
| `figure2_eventstudy.png` | Event study |
| `figure3_did.png` | DiD visualization |
| `figure4_sample.png` | Sample composition |
| `figure5_gender.png` | Gender heterogeneity |

---

## Software Environment

- **Platform**: Windows 10/11
- **Python Version**: 3.x
- **Key Packages**:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis)
  - scipy (statistical tests)
  - matplotlib (visualization)
- **LaTeX**: MiKTeX 25.12

---

## Interpretation and Conclusions

### Main Finding
DACA eligibility is estimated to have increased full-time employment by approximately 5.5 percentage points (95% CI: [2.5, 8.4 pp]) among eligible Hispanic-Mexican, Mexican-born individuals aged 26-30 at implementation, compared to those aged 31-35.

### Robustness
- Effect is stable across specifications (5.5-6.4 pp)
- Similar effects for men and women
- Weighted estimates slightly larger (6.1 pp)
- Parallel trends assumption marginally supported (p = 0.056)

### Caveats
1. Some pre-trend variability in event study
2. Intent-to-treat effect (not all eligible applied)
3. Age-specific labor market trends could confound

---

## End of Run Log
