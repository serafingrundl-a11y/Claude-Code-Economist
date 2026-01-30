# DACA Replication Study - Run Log

## Replication ID: 41
## Date: January 2026

---

## Overview

This document logs all commands executed and key decisions made during the replication of the DACA employment effects study.

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions

**Command:** Extracted text from `replication_instructions.docx` using Python's docx library.

**Key findings from instructions:**
- Research Question: Effect of DACA eligibility on full-time employment
- Treatment group: Ages 26-30 as of June 15, 2012 (ELIGIBLE = 1)
- Control group: Ages 31-35 as of June 15, 2012 (ELIGIBLE = 0)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded)
- Outcome: FT (full-time employment, 35+ hours/week)
- Sample: Hispanic-Mexican Mexican-born individuals

### 1.2 Data Files Identified

```
data/prepared_data_numeric_version.csv  - Main analysis data
data/prepared_data_labelled_version.csv - Labeled version
data/acs_data_dict.txt                  - Variable documentation
```

### 1.3 Data Structure Verification

**Command:**
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['YEAR'].unique())}")
```

**Output:**
- Total observations: 17,382
- Years: [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
- Confirmed 2012 is excluded
- 105 variables available

---

## 2. Sample Verification

### 2.1 Treatment/Control Distribution

| Group | ELIGIBLE | Pre-Period | Post-Period | Total |
|-------|----------|------------|-------------|-------|
| Control (31-35) | 0 | 3,294 | 2,706 | 6,000 |
| Treatment (26-30) | 1 | 6,233 | 5,149 | 11,382 |
| Total | - | 9,527 | 7,855 | 17,382 |

### 2.2 Age Verification

**Decision:** Verified that ELIGIBLE variable correctly identifies age groups:
- ELIGIBLE=1: Mean age in June 2012 = 28.1 (range: 26.0-30.75)
- ELIGIBLE=0: Mean age in June 2012 = 32.9 (range: 31.0-35.0)

---

## 3. Analysis Decisions

### 3.1 Identification Strategy

**Decision:** Use difference-in-differences (DiD) design exploiting the age-based eligibility cutoff.

**Justification:**
- The age 31 cutoff as of June 15, 2012 is arbitrary
- Individuals just above and below cutoff are otherwise similar
- Allows estimation of causal effect of DACA eligibility

### 3.2 Model Specification

**Decision:** Estimate multiple specifications with progressive controls:

1. Basic DiD (no controls)
2. DiD + demographic controls (sex, age, marital status, children)
3. DiD + demographics + education
4. DiD + demographics + education + year FE
5. DiD + demographics + education + year FE + state FE
6. Same as (5) with robust standard errors
7. Same as (5) with state-clustered standard errors (PREFERRED)

**Justification:**
- Year FE control for common time shocks
- State FE control for time-invariant state characteristics
- Clustered SE account for within-state correlation

### 3.3 Covariates Selected

**Decision:** Include the following covariates:
- SEX (female indicator)
- AGE (continuous)
- MARST (married indicator)
- NCHILD (has children indicator)
- EDUC_RECODE (education category dummies)

**Justification:**
- These are standard demographic controls in labor economics
- May improve precision and balance between groups
- Education may affect both DACA eligibility decisions and employment

### 3.4 Standard Error Calculation

**Decision:** Report clustered standard errors at the state level as preferred specification.

**Justification:**
- Treatment (DACA) operates at the national level but outcomes may be correlated within states
- Conservative approach that accounts for arbitrary within-state correlation
- Standard in policy evaluation literature

### 3.5 Sample Restrictions

**Decision:** Use full provided sample; do not drop any observations.

**Justification:**
- Instructions explicitly state "do not further limit the sample"
- Keep those not in labor force (coded FT=0) per instructions
- ELIGIBLE variable already identifies correct comparison groups

---

## 4. Analysis Execution

### 4.1 Main Analysis Script

**File created:** `analysis.py`

**Key components:**
1. Data loading and verification
2. Descriptive statistics
3. Simple 2x2 DiD calculation
4. Regression-based DiD (multiple specifications)
5. Parallel trends assessment
6. Event study analysis
7. Heterogeneity analysis
8. Robustness checks
9. Figure generation

### 4.2 Script Execution

**Command:**
```bash
python analysis.py
```

**Runtime:** ~30 seconds

**Output files generated:**
- `results_summary.csv` - Table of coefficient estimates
- `model_output.txt` - Full regression output
- `figure1_trends.png` - Employment trends
- `figure2_event_study.png` - Event study plot
- `figure3_robustness.png` - Specification curve

---

## 5. Key Results

### 5.1 Main Estimates

| Specification | DiD Estimate | SE | 95% CI |
|--------------|-------------|-----|---------|
| Basic (no controls) | 0.0643 | 0.0153 | [0.034, 0.094] |
| + Demographics | 0.0549 | 0.0143 | [0.027, 0.083] |
| + Education | 0.0523 | 0.0143 | [0.024, 0.080] |
| + Year FE | 0.0507 | 0.0143 | [0.023, 0.079] |
| + State FE | 0.0508 | 0.0143 | [0.023, 0.079] |
| Robust SE | 0.0508 | 0.0142 | [0.023, 0.079] |
| Clustered SE (PREFERRED) | 0.0508 | 0.0147 | [0.022, 0.080] |

### 5.2 Preferred Estimate

**DiD coefficient:** 0.0508 (5.08 percentage points)
**Standard error:** 0.0147 (clustered at state level)
**95% CI:** [0.022, 0.080]
**p-value:** 0.0005
**Sample size:** 17,382

### 5.3 Parallel Trends

**Pre-trend test:**
- Coefficient on ELIGIBLE x Year (pre-period): 0.0151
- SE: 0.0093
- p-value: 0.103
- Conclusion: No significant differential pre-trend

### 5.4 Placebo Test

**Placebo DiD (2010-2011 vs 2008-2009):**
- Coefficient: 0.0164
- SE: 0.0195
- p-value: 0.40
- Conclusion: Insignificant, supports identification

---

## 6. Report Generation

### 6.1 LaTeX Report

**File created:** `replication_report_41.tex`

**Contents (~23 pages):**
1. Abstract
2. Introduction
3. Background on DACA
4. Data description
5. Empirical methodology
6. Main results
7. Robustness and heterogeneity
8. Discussion
9. Conclusion
10. Appendix (variable definitions, full output, yearly rates)

### 6.2 PDF Compilation

**Commands:**
```bash
pdflatex -interaction=nonstopmode replication_report_41.tex
pdflatex -interaction=nonstopmode replication_report_41.tex
pdflatex -interaction=nonstopmode replication_report_41.tex
```

**Note:** Three passes required for table of contents and cross-references.

**Output:** `replication_report_41.pdf` (23 pages)

---

## 7. Files Produced

| File | Description |
|------|-------------|
| `analysis.py` | Main Python analysis script |
| `results_summary.csv` | Summary of coefficient estimates |
| `model_output.txt` | Full regression output |
| `figure1_trends.png` | Employment trends by group |
| `figure2_event_study.png` | Event study coefficients |
| `figure3_robustness.png` | Robustness across specifications |
| `replication_report_41.tex` | LaTeX source for report |
| `replication_report_41.pdf` | Final PDF report |
| `run_log_41.md` | This run log |

---

## 8. Software and Versions

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- matplotlib (figure generation)
- pdflatex (LaTeX compilation via MiKTeX)

---

## 9. Summary of Analytical Choices

1. **Identification:** DiD using age eligibility cutoff
2. **Treatment group:** Ages 26-30 as of June 2012 (ELIGIBLE=1)
3. **Control group:** Ages 31-35 as of June 2012 (ELIGIBLE=0)
4. **Pre-period:** 2008-2011
5. **Post-period:** 2013-2016
6. **Outcome:** Full-time employment (FT=1 if 35+ hours/week)
7. **Controls:** Sex, age, marital status, children, education, year FE, state FE
8. **Standard errors:** Clustered at state level
9. **Sample:** Full provided sample (17,382 observations)
10. **Weights:** Main results unweighted; weighted results reported as robustness

---

## 10. Interpretation

**Finding:** DACA eligibility increased full-time employment by approximately 5.1 percentage points among Hispanic-Mexican Mexican-born individuals who were 26-30 years old at the time of implementation.

**Interpretation:** This represents an 8.1% increase relative to the pre-treatment mean (62.6%) for the treatment group. The effect is statistically significant at the 1% level and robust across specifications.

**Mechanisms:** The effect likely operates through legal work authorization, access to driver's licenses in some states, and reduced fear of deportation enabling formal employment.

---

*End of Run Log*
