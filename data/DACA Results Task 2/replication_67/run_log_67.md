# Replication Run Log - ID 67

## Overview

This document logs all commands and key decisions made during the replication study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

---

## Date: 2026-01-26

---

## 1. Initial Setup and Data Exploration

### 1.1 Read Replication Instructions

Extracted text from `replication_instructions.docx` using Python's `python-docx` library:

```python
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]
```

**Key research question identified:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on full-time employment (usually working 35+ hours per week)?

### 1.2 Data Files Identified

- `data/data.csv` - Main ACS data file (~33.8 million rows)
- `data/acs_data_dict.txt` - Data dictionary for ACS variables
- `data/state_demo_policy.csv` - Optional state-level supplemental data (not used)

### 1.3 Initial Data Exploration

```bash
head -5 data.csv && wc -l data.csv
```

Output: 33,851,425 rows including header

Explored column structure using Python:
```python
import pandas as pd
df = pd.read_csv('data/data.csv', nrows=1000)
print(df.columns)
print(df.dtypes)
```

54 columns identified including key variables: YEAR, PERWT, AGE, BIRTHYR, BIRTHQTR, HISPAN, BPL, CITIZEN, YRIMMIG, UHRSWORK, etc.

---

## 2. Key Methodological Decisions

### 2.1 Sample Selection Criteria

**Decision:** Filter to individuals meeting DACA eligibility criteria (excluding age)

1. **Hispanic-Mexican ethnicity:** HISPAN = 1
   - Rationale: Research task specifies "ethnically Hispanic-Mexican"

2. **Born in Mexico:** BPL = 200
   - Rationale: Research task specifies "Mexican-born"

3. **Non-citizen:** CITIZEN = 3
   - Rationale: DACA targets undocumented immigrants; cannot distinguish documented vs undocumented, so assume non-citizens are potentially DACA-eligible per instructions

4. **Continuous residence:** YRIMMIG <= 2007
   - Rationale: DACA required continuous US residence since June 15, 2007

5. **Arrived before age 16:** YRIMMIG - BIRTHYR < 16
   - Rationale: DACA required arrival before 16th birthday

### 2.2 Treatment and Control Groups

**Decision:** Use age-based assignment

- **Treatment group:** Ages 26-30 as of June 15, 2012 (DACA-eligible by age)
- **Control group:** Ages 31-35 as of June 15, 2012 (would be eligible except for age)

**Age calculation:**
```python
age_june2012 = 2012 - BIRTHYR
# Adjust for those born Jul-Dec (haven't had birthday by June 15)
if BIRTHQTR in [3, 4]:
    age_june2012 -= 1
```

Rationale: The age cutoff of 31 as of June 15, 2012 provides a natural experiment. Those just under the cutoff are eligible; those just over are not.

### 2.3 Time Periods

**Decision:**
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Exclude 2012

**Rationale:** DACA was implemented on June 15, 2012. The ACS does not record interview month, so 2012 observations cannot be classified as pre or post. Excluding 2012 avoids contamination.

### 2.4 Outcome Variable

**Decision:** Full-time employment = UHRSWORK >= 35

**Rationale:** Standard definition of full-time employment per instructions ("usually working 35 hours per week or more").

### 2.5 Weighting

**Decision:** Use person-level weights (PERWT) for main specifications

**Rationale:** ACS uses complex survey design; weights needed for nationally representative estimates.

---

## 3. Analysis Commands

### 3.1 Main Analysis Script

Created `analysis.py` containing:

1. Data loading with chunked reading for memory efficiency
2. Sample filtering per criteria above
3. Variable construction (age at June 2012, treatment group, full-time employment)
4. Descriptive statistics
5. Simple DiD calculation
6. Regression-based DiD (unweighted and weighted)
7. DiD with covariates (sex, marital status, education)
8. DiD with state fixed effects
9. DiD with robust standard errors
10. Event study analysis
11. Robustness checks (subgroup by sex, alternative age windows, placebo test)
12. Figure generation

### 3.2 Run Analysis

```bash
python analysis.py
```

**Output files generated:**
- `results_summary.txt` - Numerical results
- `figure1_trends.png` - Employment trends over time
- `figure2_eventstudy.png` - Event study coefficients
- `figure3_did.png` - DiD illustration

---

## 4. Sample Sizes Through Filtering

| Filter Step | N |
|-------------|---|
| Hispanic-Mexican, born in Mexico | 991,261 |
| Non-citizens only | 701,347 |
| YRIMMIG <= 2007 | 654,693 |
| Arrived before age 16 | 195,023 |
| Ages 26-35 as of June 2012 | 47,418 |
| Excluding 2012 | 43,238 |

Final sample: **43,238 observations**
- Treatment group: 25,470
- Control group: 17,768

---

## 5. Main Results

### 5.1 Preferred Estimate (Model 2: Weighted Basic DiD)

| Parameter | Value |
|-----------|-------|
| DiD Coefficient | 0.0590 |
| Standard Error | 0.0098 |
| t-statistic | 6.03 |
| p-value | < 0.0001 |
| 95% CI Lower | 0.0398 |
| 95% CI Upper | 0.0782 |

**Interpretation:** DACA eligibility increased full-time employment by 5.9 percentage points.

### 5.2 Alternative Specifications

| Specification | DiD Estimate | SE |
|---------------|--------------|-----|
| Unweighted | 0.052 | 0.010 |
| Weighted (preferred) | 0.059 | 0.010 |
| + Covariates | 0.046 | 0.009 |
| + State FE | 0.048 | 0.009 |
| Robust SE | 0.048 | 0.011 |

### 5.3 Event Study Coefficients (relative to 2011)

| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.008 | 0.020 |
| 2007 | -0.044 | 0.020 |
| 2008 | -0.002 | 0.020 |
| 2009 | -0.014 | 0.020 |
| 2010 | -0.020 | 0.020 |
| 2013 | 0.038 | 0.021 |
| 2014 | 0.043 | 0.021 |
| 2015 | 0.023 | 0.022 |
| 2016 | 0.068 | 0.022 |

Pre-period coefficients are close to zero, supporting parallel trends assumption.

### 5.4 Robustness Checks

- **By sex:** Men = 0.046 (0.011), Women = 0.047 (0.015) - similar effects
- **Narrower age window (27-29 vs 32-34):** 0.054 - consistent with main estimate
- **Placebo test (2009-2011 vs 2006-2008):** 0.006 (0.012) - not significant, as expected

---

## 6. Report Generation

### 6.1 Create LaTeX Report

Created `replication_report_67.tex` containing:
- Abstract
- Introduction
- Background and Identification Strategy
- Data and Sample Construction
- Empirical Methodology
- Results
- Discussion
- Conclusion
- Appendices (additional tables, figures, variable definitions, detailed methods)

### 6.2 Compile PDF

```bash
pdflatex -interaction=nonstopmode replication_report_67.tex
pdflatex -interaction=nonstopmode replication_report_67.tex
pdflatex -interaction=nonstopmode replication_report_67.tex
```

Three passes to resolve cross-references.

**Output:** `replication_report_67.pdf` (23 pages)

---

## 7. Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| `replication_report_67.tex` | LaTeX source | Complete |
| `replication_report_67.pdf` | Final report (23 pages) | Complete |
| `run_log_67.md` | This log file | Complete |
| `analysis.py` | Python analysis script | Complete |
| `results_summary.txt` | Numerical results | Complete |
| `figure1_trends.png` | Employment trends figure | Complete |
| `figure2_eventstudy.png` | Event study figure | Complete |
| `figure3_did.png` | DiD illustration figure | Complete |

---

## 8. Key Decisions Summary

1. **Sample definition:** Used HISPAN=1, BPL=200, CITIZEN=3, YRIMMIG<=2007, arrived before 16
2. **Age groups:** 26-30 (treatment) vs 31-35 (control) as of June 15, 2012
3. **Time periods:** Pre = 2006-2011, Post = 2013-2016, excluded 2012
4. **Outcome:** Full-time = UHRSWORK >= 35
5. **Estimation:** Weighted least squares with person weights
6. **Preferred estimate:** Basic weighted DiD = 0.059 (SE = 0.010)
7. **Robustness:** Consistent across specifications (0.046-0.059)
8. **Parallel trends:** Supported by event study (pre-period coefficients near zero)
9. **Placebo test:** Passed (estimate near zero for fake pre-period treatment)

---

## End of Log
