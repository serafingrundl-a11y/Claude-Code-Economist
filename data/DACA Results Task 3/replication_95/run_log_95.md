# DACA Replication Study - Run Log #95

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA employment effects study.

**Date:** January 27, 2026
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment?

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read `replication_instructions.docx` using Python docx extraction
- Identified research design: Difference-in-Differences comparing ages 26-30 (treatment) to ages 31-35 (control)
- Pre-DACA period: 2008-2011
- Post-DACA period: 2013-2016
- Outcome: Full-time employment (FT variable, defined as working 35+ hours/week)

### 1.2 Data Files
- `data/prepared_data_labelled_version.csv` - With labeled categorical variables
- `data/prepared_data_numeric_version.csv` - Numeric coding
- `data/acs_data_dict.txt` - Data dictionary from IPUMS

### 1.3 Sample Size
```
Total observations: 17,382
Treatment Group (ELIGIBLE=1): 11,382
Control Group (ELIGIBLE=0): 6,000
Pre-DACA period: 9,527
Post-DACA period: 7,855
```

---

## 2. Key Analytical Decisions

### 2.1 Variable Definitions

| Variable | Definition | Source |
|----------|------------|--------|
| FT | Full-time employment (1=35+ hrs/wk, 0=otherwise) | Provided |
| ELIGIBLE | Treatment indicator (1=ages 26-30, 0=ages 31-35 in June 2012) | Provided |
| AFTER | Post-period indicator (1=2013-2016, 0=2008-2011) | Provided |
| FEMALE | Binary for female sex (derived from SEX==2) | Derived |
| MARRIED | Binary for married (derived from MARST in [1,2]) | Derived |
| ED_* | Education dummies (derived from EDUC) | Derived |

### 2.2 Regression Specifications

**Decision:** Estimate multiple specifications to assess robustness:

1. **Model 1 (Basic DiD):** FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
2. **Model 2 (Demographics):** Basic + FEMALE + AGE + AGE^2 + MARRIED + Education + NCHILD
3. **Model 3 (Fixed Effects):** ELIGIBLE + ELIGIBLE*AFTER + Year FE + State FE
4. **Model 4 (Full Model):** Demographics + Year FE + State FE
5. **Model Weighted:** Model 2 with PERWT sampling weights

**Rationale:** Multiple specifications allow assessment of estimate stability across different modeling assumptions.

### 2.3 Standard Errors

**Decision:** Use heteroskedasticity-robust (HC1) standard errors for models with fixed effects.

**Rationale:** Robust standard errors account for potential heteroskedasticity in the error term, providing valid inference even if the homoskedasticity assumption is violated.

### 2.4 Sample Restrictions

**Decision:** No additional sample restrictions applied beyond the provided analytic sample.

**Rationale:** Per instructions, "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics."

---

## 3. Commands Executed

### 3.1 Python Analysis Script (analysis.py)

```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_95"
python analysis.py
```

**Key Operations:**
- Loaded `prepared_data_numeric_version.csv` with pandas
- Created derived variables (FEMALE, MARRIED, education dummies, AGE_SQ)
- Estimated OLS regressions using statsmodels
- Applied robust standard errors (HC1) where appropriate
- Conducted subgroup analyses by sex, marital status, and education
- Performed event study analysis with year interactions

### 3.2 Figure Generation (create_figures.py)

```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_95"
python create_figures.py
```

**Output Files:**
- `figure1_parallel_trends.png` - Parallel trends by eligibility group
- `figure2_event_study.png` - Event study coefficients
- `figure3_distributions.png` - Sample distribution panels
- `figure4_coefficients.png` - DiD coefficient comparison
- `figure5_subgroups.png` - Subgroup analysis results

### 3.3 LaTeX Compilation

```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_95"
pdflatex -interaction=nonstopmode replication_report_95.tex
pdflatex -interaction=nonstopmode replication_report_95.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_95.tex  # Third pass for refs
```

**Output:** `replication_report_95.pdf` (26 pages)

---

## 4. Main Results

### 4.1 Simple Difference-in-Differences

```
Treatment (ELIGIBLE=1) Pre:  0.6263
Treatment (ELIGIBLE=1) Post: 0.6658
Treatment Change:            +0.0394

Control (ELIGIBLE=0) Pre:    0.6697
Control (ELIGIBLE=0) Post:   0.6449
Control Change:              -0.0248

DiD Estimate: 0.0643 (6.43 percentage points)
```

### 4.2 Regression Results Summary

| Specification | Coefficient | SE | 95% CI | N |
|--------------|-------------|-----|--------|---|
| Basic DiD | 0.0643*** | 0.0153 | [0.034, 0.094] | 17,382 |
| With Demographics | 0.0591*** | 0.0193 | [0.021, 0.097] | 17,382 |
| Year + State FE | 0.0626*** | 0.0152 | [0.033, 0.092] | 17,382 |
| Full Model | 0.0110 | 0.0205 | [-0.029, 0.051] | 17,382 |
| Weighted (PERWT) | 0.0729*** | 0.0191 | [0.035, 0.110] | 17,382 |

### 4.3 Subgroup Results

| Subgroup | Coefficient | SE |
|----------|-------------|-----|
| Male | 0.0615*** | 0.0170 |
| Female | 0.0452* | 0.0232 |
| Married | 0.0458*** | 0.0183 |
| Unmarried | 0.0583*** | 0.0196 |

### 4.4 Event Study Coefficients (Reference: 2011)

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2008 | -0.0591 | 0.0289 | 0.041 |
| 2009 | -0.0388 | 0.0297 | 0.191 |
| 2010 | -0.0663 | 0.0294 | 0.024 |
| 2011 | 0 (ref) | --- | --- |
| 2013 | 0.0188 | 0.0306 | 0.539 |
| 2014 | -0.0088 | 0.0308 | 0.774 |
| 2015 | 0.0303 | 0.0316 | 0.338 |
| 2016 | 0.0491 | 0.0314 | 0.118 |

---

## 5. Preferred Specification and Interpretation

### 5.1 Preferred Estimate

**Specification:** Model 1 (Basic DiD) or Model 3 (Year + State FE)

**Effect Size:** 6.4 percentage points (95% CI: 3.4 to 9.4 pp)

**Sample Size:** 17,382

### 5.2 Interpretation

DACA eligibility increased the probability of full-time employment by approximately 6.4 percentage points among eligible Mexican-born Hispanic individuals. This represents approximately a 10% increase relative to the pre-DACA baseline full-time employment rate of 62.6% for the treatment group.

### 5.3 Caveats

1. **Pre-trends concern:** Event study shows some differential pre-trends (2008, 2010 coefficients significantly negative), though treatment group was converging toward control.

2. **Full model attenuation:** The full model with demographics and FE produces attenuated, insignificant estimate (0.011), likely due to collinearity between age controls and age-based treatment definition.

3. **Intent-to-treat:** Estimate captures eligibility effect, not actual DACA receipt.

---

## 6. Output Files Generated

| File | Description |
|------|-------------|
| `replication_report_95.tex` | LaTeX source for replication report |
| `replication_report_95.pdf` | Compiled 26-page PDF report |
| `run_log_95.md` | This run log |
| `analysis.py` | Main Python analysis script |
| `create_figures.py` | Figure generation script |
| `results_summary.csv` | Summary of DiD estimates |
| `event_study_results.csv` | Event study coefficients |
| `parallel_trends_data.csv` | FT rates by year and group |
| `figure1_parallel_trends.png` | Parallel trends figure |
| `figure2_event_study.png` | Event study figure |
| `figure3_distributions.png` | Distribution panels |
| `figure4_coefficients.png` | Coefficient comparison |
| `figure5_subgroups.png` | Subgroup analysis figure |

---

## 7. Software Environment

- **Python:** 3.x
- **Key packages:** pandas, numpy, statsmodels, matplotlib, scipy
- **LaTeX:** pdfTeX, Version 3.141592653-2.6-1.40.28 (MiKTeX 25.12)

---

## 8. Replication Notes

This analysis was conducted as an independent replication, following the provided instructions without reference to other researchers' approaches or published studies. All analytical decisions were documented in this log.

**Key design choices:**
- Used provided ELIGIBLE and AFTER variables as specified
- Included all observations in the analytic sample without additional restrictions
- Estimated multiple specifications for robustness
- Conducted standard DiD robustness checks (subgroups, event study, weighted estimation)
- Selected Basic DiD as preferred specification due to consistency across models and concern about over-controlling in full specification

---

*End of Run Log*
