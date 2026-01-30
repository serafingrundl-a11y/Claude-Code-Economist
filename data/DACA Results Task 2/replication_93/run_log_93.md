# Run Log - DACA Replication Study (ID: 93)

## Date: January 26, 2026

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA effect on full-time employment study.

---

## 1. Data Exploration

### 1.1 Initial Data Assessment
- **Data file**: `data/data.csv` (6.3 GB, 33,851,424 observations)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Years covered**: 2006-2016 (ACS 1-year files)

### 1.2 Key Variables Identified
From data dictionary review:
- YEAR: Survey year
- PERWT: Person weight for population estimates
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1-4)
- UHRSWORK: Usual hours worked per week
- SEX, MARST, EDUC, STATEFIP, METRO, NCHILD, FAMSIZE: Demographic controls

---

## 2. Sample Construction Decisions

### 2.1 Eligibility Criteria Applied
1. **Hispanic-Mexican ethnicity**: HISPAN == 1
2. **Born in Mexico**: BPL == 200
3. **Non-citizen**: CITIZEN == 3 (proxy for undocumented status)
4. **Arrived before age 16**: YRIMMIG - BIRTHYR < 16
5. **Continuous residence since June 2007**: YRIMMIG <= 2007
6. **Age groups**: 26-30 (treatment) or 31-35 (control) on June 15, 2012
7. **Exclude 2012**: Partial treatment year

### 2.2 Sample Sizes at Each Step
| Filter | Observations | Dropped |
|--------|-------------|---------|
| Initial | 33,851,424 | -- |
| Hispanic-Mexican | 2,945,521 | 30,905,903 |
| Mexico-born | 991,261 | 1,954,260 |
| Non-citizen | 701,347 | 289,914 |
| Arrived before 16 | 205,327 | 496,020 |
| Continuous residence | 195,023 | 10,304 |
| Age 26-35 on June 2012 | 47,418 | 147,605 |
| Exclude 2012 | 43,238 | 4,180 |

### 2.3 Final Sample Composition
- **Treatment group (ages 26-30)**: 25,470 observations
- **Control group (ages 31-35)**: 17,768 observations
- **Pre-period (2006-2011)**: 28,377 observations
- **Post-period (2013-2016)**: 14,861 observations

---

## 3. Variable Construction

### 3.1 Age on June 15, 2012
```python
def age_on_june_15_2012(birthyr, birthqtr):
    base_age = 2012 - birthyr
    if birthqtr in [1, 2]:  # Jan-June: already had birthday
        return base_age
    else:  # July-Dec: not yet had birthday
        return base_age - 1
```

### 3.2 Outcome Variable
- **Full-time employment**: UHRSWORK >= 35

### 3.3 Treatment Indicators
- **treated**: 1 if age_june_2012 in [26, 30], 0 otherwise
- **post**: 1 if YEAR >= 2013, 0 otherwise
- **treated_post**: treated * post (DiD interaction)

### 3.4 Control Variables
- female: SEX == 2
- married: MARST <= 2
- has_children: NCHILD > 0
- metro: METRO >= 2
- educ_lessHS: EDUCD < 62
- educ_HS: EDUCD in [62, 64]
- educ_someCollege: EDUCD in (64, 101)
- educ_BA: EDUCD >= 101

---

## 4. Analysis Commands

### 4.1 Python Script Execution
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_93"
python analysis.py
```

### 4.2 Model Specifications Estimated

**Model 1: Basic DiD**
```
fulltime ~ treated + post + treated_post
```

**Model 2: DiD with demographic controls**
```
fulltime ~ treated + post + treated_post + female + married + has_children + metro + educ_HS + educ_someCollege + educ_BA
```

**Model 3: DiD with year fixed effects**
```
fulltime ~ treated + C(YEAR) + treated_post + [controls]
```

**Model 4: DiD with year and state fixed effects**
```
fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + [controls]
```

**Model 5: Preferred specification (WLS with robust SE)**
```
fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + [controls]
weights = PERWT
cov_type = 'HC3'
```

---

## 5. Key Results

### 5.1 Simple DiD Calculation
| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Treatment (26-30) | 0.6147 | 0.6339 | +0.0192 |
| Control (31-35) | 0.6461 | 0.6136 | -0.0325 |
| **DiD** | | | **+0.0517** |

### 5.2 Regression Results Summary
| Model | DiD Coefficient | SE | p-value |
|-------|----------------|-----|---------|
| 1. Basic | 0.0516 | 0.0100 | <0.001 |
| 2. + Controls | 0.0396 | 0.0092 | <0.001 |
| 3. + Year FE | 0.0388 | 0.0092 | <0.001 |
| 4. + State FE | 0.0380 | 0.0092 | <0.001 |
| 5. Preferred (weighted, robust) | 0.0407 | 0.0107 | <0.001 |

### 5.3 Preferred Estimate
- **Effect**: 4.07 percentage points
- **95% CI**: [1.97, 6.16]
- **p-value**: 0.0001
- **N**: 43,238

### 5.4 Robustness Checks
- **Male effect**: 2.60 pp (SE = 0.0124)
- **Female effect**: 5.23 pp (SE = 0.0181)
- **Pre-trend test**: -0.0007 (p = 0.842) - no differential pre-trend

---

## 6. Output Files Generated

### 6.1 Analysis Outputs
- `parallel_trends.png` - Figure showing employment trends by group
- `regression_results.txt` - Full regression output
- `summary_statistics.csv` - Summary statistics by group/period
- `results_for_report.json` - Key results in machine-readable format

### 6.2 Report Files
- `replication_report_93.tex` - LaTeX source
- `replication_report_93.pdf` - Final 18-page report

### 6.3 Log File
- `run_log_93.md` - This file

---

## 7. Key Analytical Decisions

### 7.1 Decisions Made
1. **Age calculation**: Used birth quarter to determine exact age on June 15, 2012
2. **Control group**: Ages 31-35 (5-year bandwidth matching treatment group)
3. **Proxy for undocumented**: Non-citizens who arrived as children
4. **Continuous residence**: Required immigration by 2007
5. **Outcome**: Full-time = 35+ hours/week (standard BLS definition)
6. **Weights**: Used person weights (PERWT) for preferred specification
7. **Standard errors**: Heteroskedasticity-robust (HC3)

### 7.2 Decisions Not Made (Alternatives Considered)
1. Did not use narrower age bandwidth to maximize sample size
2. Did not include age-specific trends
3. Did not cluster standard errors by state (single cross-section in each year)
4. Did not use regression discontinuity design

---

## 8. Software Environment

### 8.1 Versions
- Python 3.14.2
- pandas 2.3.3
- numpy 2.3.5
- statsmodels 0.14.6
- matplotlib 3.10.8
- seaborn 0.13.2
- pdflatex (MiKTeX)

### 8.2 Operating System
- Windows 10/11 (win32)

---

## 9. Interpretation Summary

DACA eligibility is estimated to have increased full-time employment by approximately 4.07 percentage points among Hispanic-Mexican non-citizens who arrived in the US before age 16 and had been continuously resident since 2007. This represents roughly a 6.6% increase relative to baseline employment rates.

The effect is larger for women (5.23 pp) than men (2.60 pp), possibly reflecting that DACA enabled women to enter the formal labor market more readily than men who may have already been working informally.

Pre-trend tests support the parallel trends assumption (p = 0.84), and the results are robust across multiple specifications with and without demographic controls, year fixed effects, and state fixed effects.

---

## 10. Checklist

- [x] Read replication instructions
- [x] Explored data structure
- [x] Constructed analytic sample
- [x] Defined DACA eligibility criteria
- [x] Created outcome and treatment variables
- [x] Ran difference-in-differences analysis
- [x] Conducted robustness checks
- [x] Generated figures
- [x] Wrote LaTeX report (~20 pages)
- [x] Compiled PDF
- [x] Created run log

---

*End of Run Log*
