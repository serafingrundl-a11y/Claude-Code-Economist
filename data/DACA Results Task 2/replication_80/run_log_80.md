# Run Log - DACA Replication Analysis (ID: 80)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects analysis.

## Date: 2026-01-26

---

## 1. Data Exploration

### 1.1 Reading Replication Instructions
- Read `replication_instructions.docx` using Python's `python-docx` library
- Key research question: Effect of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals
- Treatment group: Ages 26-30 at time of policy (June 15, 2012)
- Control group: Ages 31-35 at time of policy (June 15, 2012)
- Outcome: Full-time employment (usually working 35+ hours/week)

### 1.2 Data Files Identified
- `data/data.csv` - Main ACS data file (33,851,425 rows)
- `data/acs_data_dict.txt` - Data dictionary
- `data/state_demo_policy.csv` - Optional state-level data (not used)

### 1.3 Key Variables from Data Dictionary
- `YEAR`: Census year (2006-2016)
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status (1 = Employed)
- `SEX`, `MARST`, `EDUC`, `STATEFIP`: Demographic controls

---

## 2. Sample Construction

### 2.1 Eligibility Criteria
Based on DACA requirements:
1. Hispanic-Mexican ethnicity (`HISPAN == 1`)
2. Born in Mexico (`BPL == 200`)
3. Not a citizen (`CITIZEN == 3`) - proxy for undocumented status
4. Arrived before age 16 (`YRIMMIG - BIRTHYR < 16`)
5. Continuous presence since June 15, 2007 (`YRIMMIG <= 2007`)

### 2.2 Treatment/Control Group Definitions
- **Treatment Group**: Born 1982-1986 (ages 26-30 in 2012)
- **Control Group**: Born 1977-1981 (ages 31-35 in 2012)

### 2.3 Time Period
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016
- Year 2012 excluded due to mid-year DACA implementation (June 15, 2012)

### 2.4 Sample Size Progression
```
Total Hispanic-Mexican Mexico-born observations: 991,261
Non-citizen observations: 701,347
After filtering arrived before age 16: 205,327
After filtering arrived by 2007: 195,023
Analysis sample (treatment + control): 49,019
After excluding 2012: 44,725
```

---

## 3. Analysis Decisions

### 3.1 Outcome Variable
- Full-time employment defined as `UHRSWORK >= 35` (binary indicator)
- This follows the standard BLS definition of full-time work

### 3.2 Difference-in-Differences Specification
Basic model:
```
fulltime = β₀ + β₁(treated) + β₂(post) + β₃(treated × post) + ε
```

### 3.3 Control Variables
- `female`: Binary indicator for sex = 2
- `married`: Binary indicator for marital status ≤ 2
- `educ_hs`: Binary indicator for high school education or more

### 3.4 Standard Errors
- Clustered at the state level (`STATEFIP`) to account for within-state correlation

### 3.5 Fixed Effects
- Year fixed effects (absorbed time trends)
- State fixed effects (absorbed geographic heterogeneity)

---

## 4. Key Results

### 4.1 Main DiD Estimates
| Model | DiD Estimate | Std Error | p-value | 95% CI |
|-------|-------------|-----------|---------|--------|
| Basic DiD | 0.0551 | 0.0098 | <0.001 | [0.036, 0.074] |
| DiD (Clustered SE) | 0.0551 | 0.0065 | <0.001 | [0.042, 0.068] |
| DiD + Demographics | 0.0497 | 0.0086 | <0.001 | [0.033, 0.067] |
| DiD + Year FE | 0.0498 | 0.0088 | <0.001 | [0.032, 0.067] |
| DiD + Year & State FE | 0.0489 | 0.0092 | <0.001 | [0.031, 0.067] |

### 4.2 Preferred Estimate (Model 4)
- **Effect size**: 0.0489 (4.89 percentage points)
- **Standard error**: 0.0092
- **95% CI**: [0.031, 0.067]
- **p-value**: <0.0001
- **Sample size**: 44,725

### 4.3 Manual DiD Calculation
| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Treatment (26-30) | 0.611 | 0.634 | +0.023 |
| Control (31-35) | 0.643 | 0.611 | -0.032 |
| **DiD** | | | **0.055** |

### 4.4 Event Study Results
Pre-treatment coefficients (relative to 2011):
- 2006: -0.035 (p=0.034)
- 2007: -0.023 (p=0.174)
- 2008: -0.003 (p=0.841)
- 2009: -0.001 (p=0.950)
- 2010: -0.011 (p=0.440)

Post-treatment coefficients:
- 2013: 0.030 (p=0.015)
- 2014: 0.040 (p=0.002)
- 2015: 0.039 (p=0.047)
- 2016: 0.064 (p=0.001)

### 4.5 Robustness Checks
- **Any employment** (alternative outcome): DiD = 0.0404 (SE = 0.004, p<0.001)
- **Males only**: DiD = 0.0464 (SE = 0.0107, p<0.001), N = 25,058
- **Females only**: DiD = 0.0444 (SE = 0.0096, p<0.001), N = 19,667

---

## 5. Commands Executed

```bash
# Initial data exploration
head -5 data/data.csv
wc -l data/data.csv

# Run main analysis
python analysis.py
```

---

## 6. Output Files Generated

1. `analysis.py` - Main analysis script
2. `data/filtered_data.csv` - Filtered Hispanic-Mexican Mexico-born sample
3. `analysis_sample.csv` - Final analysis sample
4. `did_results.csv` - Summary of DiD estimates
5. `event_study_results.csv` - Event study coefficients
6. `summary_stats.csv` - Summary statistics by group
7. `preferred_estimate.txt` - Preferred estimate details
8. `replication_report_80.tex` - LaTeX report
9. `replication_report_80.pdf` - PDF report

---

## 7. Interpretation

The difference-in-differences analysis finds that DACA eligibility increased full-time employment by approximately 4.9 percentage points (95% CI: 3.1 to 6.7 pp) among the treatment group relative to the control group. This effect is statistically significant at the 1% level and robust to the inclusion of demographic controls, year fixed effects, and state fixed effects.

The event study analysis provides support for the parallel trends assumption: pre-treatment coefficients are generally small and statistically insignificant (with the exception of 2006), while post-treatment coefficients show a clear positive and increasing pattern. The effect appears to grow over time, reaching 6.4 percentage points by 2016.

The effect is similar for males (4.6 pp) and females (4.4 pp), suggesting DACA's employment effects were broadly distributed across genders.

---

## 8. Limitations and Notes

1. Cannot distinguish documented from undocumented non-citizens; assume all non-citizens without papers are undocumented
2. ACS does not identify the month of data collection, so 2012 observations before/after DACA cannot be distinguished
3. The 2006 pre-treatment coefficient is marginally significant, which may indicate some violation of parallel trends, though other pre-treatment years show no significant differences
4. Results use clustered standard errors at state level to account for within-state correlation
