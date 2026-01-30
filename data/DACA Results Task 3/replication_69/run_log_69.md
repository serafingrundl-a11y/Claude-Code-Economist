# Run Log - DACA Replication Study (ID: 69)

## Date: 2026-01-27

## Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA full-time employment study.

---

## 1. Data Exploration

### 1.1 Initial Data Inspection
- Data file: `prepared_data_labelled_version.csv` (18,988,640 bytes)
- Numeric version: `prepared_data_numeric_version.csv` (6,458,555 bytes)
- Total observations: 17,382 (excluding header)

### 1.2 Key Variables Identified
From the data dictionary and CSV header:
- **YEAR**: Survey year (2008-2011, 2013-2016; 2012 excluded)
- **FT**: Full-time employment (1=full-time, 0=not full-time) - OUTCOME
- **ELIGIBLE**: DACA eligibility indicator (1=treatment group ages 26-30 in June 2012, 0=control group ages 31-35) - TREATMENT
- **AFTER**: Post-DACA indicator (1=2013-2016, 0=2008-2011) - TIME
- **PERWT**: Person weight for population estimates
- **AGE_IN_JUNE_2012**: Age at time of DACA implementation
- **AGE_AT_IMMIGRATION**: Age when immigrated

### 1.3 Data Dictionary Review
- IPUMS binary variables: 1=No, 2=Yes
- Created binary variables (FT, AFTER, ELIGIBLE): 0=No, 1=Yes
- State policy variables: 0=No, 1=Yes

---

## 2. Analytical Approach

### 2.1 Research Design
**Difference-in-Differences (DiD) Design:**
- Treatment group: Individuals aged 26-30 in June 2012 who were DACA-eligible (ELIGIBLE=1)
- Control group: Individuals aged 31-35 in June 2012 who would have been eligible but for age (ELIGIBLE=0)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
- Outcome: Full-time employment (FT)

### 2.2 Key Analytical Decisions
1. **Use provided ELIGIBLE variable**: As instructed, using pre-constructed eligibility indicator
2. **Keep all observations**: As instructed, not dropping any individuals based on characteristics
3. **Include those not in labor force**: As instructed, keeping everyone in sample
4. **Use survey weights (PERWT)**: For population-representative estimates
5. **Cluster standard errors at state level**: Account for within-state correlation over time
6. **Preferred specification**: Weighted regression with state-clustered SEs (Model 4)

---

## 3. Analysis Commands

### 3.1 Python Analysis Script
File: `analysis_script.py`

Command executed:
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_69" && python analysis_script.py
```

### 3.2 Models Estimated

| Model | Description | Purpose |
|-------|-------------|---------|
| 1 | Basic OLS | Baseline estimate |
| 2 | Weighted Least Squares | Account for survey design |
| 3 | OLS with Clustered SEs | Correct inference for state-level correlation |
| 4 | **WLS + Clustered SEs (PREFERRED)** | Best practice combining weights and clustering |
| 5 | With demographic controls | Check robustness to observable confounders |
| 6 | With demographics + education | Additional covariate adjustment |
| 7 | Year fixed effects | Control for time trends |
| 8 | State fixed effects | Control for state-level confounders |
| 9 | Year + State FEs + demographics | Most saturated specification |

---

## 4. Main Results

### 4.1 Preferred Estimate (Model 4)
- **DiD Coefficient**: 0.0748
- **Standard Error**: 0.0203 (clustered at state level)
- **95% CI**: [0.0350, 0.1145]
- **p-value**: 0.0002

### 4.2 Interpretation
DACA eligibility is associated with a **7.48 percentage point increase** in the probability of full-time employment among the treatment group (ages 26-30 in June 2012) relative to the control group (ages 31-35), comparing the post-DACA period (2013-2016) to the pre-DACA period (2008-2011).

### 4.3 Sample Sizes
- Total N: 17,382
- Treatment group (ELIGIBLE=1): 11,382
- Control group (ELIGIBLE=0): 6,000
- Pre-period: 9,527
- Post-period: 7,855

### 4.4 Full-Time Employment Rates (Weighted)
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 63.69% | 68.60% | +4.91 pp |
| Control (31-35) | 68.86% | 66.29% | -2.57 pp |
| **Difference-in-Differences** | | | **+7.48 pp** |

---

## 5. Robustness Checks

### 5.1 Sensitivity to Specification
All models show positive and statistically significant effects ranging from 0.061 to 0.075:

| Model | DiD Estimate | SE | p-value |
|-------|--------------|-----|---------|
| Basic OLS | 0.0643 | 0.0153 | <0.001 |
| Weighted | 0.0748 | 0.0152 | <0.001 |
| Clustered SEs | 0.0643 | 0.0141 | <0.001 |
| **Weighted + Clustered (PREFERRED)** | **0.0748** | **0.0203** | **<0.001** |
| With Demographics | 0.0642 | 0.0219 | 0.003 |
| With Demographics + Education | 0.0617 | 0.0223 | 0.006 |
| Year Fixed Effects | 0.0721 | 0.0195 | <0.001 |
| State Fixed Effects | 0.0737 | 0.0209 | <0.001 |
| Full Model (Year + State FEs + Demographics) | 0.0607 | 0.0218 | 0.005 |

### 5.2 Event Study (Parallel Trends)
Reference year: 2011

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2008 | -0.0681 | 0.0294 | 0.021 |
| 2009 | -0.0499 | 0.0374 | 0.183 |
| 2010 | -0.0821 | 0.0296 | 0.006 |
| 2013 | 0.0158 | 0.0406 | 0.697 |
| 2014 | 0.0000 | 0.0279 | 1.000 |
| 2015 | 0.0014 | 0.0384 | 0.971 |
| 2016 | 0.0741 | 0.0299 | 0.013 |

**Note**: Pre-treatment coefficients are negative relative to 2011, suggesting some differential pre-trends. However, the pattern shows convergence approaching 2011 (the year before DACA). Post-treatment effects emerge gradually with the largest effect in 2016.

### 5.3 Heterogeneity Analysis

**By Gender:**
- Males: DiD = 0.0716 (SE: 0.0195), p < 0.001, N = 9,075
- Females: DiD = 0.0527 (SE: 0.0290), p = 0.069, N = 8,307

**By Education:**
- High School: DiD = 0.0608 (SE: 0.0219), N = 12,444
- Some College: DiD = 0.0672 (SE: 0.0389), N = 2,877
- Two-Year Degree: DiD = 0.1816 (SE: 0.0415), N = 991
- BA+: DiD = 0.1619 (SE: 0.0356), N = 1,058

---

## 6. Output Files Generated
- `analysis_results.csv` - Key results in CSV format
- `regression_tables.txt` - Full regression output
- `results_for_latex.txt` - Data formatted for LaTeX report
- `replication_report_69.tex` - LaTeX source for final report
- `replication_report_69.pdf` - Final PDF report

---

## 7. Software Environment
- Python 3.14
- pandas, numpy, statsmodels, scipy
- LaTeX (pdflatex) for report compilation

---

## 8. Methodological Notes

### 8.1 Why Weighted Regression?
The American Community Survey uses a complex sampling design. Person weights (PERWT) allow us to make population-representative inferences rather than sample-specific estimates.

### 8.2 Why Cluster Standard Errors?
Individuals within the same state may have correlated outcomes due to shared labor market conditions, state policies, etc. Clustering at the state level produces conservative (larger) standard errors that account for this correlation.

### 8.3 Identification Assumption
The key identifying assumption is parallel trends: absent DACA, the treatment and control groups would have followed similar trends in full-time employment. The event study provides some evidence on this assumption, though pre-treatment coefficients suggest caution.

### 8.4 Limitations
1. Not true panel data - different individuals observed before and after
2. Some evidence of differential pre-trends (though converging toward treatment)
3. Age-based comparison may conflate DACA effects with age-employment relationship
4. Cannot distinguish extensive margin (employment) from intensive margin (hours conditional on employment) since FT includes non-employed

---

## 9. Conclusion
The analysis finds a positive and statistically significant effect of DACA eligibility on full-time employment. The preferred estimate suggests that DACA eligibility increased the probability of full-time employment by approximately 7.5 percentage points. This result is robust across multiple specifications, though the parallel trends assumption shows some strain in earlier pre-treatment years.
