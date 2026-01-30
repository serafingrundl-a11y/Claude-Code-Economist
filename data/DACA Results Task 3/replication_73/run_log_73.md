# Run Log - Replication 73

## DACA Impact on Full-Time Employment Replication Study

### Session Start
Date: 2026-01-27

---

## 1. Initial Setup and Data Understanding

### 1.1 Read Replication Instructions
- Extracted content from `replication_instructions.docx`
- Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US, what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?
- Treatment Group: Ages 26-30 at time of DACA implementation (June 15, 2012)
- Control Group: Ages 31-35 at time of DACA implementation
- Method: Difference-in-Differences (DiD)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded due to implementation timing)

### 1.2 Data Files Identified
- `prepared_data_labelled_version.csv` - Data with labeled categories
- `prepared_data_numeric_version.csv` - Numeric version for analysis
- `acs_data_dict.txt` - Data dictionary from IPUMS

### 1.3 Key Variables (from instructions)
- **ELIGIBLE**: 1 = eligible for DACA (treatment group), 0 = comparison group
- **FT**: 1 = full-time work (35+ hrs/week), 0 = not full-time
- **AFTER**: 1 = years 2013-2016, 0 = years 2008-2011
- **PERWT**: Person weight for population-representative estimates

### Key Decisions Made:
1. Use ELIGIBLE variable as provided (do not create own eligibility variable)
2. Keep those not in labor force in analysis (as FT=0)
3. Use difference-in-differences approach comparing treated (26-30) vs control (31-35)
4. Use person weights (PERWT) for population-representative estimates

---

## 2. Data Exploration

### 2.1 Loading Data
```
Command: python analysis.py
```

**Dataset characteristics:**
- Shape: 17,382 observations x 105 variables
- Years covered: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016
- 2012 confirmed excluded from data

### 2.2 Sample Distribution
| Group     | Pre-DACA (0) | Post-DACA (1) | Total  |
|-----------|--------------|---------------|--------|
| Control   | 3,294        | 2,706         | 6,000  |
| Treatment | 6,233        | 5,149         | 11,382 |
| **Total** | **9,527**    | **7,855**     | **17,382** |

### 2.3 Key Variable Summary
- ELIGIBLE: Treatment = 11,382; Control = 6,000
- AFTER: Pre = 9,527; Post = 7,855
- FT (Full-time): Yes = 11,283; No = 6,099
- PERWT: Range [2, 1176], Mean = 139.01

---

## 3. Analysis Decisions

### 3.1 Model Specification
**Decision:** Use Weighted Least Squares (WLS) with person weights (PERWT)
- **Rationale:** Person weights make results representative of the target population

### 3.2 Standard Error Calculation
**Decision:** Use heteroskedasticity-robust (HC1) standard errors
- **Rationale:** Binary outcome variable (FT) causes heteroskedasticity; HC1 is a conservative choice

### 3.3 Covariates Selection
**Model 1 (Basic):** FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER
**Model 2 (Demographics):** + MALE + MARRIED + AGE + Year FE
**Model 3 (Full):** + Education FE + NCHILD + State FE
- **Rationale:** Including controls improves precision without biasing the DiD estimate (assuming parallel trends hold)

### 3.4 Parallel Trends Test
**Decision:** Event study specification with 2011 as reference year
- **Rationale:** Pre-treatment coefficients should be statistically insignificant if parallel trends hold

---

## 4. Results Summary

### 4.1 Descriptive Statistics (Weighted)
| Group             | Pre-DACA | Post-DACA | Difference |
|-------------------|----------|-----------|------------|
| Control (31-35)   | 0.6886   | 0.6629    | -0.0257    |
| Treatment (26-30) | 0.6369   | 0.6860    | +0.0491    |
| **DiD Estimate**  |          |           | **+0.0748**|

### 4.2 Regression Results
| Model | DiD Estimate | Std. Error | 95% CI | p-value | R² |
|-------|--------------|------------|--------|---------|-----|
| Basic | 0.0748 | 0.0181 | [0.039, 0.110] | <0.0001 | 0.003 |
| Demographics | 0.0624 | 0.0167 | [0.030, 0.095] | 0.0002 | 0.130 |
| Full | 0.0612 | 0.0166 | [0.029, 0.094] | 0.0002 | 0.139 |

### 4.3 Event Study Results
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2008 | -0.0681 | 0.0351 | 0.052 |
| 2009 | -0.0499 | 0.0359 | 0.164 |
| 2010 | -0.0821 | 0.0357 | 0.021 |
| 2011 | 0 (ref) | - | - |
| 2013 | +0.0158 | 0.0375 | 0.674 |
| 2014 | +0.0000 | 0.0384 | 1.000 |
| 2015 | +0.0014 | 0.0381 | 0.970 |
| 2016 | +0.0741 | 0.0384 | 0.053 |

**Note:** Pre-treatment coefficients are marginally significant in some years, suggesting some concern about parallel trends assumption.

### 4.4 Robustness Checks
| Check | DiD Estimate | SE |
|-------|--------------|-----|
| Unweighted | 0.0643 | 0.0153 |
| Males Only | 0.0716 | 0.0199 |
| Females Only | 0.0527 | 0.0281 |
| Clustered SE (State) | 0.0748 | 0.0203 |

---

## 5. Preferred Estimate

**Selected Model:** Model 3 (Full model with controls)

**Effect Size:** 0.0612 (6.12 percentage points)
**Standard Error:** 0.0166
**95% CI:** [0.0286, 0.0938]
**p-value:** 0.0002
**Sample Size:** 17,379

---

## 6. Files Generated

1. `analysis.py` - Main analysis script
2. `regression_results.csv` - Regression coefficients
3. `event_study_results.csv` - Event study coefficients
4. `summary_statistics.csv` - Descriptive statistics
5. `figure1_ft_rates.png` - Full-time employment trends
6. `figure2_event_study.png` - Event study plot
7. `figure3_did_visualization.png` - DiD visualization
8. `model1_summary.txt` - Basic model summary
9. `model3_summary.txt` - Full model summary
10. `replication_report_73.tex` - LaTeX report
11. `replication_report_73.pdf` - Final PDF report

---

## 7. Interpretation

The difference-in-differences analysis suggests that DACA eligibility increased the probability of full-time employment by approximately 6.1 percentage points for eligible individuals aged 26-30 compared to the comparison group (aged 31-35). This effect is statistically significant at conventional levels (p < 0.001). The estimate is robust to the inclusion of demographic and geographic controls, as well as alternative specifications.

However, the event study analysis reveals some concerns about the parallel trends assumption, with marginally significant pre-treatment coefficients in 2008 and 2010. This suggests caution in interpreting the results as purely causal.

---

## Session End
Analysis completed successfully.
