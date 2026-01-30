# Run Log - Replication 68

## Overview
Independent replication analyzing the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of DACA eligibility (treatment) on the probability of full-time employment (35+ hours/week) among eligible individuals?

## Identification Strategy
- **Treatment group**: Ages 26-30 at time of DACA implementation (June 2012), ELIGIBLE=1
- **Control group**: Ages 31-35 at time of DACA implementation, ELIGIBLE=0
- **Design**: Difference-in-differences comparing pre (2008-2011) to post (2013-2016) periods
- **Data**: ACS repeated cross-section (not panel data)

---

## Session Log

### Step 1: Initial Setup and Data Exploration
- Read replication instructions from docx file using python-docx
- Identified data files:
  - `prepared_data_numeric_version.csv` (used for analysis)
  - `prepared_data_labelled_version.csv`
  - `acs_data_dict.txt`
- Key variables identified:
  - `FT`: Full-time employment (1=yes, 0=no) - OUTCOME
  - `ELIGIBLE`: Treatment group indicator (1=ages 26-30, 0=ages 31-35)
  - `AFTER`: Post-treatment indicator (1=2013-2016, 0=2008-2011)
  - `PERWT`: Person weight for weighted analysis
- Note: 2012 data excluded since timing relative to DACA implementation is ambiguous

### Step 2: Data Loading and Initial Checks
- Loaded data using Python/pandas
- Dataset: 17,382 observations, 105 variables
- Years present: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- No missing values in key variables (ELIGIBLE, AFTER, FT, PERWT, YEAR, AGE, SEX)

### Step 3: Sample Verification
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (AFTER=0): 9,527 observations
- Post-period (AFTER=1): 7,855 observations

Sample by Group and Period:
| Group | Pre (2008-2011) | Post (2013-2016) | Total |
|-------|-----------------|------------------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |
| Total | 9,527 | 7,855 | 17,382 |

### Step 4: Descriptive Statistics
Weighted Full-Time Employment Rates:
| Group | Pre-Period | Post-Period | Change |
|-------|------------|-------------|--------|
| Control (31-35) | 0.689 | 0.663 | -0.026 |
| Treatment (26-30) | 0.637 | 0.686 | +0.049 |
| **DiD** | | | **0.075** |

### Step 5: Difference-in-Differences Analysis
Created analysis script (`analysis_script.py`) implementing:
1. Basic OLS (unweighted)
2. Weighted Least Squares (WLS) with PERWT weights
3. WLS with heteroskedasticity-robust (HC1) standard errors
4. Adding demographic controls (FEMALE, MARRIED, HAS_CHILDREN)
5. Adding education controls (EDUC_HS, EDUC_SOMECOLL, EDUC_AA, EDUC_BA)
6. Adding year fixed effects
7. Adding state fixed effects (STATEFIP)

### Step 6: Results Summary

| Model | DiD Estimate | Std. Error | p-value |
|-------|-------------|------------|---------|
| Basic OLS (unweighted) | 0.0643 | 0.0153 | 0.0000 |
| Basic WLS (weighted) | 0.0748 | 0.0152 | 0.0000 |
| WLS with robust SE | 0.0748 | 0.0181 | 0.0000 |
| + Demographics | 0.0642 | 0.0168 | 0.0001 |
| + Education | 0.0611 | 0.0167 | 0.0003 |
| + Year FE | 0.0582 | 0.0167 | 0.0005 |
| **+ Year + State FE (PREFERRED)** | **0.0576** | **0.0166** | **0.0005** |

### Step 7: Preferred Estimate
- **Effect Estimate:** 0.0576 (5.76 percentage points)
- **Standard Error:** 0.0166
- **95% Confidence Interval:** [0.0250, 0.0902]
- **p-value:** 0.0005
- **Sample Size:** 17,382

### Step 8: Visualizations Created
- `figure1_parallel_trends.png`: Time trends by treatment status
- `figure2_did_visualization.png`: DiD graphical illustration
- `figure3_coefficient_plot.png`: Coefficient estimates across specifications
- `figure4_sample_size.png`: Sample distribution by year

### Step 9: Report Generation
- Created LaTeX report (`replication_report_68.tex`)
- Compiled to PDF (`replication_report_68.pdf`)
- Report includes: Abstract, Introduction, Background, Data, Methodology, Results, Discussion, Conclusion, and Appendix

---

## Key Decisions

### 1. Use of Provided Variables
- Used ELIGIBLE variable as provided (did not create own eligibility indicator)
- Used AFTER variable as provided (did not modify time period definitions)
- Used FT variable as provided for full-time employment outcome

### 2. Estimation Approach
- **Linear Probability Model:** Chose OLS/WLS over logit/probit for straightforward interpretation of DiD coefficient
- **Survey Weights:** Used PERWT weights to obtain population-representative estimates
- **Robust Standard Errors:** Used HC1 (heteroskedasticity-consistent) standard errors

### 3. Control Variables
- **Demographics:** Sex (FEMALE), marital status (MARRIED from MARST=1), children (HAS_CHILDREN from NCHILD>0)
- **Education:** Dummy variables from EDUC_RECODE (reference = Less than High School)
- **Fixed Effects:** Year dummies (reference = 2008), State dummies (STATEFIP)

### 4. Preferred Specification
Selected Model 7 (Year + State FE with full controls) because:
- Survey weights account for complex survey design
- Robust standard errors address heteroskedasticity
- Covariates control for observable differences between groups
- Year FE control for common time trends
- State FE control for time-invariant state-level confounders

### 5. Sample Restrictions
- No observations dropped from provided analytic sample
- Included individuals not in labor force (as FT=0)

---

## Files Created

### Analysis Files
- `analysis_script.py`: Main analysis script
- `create_figures.py`: Figure generation script
- `results_summary.csv`: Results table
- `yearly_ft_rates.csv`: Yearly trends data
- `preferred_model_summary.txt`: Full regression output

### Figures
- `figure1_parallel_trends.png` / `.pdf`
- `figure2_did_visualization.png` / `.pdf`
- `figure3_coefficient_plot.png` / `.pdf`
- `figure4_sample_size.png` / `.pdf`

### Report
- `replication_report_68.tex`: LaTeX source
- `replication_report_68.pdf`: Final report (20 pages)

---

## Summary

This replication finds that DACA eligibility increased the probability of full-time employment by approximately 5.76 percentage points (95% CI: [2.50%, 9.02%], p = 0.0005). The effect is statistically significant across all model specifications and robust to the inclusion of demographic controls, education, year fixed effects, and state fixed effects.
