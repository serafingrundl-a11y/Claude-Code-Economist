# Run Log - Replication 87

## Date: 2026-01-27

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at time of policy implementation (June 15, 2012)
- **Control Group**: DACA-ineligible individuals aged 31-35 at time of policy implementation (otherwise would have been eligible)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment Period**: 2008-2011
- **Post-treatment Period**: 2013-2016 (2012 excluded)

## Key Variables
- **FT**: Full-time employment (1 = yes, 0 = no), outcome variable
- **ELIGIBLE**: Treatment group indicator (1 = DACA-eligible ages 26-30, 0 = comparison group ages 31-35)
- **AFTER**: Post-treatment period indicator (1 = 2013-2016, 0 = 2008-2011)

---

## Session Log

### Step 1: Initial Setup and Data Exploration
**Time**: Start of analysis

**Actions**:
- Read replication instructions from docx file
- Examined data dictionary (acs_data_dict.txt)
- Reviewed data files: prepared_data_numeric_version.csv and prepared_data_labelled_version.csv

**Key Observations from Data**:
- Data spans 2008-2016 (2012 omitted)
- Variables include demographic, employment, education, and state-level policy variables
- ELIGIBLE, AFTER, and FT variables pre-constructed
- AGE_IN_JUNE_2012 provides exact age at policy implementation

**Decision**: Use numeric version for analysis (cleaner for regression)

---

### Step 2: Data Loading and Initial Statistics
**Time**: Analysis execution

**Dataset Characteristics**:
- Total observations: 17,382
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Columns: 105 variables

**Sample Composition**:
| Group | Pre-Period | Post-Period | Total |
|-------|------------|-------------|-------|
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| **Total** | 9,527 | 7,855 | 17,382 |

**Key Decision**: Used PERWT (person weights) for weighted regressions to obtain population-representative estimates.

---

### Step 3: Difference-in-Differences Analysis
**Time**: Main regression analysis

**Raw DiD Calculations**:
- Unweighted Raw DiD: 0.0643
- Weighted Raw DiD: 0.0748

**Model Specifications Tested**:
1. Basic DiD (unweighted)
2. Basic DiD (weighted with PERWT)
3. Year Fixed Effects (unweighted)
4. Year Fixed Effects (weighted)
5. Covariates only (unweighted)
6. **Covariates (weighted) - PREFERRED**
7. Covariates + State FE (weighted)
8. Full model (covariates + year FE + state FE, weighted)

**Preferred Specification (Model 6)**:
- Formula: FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA
- Weighted by PERWT

**Covariate Choices Rationale**:
- AGE: Controls for age-specific employment patterns within groups
- FEMALE: Strong predictor of employment (women 32.8pp less likely to be FT employed)
- MARRIED: Marital status affects labor supply decisions
- NCHILD: Number of children impacts workforce participation
- Education dummies: Education strongly associated with employment outcomes

---

### Step 4: Results Summary

**Main Results (Preferred Model 6)**:
- DiD Coefficient: **0.0646**
- Standard Error: 0.0142
- 95% CI: [0.0368, 0.0924]
- P-value: < 0.0001
- Sample Size: 17,382
- R-squared: 0.1308

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 6.5 percentage points among the eligible population.

---

### Step 5: Robustness Checks

**1. Event Study / Parallel Trends**:
- Pre-treatment coefficients (2008, 2009, 2010) relative to 2011 show some volatility
- Pre-trend concerns: 2008 (coef=-0.068, p=0.019), 2010 (coef=-0.082, p=0.005)
- However, treatment effects emerge post-2012, especially in 2016 (coef=0.074, p=0.018)

**2. Placebo Test (2010 as fake treatment)**:
- Placebo DiD: 0.0167
- P-value: 0.386
- Result: Non-significant, supports parallel trends assumption

**3. Narrower Age Band (27-29 vs 32-34)**:
- DiD: 0.0556
- SE: 0.0199
- Result: Consistent with main finding

**4. State Policy Controls**:
- DiD with policy controls: 0.0623
- Result: Robust to inclusion of state-level immigration policies

---

### Step 6: Subgroup Analysis

| Subgroup | DiD Estimate | SE | P-value | N |
|----------|--------------|-----|---------|---|
| Male | 0.0623 | 0.0170 | 0.0003 | 9,075 |
| Female | 0.0573 | 0.0228 | 0.0121 | 8,307 |
| High School Degree | 0.0494 | 0.0164 | 0.0026 | 12,444 |
| Some College | 0.0694 | 0.0370 | 0.0610 | 2,877 |
| Two-Year Degree | 0.1746 | 0.0634 | 0.0060 | 991 |
| BA+ | 0.1304 | 0.0585 | 0.0260 | 1,058 |
| Married | 0.0113 | 0.0192 | 0.5551 | 7,851 |
| Not Married | 0.0952 | 0.0205 | 0.0000 | 9,531 |

**Key Findings**:
- Effects similar for males and females
- Stronger effects for those with higher education (Two-Year Degree, BA+)
- Effects driven primarily by unmarried individuals
- Married individuals show no significant effect

---

### Step 7: Outputs Generated

**Files Created**:
1. `analysis.py` - Main analysis script
2. `results_summary.csv` - Summary of all model results
3. `figure1_parallel_trends.png` - FT employment trends by group
4. `figure2_event_study.png` - Event study coefficients
5. `figure3_model_comparison.png` - Comparison across specifications
6. `figure4_subgroups.png` - Subgroup analysis visualization
7. `replication_report_87.tex` - LaTeX report
8. `replication_report_87.pdf` - Final PDF report

---

## Key Methodological Decisions

1. **Survey Weights**: Used PERWT to obtain population-representative estimates
2. **Standard Errors**: Reported both conventional and heteroskedasticity-robust (HC1) SEs
3. **Covariates**: Included demographic controls (age, sex, marital status, children, education) to improve precision and account for composition differences
4. **Fixed Effects**: Tested year and state fixed effects for robustness
5. **Reference Year**: Used 2011 as reference in event study (immediate pre-treatment year)
6. **Sample**: Used full provided sample without additional restrictions per instructions

## Final Preferred Estimate

| Metric | Value |
|--------|-------|
| Effect Size | 0.0646 (6.46 pp) |
| Standard Error | 0.0142 |
| 95% Confidence Interval | [0.0368, 0.0924] |
| Sample Size | 17,382 |
