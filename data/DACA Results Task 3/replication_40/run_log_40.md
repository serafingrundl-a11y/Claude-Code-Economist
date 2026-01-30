# DACA Replication Study - Run Log 40

## Date: 2026-01-27

## Task Overview
Replicate the analysis examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born individuals in the United States.

---

## Step 1: Read and Understand Instructions (Completed)

**Time:** Start of session

### Key Design Elements from Instructions:
- **Research Question:** Effect of DACA eligibility on probability of full-time employment (35+ hours/week)
- **Treatment Group:** Ages 26-30 at time of policy (June 15, 2012), ELIGIBLE=1
- **Comparison Group:** Ages 31-35 at time of policy (June 15, 2012), ELIGIBLE=0
- **Method:** Difference-in-Differences (DiD)
- **Pre-period:** 2008-2011
- **Post-period:** 2013-2016 (2012 excluded as transition year)
- **Outcome:** FT (1 = full-time work, 0 = not full-time)

### Key Variables Provided:
- `FT`: Outcome variable (1 = full-time, 0 = not full-time)
- `AFTER`: Treatment period indicator (1 = 2013-2016, 0 = 2008-2011)
- `ELIGIBLE`: Treatment group indicator (1 = ages 26-30, 0 = ages 31-35)
- Survey weights: `PERWT`

### Important Notes:
- Data is repeated cross-section, NOT panel data
- Keep all observations including those not in labor force
- Do not create own eligibility variable - use provided ELIGIBLE
- Binary IPUMS variables: 1=No, 2=Yes
- Created variables (FT, AFTER, ELIGIBLE): 0=No, 1=Yes

---

## Step 2: Data Exploration (Completed)

### Data Files:
- `prepared_data_numeric_version.csv`: 17,382 observations, 105 variables
- `prepared_data_labelled_version.csv`: Same data with labeled values

### Key Variable Summary:
- **YEAR:** 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **FT:** 11,283 (65%) full-time, 6,099 (35%) not full-time
- **AFTER:** 9,527 pre-period, 7,855 post-period
- **ELIGIBLE:** 11,382 in treatment group (ages 26-30), 6,000 in comparison group (ages 31-35)

---

## Step 3: Difference-in-Differences Analysis (Completed)

### Primary Model Specification:
FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE×AFTER) + ε

Where β3 is the DiD estimator representing the causal effect of DACA eligibility on full-time employment.

### Analysis Decisions:
1. Use person weights (PERWT) for survey-weighted estimates
2. Cluster standard errors by state (STATEFIP) to account for within-state correlation
3. Run both unweighted and weighted specifications
4. Run specifications with and without covariates
5. Check parallel trends assumption using event study approach

---

## Step 4: Analysis Implementation (Completed)

### Key Analytical Choices:

1. **Estimation Method:** Linear Probability Model (OLS/WLS) as primary specification
   - Rationale: Direct interpretation of coefficients as percentage point changes
   - Logit model run for robustness check

2. **Weighting:** Used PERWT (person weights) for survey-representative estimates
   - ACS is a complex survey; weights account for sampling design

3. **Standard Errors:** Clustered at state level (STATEFIP)
   - Rationale: Policy implementation and labor markets vary by state
   - Accounts for within-state correlation in outcomes

4. **Covariates included:**
   - Demographics: FEMALE, MARRIED, AGE, NCHILD
   - Education: HS degree, Some college, Two-year degree, BA+
   - Reference category: Less than high school

5. **Fixed Effects:**
   - Year fixed effects: Control for common time trends
   - State fixed effects: Control for time-invariant state differences

### Models Estimated:
1. Basic OLS (no weights, no clustering)
2. Robust standard errors (HC3)
3. State-clustered standard errors
4. Weighted with state-clustered SE
5. + Demographics
6. + Education
7. + State labor market (UNEMP, LFPR)
8. Year fixed effects
9. State fixed effects
10. Year + State fixed effects
11. **Preferred:** Year FE + State FE + Demographics + Education (weighted, clustered)

---

## Step 5: Results Summary

### Raw DiD Calculation (Weighted):
| Group | Pre-Period | Post-Period | Change |
|-------|------------|-------------|--------|
| Treatment (26-30) | 0.6369 | 0.6860 | +0.0491 |
| Comparison (31-35) | 0.6886 | 0.6629 | -0.0257 |
| **DiD** | | | **+0.0748** |

### Regression Results Summary:

| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Basic OLS | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| Weighted + Clustered | 0.0748 | 0.0203 | [0.035, 0.114] | <0.001 |
| + Demographics | 0.0674 | 0.0213 | [0.026, 0.109] | 0.002 |
| + Education | 0.0646 | 0.0217 | [0.022, 0.107] | 0.003 |
| Year + State FE | 0.0710 | 0.0202 | [0.032, 0.111] | <0.001 |
| **Preferred (Full)** | **0.0614** | **0.0215** | **[0.019, 0.104]** | **0.004** |

### Preferred Estimate:
- **Effect Size:** 6.14 percentage points
- **Standard Error:** 2.15 percentage points
- **95% CI:** [1.92, 10.36] percentage points
- **p-value:** 0.0044

### Interpretation:
DACA eligibility is associated with a statistically significant 6.14 percentage point increase in the probability of full-time employment. This effect is robust across various specifications, with estimates ranging from 6.1 to 7.5 percentage points.

---

## Step 6: Parallel Trends Assessment

### Event Study Results (relative to 2011):
| Year | Coefficient | SE | 95% CI |
|------|------------|-----|--------|
| 2008 | -0.0722 | 0.0283 | [-0.128, -0.017] |
| 2009 | -0.0517 | 0.0377 | [-0.126, 0.022] |
| 2010 | -0.0832 | 0.0294 | [-0.141, -0.026] |
| 2011 | 0 (ref) | - | - |
| 2013 | 0.0171 | 0.0396 | [-0.061, 0.095] |
| 2014 | -0.0062 | 0.0274 | [-0.060, 0.048] |
| 2015 | -0.0026 | 0.0383 | [-0.078, 0.072] |
| 2016 | 0.0709 | 0.0300 | [0.012, 0.130] |

### Assessment:
- Pre-period coefficients show some differential trends (2008, 2010 significant)
- This suggests potential violation of parallel trends assumption
- Caution warranted in interpreting causal effects
- Results may partially reflect pre-existing differential trends

---

## Step 7: Heterogeneity Analysis

### By Sex:
| Group | N | DiD Coefficient | SE | 95% CI | p-value |
|-------|---|-----------------|-----|--------|---------|
| Male | 9,075 | 0.0716 | 0.0195 | [0.033, 0.110] | <0.001 |
| Female | 8,307 | 0.0527 | 0.0290 | [-0.004, 0.110] | 0.070 |

The effect appears stronger and more precisely estimated for males. The effect for females is similar in magnitude but not statistically significant at conventional levels.

---

## Step 8: Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Estimation method | Linear Probability Model | Direct interpretation, consistent with literature |
| Weights | PERWT | Survey weights for representativeness |
| SE clustering | State level | Policy and labor market variation |
| Covariates | Demographics + Education | Control for observable differences |
| Fixed effects | Year + State | Control for time trends and state heterogeneity |
| Sample | Full sample | As instructed, no subgroup restrictions |

---

## Step 9: Files Generated

1. `analysis_script.py` - Main analysis code
2. `regression_results_summary.csv` - Summary of all model results
3. `replication_report_40.tex` - LaTeX report
4. `replication_report_40.pdf` - Final PDF report
5. `run_log_40.md` - This log file

---

## Conclusion

The analysis finds that DACA eligibility is associated with a 6.1 percentage point increase in full-time employment probability. This effect is statistically significant and robust across specifications. However, the event study analysis suggests some concern about pre-existing differential trends between treatment and comparison groups, warranting caution in interpreting the estimate as purely causal.

---

*Log completed: 2026-01-27*
