# Run Log for DACA Replication Study (Task 84)

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (defined as usually working 35 hours per week or more)?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at the time of policy implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at policy implementation who would otherwise be eligible if not for age
- **Identification Strategy**: Difference-in-differences (DiD) comparing changes in FT employment from pre-DACA (2008-2011) to post-DACA (2013-2016) between treatment and control groups
- **Outcome Variable**: FT (full-time employment, 1 = working 35+ hours/week)
- **Key Variables**: ELIGIBLE (treatment indicator), AFTER (post-treatment period indicator)

## Data Description
- **Source**: American Community Survey (ACS) via IPUMS USA
- **Years**: 2008-2016 (excluding 2012)
- **Sample Size**: 17,382 observations
- **Pre-period**: 2008-2011 (AFTER = 0)
- **Post-period**: 2013-2016 (AFTER = 1)

---

## Session Log

### Step 1: Data Exploration
**Timestamp**: Session start
**Action**: Read replication_instructions.docx and examined data dictionary

**Key Variables Identified:**
- `ELIGIBLE`: 1 = treatment group (ages 26-30 in June 2012), 0 = control group (ages 31-35)
- `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- `FT`: 1 = full-time employment (35+ hours/week), 0 = not full-time
- `PERWT`: Person weight for survey weighting
- Various demographic controls: SEX, AGE, MARST, YRSUSA1, EDUC_RECODE, HS_DEGREE
- State identifier: STATEFIP
- State policy variables available but not used in main specification

**Data Files:**
- `prepared_data_numeric_version.csv`: Numeric coding (used for analysis)
- `prepared_data_labelled_version.csv`: Labeled version for reference
- `acs_data_dict.txt`: Data dictionary from IPUMS

**Command:**
```python
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df)}")  # Output: 17382
```

---

### Step 2: Analysis Setup
**Decision**: Use Python with pandas and statsmodels for regression analysis
**Rationale**: Python provides robust statistical modeling capabilities with clear, reproducible code. Statsmodels allows for weighted least squares, heteroskedasticity-robust standard errors, and easy implementation of fixed effects.

**Key Analysis Decisions:**
1. Use survey weights (PERWT) to ensure nationally representative estimates
2. Use heteroskedasticity-robust (HC1) standard errors
3. Include demographic controls: FEMALE, MARRIED, AGE, YRSUSA1, HS_DEGREE_BIN
4. Include state fixed effects to control for time-invariant state-level confounders
5. Keep all observations in the sample as instructed (no further sample restrictions)

---

### Step 3: Sample Composition Analysis

**Sample Sizes by Group:**
```
                   Pre-DACA (2008-11)  Post-DACA (2013-16)  Total
Control (31-35)                  3294                 2706   6000
Treatment (26-30)                6233                 5149  11382
Total                            9527                 7855  17382
```

**Baseline Characteristics:**
- Treatment group (26-30): Mean age 27.97, 48.2% female, 19.4 years in USA
- Control group (31-35): Mean age 32.75, 47.1% female, 23.7 years in USA

---

### Step 4: Descriptive Statistics

**Full-time Employment Rates (Weighted):**
```
                   Pre-DACA  Post-DACA   Change
Control (31-35)      0.6886     0.6629   -0.0257
Treatment (26-30)    0.6369     0.6860   +0.0491
```

**Simple DiD (weighted): 0.0748**

**Year-by-Year FT Rates:**
```
Year    Control    Treatment
2008     0.7264      0.6667
2009     0.6569      0.6174
2010     0.6733      0.6064
2011     0.6175      0.6168
2013     0.6238      0.6420
2014     0.6492      0.6397
2015     0.6501      0.6797
2016     0.6598      0.7082
```

---

### Step 5: Regression Analysis

**Models Estimated:**

1. **Model 1: Basic OLS (no weights)**
   - Formula: FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER
   - DiD coefficient: 0.0643 (SE: 0.0153, p < 0.001)

2. **Model 2: WLS with survey weights**
   - Formula: FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER
   - Weights: PERWT
   - DiD coefficient: 0.0748 (SE: 0.0181, p < 0.001)

3. **Model 3: WLS with demographic controls**
   - Formula: FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN
   - DiD coefficient: 0.0649 (SE: 0.0168, p < 0.001)

4. **Model 4: WLS with demographics + state FE (PREFERRED)**
   - Formula: FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + YRSUSA1 + HS_DEGREE_BIN + C(STATEFIP)
   - DiD coefficient: 0.0645 (SE: 0.0167, p = 0.0001)
   - 95% CI: [0.0317, 0.0972]

5. **Model 5: WLS with demographics + state FE + year FE**
   - DiD coefficient: 0.0617 (SE: 0.0167, p = 0.0002)

---

### Step 6: Robustness Checks

**Pre-trends Test:**
- Tested for differential pre-trends by interacting ELIGIBLE with linear time trend in pre-period
- Coefficient: 0.0174 (SE: 0.0110, p = 0.113)
- Result: No significant differential pre-trend detected

**Placebo Test:**
- Estimated DiD using only pre-period data (2010-2011 as "post")
- Placebo coefficient: 0.0187 (SE: 0.0224, p = 0.403)
- Result: Placebo test passed (no significant effect)

**Alternative Estimators:**
- Probit marginal effect: 0.0559 (p < 0.001)
- Logit marginal effect: 0.0552 (p < 0.001)

---

### Step 7: Heterogeneity Analysis

**By Gender:**
- Males (N=9,075): DiD = 0.0635 (SE: 0.0198, p = 0.001)
- Females (N=8,307): DiD = 0.0542 (SE: 0.0277, p = 0.050)

**By Education:**
- HS degree or higher (N=17,370): DiD = 0.0648 (SE: 0.0168, p < 0.001)
- Less than HS: Too few observations for reliable estimate

---

### Step 8: Key Decisions and Justifications

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weighting | Survey weights (PERWT) | Ensures nationally representative estimates |
| Standard errors | HC1 robust | Accounts for heteroskedasticity |
| Controls | Demographics + State FE | Controls for observable confounders and time-invariant state differences |
| Sample | Full sample | Following instructions to not drop observations |
| Preferred model | Model 4 | Balances control for confounders with parsimony |

---

## Final Results Summary

**Preferred Estimate (Model 4):**
- Effect: 0.0645 (6.45 percentage points)
- Standard Error: 0.0167
- 95% CI: [0.0317, 0.0972]
- P-value: 0.0001
- Sample Size: 17,382

**Interpretation:** DACA eligibility is associated with a statistically significant 6.45 percentage point increase in full-time employment probability for the treatment group relative to the control group, after controlling for demographic characteristics and state fixed effects.

---

## Files Generated

1. `analysis_script.py` - Main analysis code
2. `analysis_results.json` - Numerical results for report generation
3. `replication_report_84.tex` - LaTeX report
4. `replication_report_84.pdf` - Compiled PDF report
5. `run_log_84.md` - This log file
