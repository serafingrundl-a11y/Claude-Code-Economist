# Replication Run Log - Study 97

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability of full-time employment (35+ hours/week)?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at policy implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at policy implementation who would have been eligible but for age
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016
- **Outcome**: Full-time employment (FT = 1 if usually working 35+ hours/week)

---

## Session Log

### Step 1: Data Exploration
**Actions**:
- Read replication instructions from `replication_instructions.docx`
- Examined data dictionary (`acs_data_dict.txt`)
- Identified key variables:
  - `FT`: Full-time employment indicator (0/1)
  - `ELIGIBLE`: DACA eligibility indicator (1 = treated group ages 26-30, 0 = control group ages 31-35)
  - `AFTER`: Post-treatment indicator (1 for 2013-2016, 0 for 2008-2011)
  - `PERWT`: Person weight for survey weighting
  - Various demographic controls available

**Data file**: `prepared_data_numeric_version.csv`
- Contains 17,382 observations
- Years: 2008-2011 (pre) and 2013-2016 (post); 2012 excluded

---

### Step 2: Data Loading and Verification

**Key Decisions**:
1. Used the provided `ELIGIBLE` variable as-is (did not construct own eligibility criteria)
2. Used the provided `AFTER` variable as-is
3. Used the provided `FT` variable as the outcome
4. Retained all observations including those not in labor force (coded as FT=0)

**Sample Verification**:
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (AFTER=0): 9,527 observations
- Post-period (AFTER=1): 7,855 observations

---

### Step 3: Analytic Approach

**Decision**: Use Difference-in-Differences (DiD) regression

**Rationale**:
- The research design explicitly calls for comparing ages 26-30 (treatment) to ages 31-35 (control) before vs. after DACA
- DiD is the standard approach for this quasi-experimental setting
- The AFTER variable already distinguishes pre/post periods
- The ELIGIBLE variable already distinguishes treatment/control groups

**Model Specification**:
```
FT = α + β₁·ELIGIBLE + β₂·AFTER + β₃·(ELIGIBLE × AFTER) + ε
```

Where β₃ is the coefficient of interest (the DiD estimate).

---

### Step 4: Estimation Choices

**Key Decisions**:

1. **Survey Weights**: Used person weights (`PERWT`) for population-representative estimates
   - Rationale: ACS is a complex survey; weights ensure estimates represent the target population

2. **Standard Errors**: Used heteroskedasticity-robust (HC1) standard errors
   - Rationale: Standard approach for cross-sectional data with potential heteroskedasticity

3. **Preferred Model**: Model 4 (WLS with robust SE)
   - Rationale: Combines proper survey weighting with robust inference

4. **Covariates**: Estimated multiple specifications for robustness
   - Baseline (no covariates)
   - Demographics (female, married, number of children)
   - Education indicators
   - Year fixed effects
   - State fixed effects

---

### Step 5: Main Results

**Preferred Estimate (Model 4: WLS with robust SE)**:
- DiD coefficient: **0.0748**
- Standard error: 0.0181
- 95% CI: [0.039, 0.110]
- p-value: < 0.001
- Sample size: 17,382

**Interpretation**: DACA eligibility increased full-time employment by approximately 7.48 percentage points.

---

### Step 6: Robustness Checks

**A. Parallel Trends Test**:
- Tested for differential pre-trends between treatment and control
- Interaction coefficient (ELIGIBLE × YEAR): 0.017 (p = 0.113)
- **Result**: No significant differential pre-trends; parallel trends assumption supported

**B. Model Stability**:
| Model | Estimate | SE | p-value |
|-------|----------|-----|---------|
| (1) OLS, unweighted | 0.0643 | 0.0153 | <0.001 |
| (2) WLS, weighted | 0.0748 | 0.0152 | <0.001 |
| (3) OLS, robust SE | 0.0643 | 0.0153 | <0.001 |
| (4) WLS, robust SE (preferred) | 0.0748 | 0.0181 | <0.001 |
| (5) WLS + demographics | 0.0668 | 0.0168 | <0.001 |
| (6) WLS + demographics + education | 0.0640 | 0.0167 | <0.001 |
| (7) WLS + year FE | 0.0613 | 0.0167 | <0.001 |
| (8) WLS + state + year FE | 0.0607 | 0.0166 | <0.001 |

**Result**: Estimates stable across specifications (range: 0.061 to 0.075)

**C. Subgroup Analysis by Sex**:
- Males: 0.0716 (SE=0.0199, p<0.001)
- Females: 0.0527 (SE=0.0281, p=0.061)
- **Result**: Effect present for both sexes; larger and more precisely estimated for males

---

### Step 7: Visualization

Created five figures:
1. `figure1_event_study.png/.pdf` - Year-by-year treatment-control differences
2. `figure2_parallel_trends.png/.pdf` - Full-time employment trends by group
3. `figure3_did_visualization.png/.pdf` - 2x2 DiD visualization with counterfactual
4. `figure4_coefficient_comparison.png/.pdf` - Coefficient comparison across models
5. `figure5_subgroup_analysis.png/.pdf` - Subgroup analysis by sex

---

### Step 8: Report Generation

- Created `replication_report_97.tex` (LaTeX source)
- Compiled to `replication_report_97.pdf` (20 pages)

---

## Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script (Python) |
| `create_figures.py` | Figure generation script |
| `replication_report_97.tex` | LaTeX report source |
| `replication_report_97.pdf` | Final PDF report (20 pages) |
| `run_log_97.md` | This run log |
| `figure1_event_study.png/.pdf` | Event study figure |
| `figure2_parallel_trends.png/.pdf` | Parallel trends figure |
| `figure3_did_visualization.png/.pdf` | DiD visualization |
| `figure4_coefficient_comparison.png/.pdf` | Model comparison figure |
| `figure5_subgroup_analysis.png/.pdf` | Subgroup analysis figure |

---

## Software Used

- Python 3.x
  - pandas (data manipulation)
  - numpy (numerical operations)
  - statsmodels (regression analysis)
  - matplotlib (visualization)
- pdfLaTeX (document compilation)

---

## Summary of Key Decisions

1. **Did not create own eligibility variable** - Used provided `ELIGIBLE` variable as instructed
2. **Did not drop any observations** - Used full provided sample as instructed
3. **Included non-labor force participants** - As specified, those not in labor force are coded FT=0
4. **Used survey weights** - Essential for population-representative inference with ACS data
5. **Used robust standard errors** - Conservative choice for valid inference
6. **Selected basic DiD as preferred model** - Parsimonious specification that captures treatment effect
7. **Conducted multiple robustness checks** - Ensured results not sensitive to specification choices

---

## Final Results Summary

**Effect of DACA eligibility on full-time employment:**
- Point estimate: **0.0748** (7.48 percentage points)
- Standard error: 0.0181
- 95% CI: [0.039, 0.110]
- p-value: < 0.001
- Sample size: 17,382

**Conclusion**: The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment. The treatment group experienced approximately a 7.5 percentage point increase in full-time employment relative to the control group after DACA implementation.
