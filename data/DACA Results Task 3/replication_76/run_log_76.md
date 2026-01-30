# Run Log - Replication 76

## Overview
Independent replication of DACA effect on full-time employment study.

## Date
2026-01-27

---

## Step 1: Read and Understand Instructions

**Action:** Read replication_instructions.docx

**Key Points from Instructions:**
- Research Question: Causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born people in the US
- Treatment Group: Ages 26-30 at time of DACA implementation (June 15, 2012)
- Control Group: Ages 31-35 at time of DACA implementation (would have been eligible except for age)
- Outcome: Full-time employment (FT=1 for working 35+ hours/week)
- Data: ACS 2008-2016, excluding 2012
- Key variables provided: ELIGIBLE (1=treatment, 0=control), AFTER (1=2013-2016, 0=2008-2011), FT (outcome)
- Analysis approach: Difference-in-Differences (DiD)

---

## Step 2: Data Examination

**Action:** Examined data structure and key variables

**Data File:** prepared_data_numeric_version.csv
- Observations: 17,382
- Variables: 105

**Key Variable Distributions:**
- YEAR: 2008-2011 (pre-treatment), 2013-2016 (post-treatment), no 2012
- ELIGIBLE: 11,382 (treatment, ages 26-30), 6,000 (control, ages 31-35)
- AFTER: 9,527 (pre), 7,855 (post)
- FT: 11,283 employed full-time, 6,099 not full-time

**Preliminary FT Rates (unweighted):**
| Group | Pre-DACA | Post-DACA | Difference |
|-------|----------|-----------|------------|
| Control (ELIGIBLE=0) | 66.97% | 64.49% | -2.48% |
| Treatment (ELIGIBLE=1) | 62.63% | 66.58% | +3.95% |

**Preliminary DiD (unweighted):** 3.95% - (-2.48%) = +6.43 percentage points

---

## Step 3: Analysis Plan

**Primary Analysis:**
1. Basic Difference-in-Differences using OLS regression
2. Model: FT = β₀ + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE×AFTER) + ε
3. β₃ is the DiD estimate (treatment effect)
4. Use person weights (PERWT) for population-representative estimates
5. Cluster standard errors by state (STATEFIP)

**Robustness Checks:**
1. Add individual covariates (age, sex, education, marital status)
2. Add state fixed effects
3. Add year fixed effects
4. Test parallel trends assumption with year-by-year analysis

---

## Step 4: Execute Analysis

**Command:** `python analysis_76.py`

### Main Results Summary

| Model | Description | DiD Coefficient | SE | p-value |
|-------|-------------|-----------------|-----|---------|
| 1 | Basic (Unweighted) | 0.0643 | 0.0153 | 0.0000 |
| 2 | Basic (Weighted) | 0.0748 | 0.0152 | 0.0000 |
| 3 | Weighted + Clustered SE | 0.0748 | 0.0203 | 0.0002 |
| 4 | + Individual Covariates | 0.0616 | 0.0213 | 0.0039 |
| 5 | + State Fixed Effects | 0.0737 | 0.0209 | 0.0004 |
| 6 | + Year Fixed Effects | 0.0721 | 0.0195 | 0.0002 |
| 7 | Full Model (All Controls) | 0.0583 | 0.0212 | 0.0059 |

### Preferred Estimate (Model 7 - Full Model)
- **Effect Size:** 0.0583 (5.83 percentage points)
- **Standard Error:** 0.0212 (clustered by state)
- **95% CI:** [0.0168, 0.0998]
- **p-value:** 0.0059
- **Sample Size:** 17,382

### Key Decisions Made:

1. **Weighting:** Used person weights (PERWT) for population-representative estimates
2. **Standard Errors:** Clustered by state (STATEFIP) to account for within-state correlation
3. **Covariates included:** Sex (MALE), Age (centered), Education dummies, Marital status, Has children
4. **Fixed Effects:** State and Year fixed effects in full model
5. **Model Selection:** Selected Model 7 (full model) as preferred because it:
   - Controls for observable individual characteristics
   - Accounts for state-level heterogeneity
   - Controls for time trends
   - Provides most conservative estimate
   - Still statistically significant at conventional levels

### Parallel Trends Assessment:

Event study shows some pre-trend differences (2008-2010 coefficients negative), which suggests caution in interpreting results. However:
- 2011 (base year) shows near-zero difference
- Post-period shows consistent positive effects
- The full model controls partially address this concern

### Heterogeneity Results:

| Subgroup | DiD | SE | p-value |
|----------|-----|-----|---------|
| Male | 0.0716 | 0.0195 | 0.0002 |
| Female | 0.0527 | 0.0290 | 0.0696 |
| High School | 0.0608 | 0.0214 | 0.0045 |
| BA+ | 0.1619 | 0.0714 | 0.0233 |

---

## Step 5: Interpretation

The preferred estimate suggests that DACA eligibility increased the probability of full-time employment by approximately 5.8 percentage points among eligible Hispanic-Mexican individuals aged 26-30 compared to the control group aged 31-35.

This effect is:
- Statistically significant at the 1% level (p = 0.006)
- Robust across multiple specifications (range: 5.8% to 7.5%)
- Larger for males and those with higher education
- Economically meaningful given baseline FT rate of ~64%

---

## Step 6: Generate Figures

**Command:** `python generate_figures.py`

Generated the following figures:
1. `figure1_parallel_trends.png` - FT rates by treatment/control over time
2. `figure2_event_study.png` - Year-specific treatment effects
3. `figure3_did_bars.png` - DiD visualization (2x2 bar chart)
4. `figure4_model_comparison.png` - Coefficient comparison across models
5. `figure5_heterogeneity_sex.png` - Heterogeneity by sex
6. `figure6_sample_distribution.png` - Sample distribution

---

## Step 7: Create LaTeX Report

**Command:** `pdflatex replication_report_76.tex` (3 passes)

Created 22-page replication report including:
- Abstract
- Introduction with research question and DACA background
- Data and Methods section
- Main results with tables
- Robustness and sensitivity analysis
- Heterogeneity analysis
- Discussion of limitations
- Conclusion
- Technical appendix

---

## Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| `replication_report_76.tex` | LaTeX source file | Complete |
| `replication_report_76.pdf` | PDF report (22 pages) | Complete |
| `run_log_76.md` | This run log | Complete |
| `analysis_76.py` | Main analysis code | Complete |
| `generate_figures.py` | Figure generation code | Complete |

---

## Summary of Analytic Choices

1. **Estimation method:** Weighted Least Squares with person weights (PERWT)
2. **Standard errors:** Clustered by state (STATEFIP)
3. **Preferred specification:** Full model with individual covariates + state FE + year FE
4. **Covariates:** Sex, age (centered), education (4 dummies), marital status, has children
5. **Sample:** Full provided sample (17,382 observations), no additional exclusions
6. **Outcome coding:** FT as provided (1 = full-time, 0 = not full-time)
7. **Treatment coding:** ELIGIBLE as provided (1 = treatment, 0 = control)

---

## Preferred Estimate for Submission

- **Effect Size:** 0.0583 (5.83 percentage points)
- **Standard Error:** 0.0212
- **95% CI:** [0.0168, 0.0998]
- **Sample Size:** 17,382
