# Run Log for DACA Replication Study (Replication 48)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA impact study on full-time employment.

**Research Question:** Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (working 35+ hours/week)?

**Research Design:** Difference-in-Differences
- Treatment group: DACA-eligible individuals aged 26-30 in June 2012 (ELIGIBLE=1)
- Control group: Individuals aged 31-35 in June 2012, otherwise would have been eligible (ELIGIBLE=0)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded due to ambiguity)

---

## Session Start
Date: 2026-01-27

### Step 1: Environment Check
- Python 3.14.2 available with required packages:
  - pandas 2.3.3
  - numpy 2.3.5
  - statsmodels 0.14.6
  - scipy 1.16.3
  - matplotlib 3.10.8
  - seaborn 0.13.2
- MiKTeX pdfTeX 4.23 available for LaTeX compilation

### Step 2: Data Files Identified
- `prepared_data_numeric_version.csv`: 17,382 observations (excluding header)
- `prepared_data_labelled_version.csv`: Same data with labeled values
- `acs_data_dict.txt`: Variable documentation

### Step 3: Key Variables Identified
From the data dictionary and instructions:
- **FT**: Full-time employment (1 = yes, 0 = no), defined as usually working 35+ hours/week
- **ELIGIBLE**: 1 for treatment group (ages 26-30 in June 2012), 0 for control group (ages 31-35)
- **AFTER**: 1 for post-DACA years (2013-2016), 0 for pre-DACA years (2008-2011)
- **PERWT**: Person weights from ACS
- Additional variables available for covariates: EDUC_RECODE, SEX, MARST, AGE, etc.
- State-level policy variables available

### Step 4: Analysis Strategy
1. Descriptive statistics for treatment and control groups
2. Pre-trend analysis to check parallel trends assumption
3. Basic DiD regression: FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*(ELIGIBLE*AFTER) + e
4. DiD with covariates for improved precision
5. Robustness checks (state fixed effects, different specifications)

---

## Analysis Commands

### Command 1: Create output directories
```bash
mkdir -p figures results
```

### Command 2: Run main analysis
```bash
python analysis.py
```

---

## Key Decisions

### Decision 1: Use of Pre-Defined ELIGIBLE Variable
**Decision:** Use the provided ELIGIBLE variable rather than constructing own eligibility indicator.
**Rationale:** Instructions explicitly state to use the provided ELIGIBLE variable and not create own eligibility variable. This ensures consistency with the research design.

### Decision 2: Treatment of Non-Labor Force Participants
**Decision:** Keep individuals not in labor force in the analysis (as 0 values for FT).
**Rationale:** Instructions explicitly state that those not in the labor force are included, usually as 0 values. This captures the full employment effect including labor force participation margin.

### Decision 3: Weighting Strategy
**Decision:** Use ACS person weights (PERWT) in weighted least squares regression.
**Rationale:** ACS data requires weighting to be population-representative. Weighted estimates provide population-level average treatment effects.

### Decision 4: Standard Error Approach
**Decision:** Report heteroskedasticity-robust (HC1) standard errors as preferred, with state-clustered standard errors as sensitivity check.
**Rationale:**
- HC1 standard errors account for potential heteroskedasticity in the outcome
- State clustering accounts for potential within-state correlation, important for policy analysis
- Both approaches yield statistically significant results

### Decision 5: Covariate Selection
**Decision:** Include demographic controls (sex, marital status, children, age) and education controls (high school, college+).
**Rationale:**
- These variables are predictive of employment outcomes
- Treatment and control groups differ on age-related characteristics (marriage, children)
- Including controls improves precision and addresses potential confounders
- Kept model parsimonious to avoid overfitting

### Decision 6: Fixed Effects Strategy
**Decision:** Present results with and without state/year fixed effects.
**Rationale:**
- State FE control for time-invariant state-level factors (labor markets, policies)
- Year FE control for common macroeconomic shocks
- Main results robust across specifications

### Decision 7: Preferred Specification
**Decision:** Model 4 (weighted, with demographic and education controls, robust SE) as preferred estimate.
**Rationale:**
- Weighted for population representativeness
- Includes key confounders without overfitting
- Robust SE accounts for heteroskedasticity
- Coefficient stable across specifications (5.92 to 7.48 pp range)

---

## Analysis Results Summary

### Sample Size
- Total N: 17,382
- Treatment (ages 26-30): 11,382
- Control (ages 31-35): 6,000

### Pre-Trend Analysis
- Control group trend: -0.0365/year (p=0.065)
- Treatment group trend: -0.0192/year (p=0.180)
- Difference in trends: +0.0172/year (p=0.380)
- **Conclusion:** No significant differential pre-trends detected

### Main Results

| Model | Coefficient | SE | P-value |
|-------|-------------|-----|---------|
| (1) Basic DiD (unweighted) | 0.0643 | 0.0153 | 0.0000 |
| (2) Basic DiD (weighted) | 0.0748 | 0.0152 | 0.0000 |
| (3) + Demographics | 0.0646 | 0.0142 | 0.0000 |
| (4) + Education | 0.0626 | 0.0142 | 0.0000 |
| (5) + State FE | 0.0621 | 0.0142 | 0.0000 |
| (6) + Year FE | 0.0598 | 0.0142 | 0.0000 |
| (7) + State + Year FE | 0.0592 | 0.0142 | 0.0000 |
| (4R) Preferred w/ Robust SE | 0.0626 | 0.0167 | 0.0002 |
| (4C) Preferred w/ Clustered SE | 0.0626 | 0.0213 | 0.0034 |

### Preferred Estimate
- **Effect Size:** 0.0626 (6.26 percentage points)
- **Standard Error:** 0.0167 (robust)
- **95% CI:** [0.0298, 0.0954]
- **P-value:** 0.0002
- **Sample Size:** 17,382

### Robustness Checks

#### Subgroup Analysis by Sex
- Males: 0.0610 (SE=0.0197, p=0.002)
- Females: 0.0532 (SE=0.0275, p=0.053)

#### Placebo Test (2008-2009 vs 2010-2011)
- Placebo coefficient: 0.0170 (SE=0.0224, p=0.449)
- **Conclusion:** No spurious effect detected

---

## Output Files Generated

### Data Analysis
- `results/regression_results.csv` - Summary of all regression coefficients
- `results/descriptive_stats.csv` - Sample sizes and FT rates by group/period
- `results/event_study_coefficients.csv` - Year-specific treatment effects

### Figures
- `figures/pretrends.png` - Time series of FT rates by treatment status
- `figures/event_study.png` - Event study plot with confidence intervals

### Report
- `replication_report_48.tex` - LaTeX source for replication report
- `replication_report_48.pdf` - Compiled PDF report (19 pages)

---

## Interpretation

DACA eligibility is associated with a statistically significant increase in full-time employment of approximately 6.26 percentage points (95% CI: 2.98 to 9.54 pp). This effect is:

1. **Robust** across multiple specifications (point estimates range from 5.92 to 7.48 pp)
2. **Statistically significant** at conventional levels (p < 0.001 for most specifications, p = 0.003 with clustered SE)
3. **Economically meaningful** - represents roughly 10% improvement relative to baseline FT rate of ~64%
4. **Consistent with theory** - legal work authorization facilitates full-time formal employment
5. **Passes placebo test** - no spurious effect detected in pre-treatment period

The effect appears present for both males and females, though stronger and more precisely estimated for males.

---

## Session End
Date: 2026-01-27

All deliverables produced:
- [x] replication_report_48.tex
- [x] replication_report_48.pdf
- [x] run_log_48.md
