# Run Log - DACA Replication Study 18

## Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA effect on full-time employment study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (≥35 hours/week)?

## Study Design
- **Treatment group**: DACA-eligible individuals aged 26-30 on June 15, 2012
- **Control group**: Individuals aged 31-35 on June 15, 2012 (would have been eligible but for age)
- **Pre-treatment period**: 2008-2011
- **Post-treatment period**: 2013-2016 (2012 excluded as treatment year)
- **Method**: Difference-in-Differences (DiD)

---

## Session Log

### Step 1: Initial Data Exploration
**Date**: 2026-01-27

**Files identified:**
- `data/prepared_data_numeric_version.csv` - Main analysis data (17,382 observations + header)
- `data/prepared_data_labelled_version.csv` - Labeled version of data
- `data/acs_data_dict.txt` - Data dictionary from IPUMS

**Key variables identified:**
- `FT`: Full-time employment (1 = full-time, 0 = not full-time) - OUTCOME
- `ELIGIBLE`: DACA eligibility indicator (1 = treatment group ages 26-30, 0 = control group ages 31-35)
- `AFTER`: Post-DACA indicator (1 = 2013-2016, 0 = 2008-2011)
- `YEAR`: Survey year
- `PERWT`: Person weight for survey weighting
- Demographics: `SEX`, `AGE`, `MARST`, `EDUC_RECODE`, `RACE_RECODE`
- State-level policies: `DRIVERSLICENSES`, `INSTATETUITION`, `STATEFINANCIALAID`, etc.

**Decision 1**: Use the `prepared_data_numeric_version.csv` file as instructed. The data has already been filtered to include only the relevant sample.

### Step 2: Analytical Approach
**Decision 2**: Primary specification will be a standard Difference-in-Differences model:
```
FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE × AFTER) + ε
```
Where β3 is the DiD estimator representing the causal effect of DACA eligibility on full-time employment.

**Decision 3**: Will use survey weights (PERWT) to account for ACS sampling design.

**Decision 4**: Will estimate robust standard errors clustered at the state level to account for within-state correlation.

**Decision 5**: Will include additional specifications with covariates for robustness:
- Demographic controls: sex, marital status, education
- State-level policy controls
- Year fixed effects
- State fixed effects

### Step 3: Python Analysis Script Creation
Creating comprehensive analysis script with:
1. Data loading and inspection
2. Summary statistics
3. Pre-trend analysis
4. Main DiD estimation
5. Robustness checks with covariates
6. Heterogeneity analysis by subgroups
7. Output generation for LaTeX report

### Step 4: Analysis Results
**Command executed**: `python analysis.py`

**Key Results:**

#### Data Summary
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (2008-2011): 9,527 observations
- Post-period (2013-2016): 7,855 observations

#### 2x2 DiD Table (Weighted Means)
|              | Pre (2008-2011) | Post (2013-2016) | Change   |
|--------------|-----------------|------------------|----------|
| Treatment    | 0.637           | 0.686            | +0.049   |
| Control      | 0.689           | 0.663            | -0.026   |
| **DiD**      |                 |                  | **0.075**|

#### Main Regression Results (Preferred: Weighted OLS)
- **DiD Estimate**: 0.0748 (7.48 percentage points)
- **Standard Error**: 0.0152
- **95% CI**: [0.045, 0.105]
- **p-value**: < 0.001

#### Robustness Checks Summary
| Model                    | DiD Estimate | SE     | p-value |
|--------------------------|--------------|--------|---------|
| Basic OLS (unweighted)   | 0.0643       | 0.0153 | <0.001  |
| Weighted OLS (preferred) | 0.0748       | 0.0152 | <0.001  |
| Robust SE (HC3)          | 0.0643       | 0.0153 | <0.001  |
| + Demographics           | 0.0623       | 0.0142 | <0.001  |
| + Year FE                | 0.0721       | 0.0151 | <0.001  |
| + State FE               | 0.0737       | 0.0152 | <0.001  |
| Full Model               | 0.0589       | 0.0142 | <0.001  |
| Clustered SE (state)     | 0.0643       | 0.0141 | <0.001  |

#### Pre-Trends Analysis
- Pre-trend coefficient (ELIGIBLE x YEAR_TREND): 0.0174
- p-value: 0.058
- Interpretation: Marginally non-significant, suggesting reasonable parallel trends assumption

#### Heterogeneity Analysis
- Males: DiD = 0.072 (SE: 0.017), statistically significant
- Females: DiD = 0.053 (SE: 0.023), statistically significant
- BA+ education: DiD = 0.162 (SE: 0.060), largest effect size

#### Event Study Results (Reference: 2011)
Pre-treatment coefficients suggest some pre-existing differences but no clear trend.
Post-treatment: 2016 shows the largest positive effect (0.074, p<0.05).

**Decision 6**: Selected weighted OLS as preferred specification because:
- Accounts for ACS complex survey design through PERWT weights
- Simple and transparent model
- Consistent with other specifications
- Effect is robust across all model specifications

### Step 5: Output Files Generated
- `results_summary.csv` - Main regression results table
- `yearly_means.csv` - FT rates by year and eligibility
- `event_study_coefficients.csv` - Event study coefficients
- `demographic_summary.csv` - Demographic characteristics by group
- `heterogeneity_results.csv` - Subgroup analysis results

### Step 6: LaTeX Report Creation
**Command executed**: Created `replication_report_18.tex`

LaTeX report structure (~21 pages):
1. Abstract
2. Table of Contents
3. Introduction
4. Institutional Background (DACA program overview, eligibility, demographics)
5. Data (source, sample construction, key variables)
6. Empirical Methodology (DiD framework, specifications, assumptions)
7. Results (descriptive statistics, main regression results, preferred estimate)
8. Robustness and Additional Analyses (pre-trends, event study, heterogeneity, state policy interactions)
9. Discussion and Conclusion
10. Appendix (variable definitions, sample sizes, complete regression output)
11. References

### Step 7: PDF Compilation
**Commands executed**:
```
pdflatex -interaction=nonstopmode replication_report_18.tex
pdflatex -interaction=nonstopmode replication_report_18.tex
pdflatex -interaction=nonstopmode replication_report_18.tex
```
(Multiple runs to resolve cross-references)

**Output**: `replication_report_18.pdf` (21 pages, ~337 KB)

---

## Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| `replication_report_18.tex` | LaTeX source for replication report | Complete |
| `replication_report_18.pdf` | PDF replication report (~21 pages) | Complete |
| `run_log_18.md` | Run log documenting commands and decisions | Complete |
| `analysis.py` | Python analysis script | Complete |
| `results_summary.csv` | Summary of main regression results | Complete |
| `yearly_means.csv` | FT employment rates by year and group | Complete |
| `event_study_coefficients.csv` | Event study coefficients | Complete |
| `demographic_summary.csv` | Demographics by treatment group | Complete |
| `heterogeneity_results.csv` | Subgroup analysis results | Complete |

---

## Summary of Key Decisions

1. **Data**: Used provided `prepared_data_numeric_version.csv` without further sample restrictions
2. **Outcome**: Full-time employment (FT variable, binary)
3. **Treatment/Control**: Used provided ELIGIBLE variable (ages 26-30 vs 31-35 on June 15, 2012)
4. **Method**: Difference-in-Differences with pre-period 2008-2011 and post-period 2013-2016
5. **Preferred specification**: Weighted OLS with ACS person weights (PERWT)
6. **Standard errors**: Reported conventional, robust, and state-clustered versions
7. **Robustness**: Multiple specifications with demographic controls, year FE, state FE
8. **Additional analyses**: Pre-trends test, event study, heterogeneity by sex and education

## Preferred Estimate Summary

- **Effect size**: 0.0748 (7.48 percentage points)
- **Standard error**: 0.0152
- **95% CI**: [0.045, 0.105]
- **Sample size**: 17,382
- **Interpretation**: DACA eligibility increased full-time employment probability by approximately 7.5 percentage points, statistically significant at p < 0.001
