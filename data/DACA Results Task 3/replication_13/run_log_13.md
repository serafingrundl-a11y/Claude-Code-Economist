# Replication Run Log - Session 13

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA analysis study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (usually working 35+ hours per week)?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at the time of policy implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at the time of policy implementation who would otherwise have been eligible
- **Identification Strategy**: Difference-in-Differences (DiD)
- **Pre-treatment Period**: 2008-2011
- **Post-treatment Period**: 2013-2016 (2012 omitted)
- **Outcome Variable**: FT (Full-time employment, 1 = works 35+ hours/week)

## Key Variables
- `ELIGIBLE`: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
- `AFTER`: 1 = post-DACA period (2013-2016), 0 = pre-DACA period (2008-2011)
- `FT`: 1 = full-time employment, 0 = not full-time employment
- `PERWT`: Person weight for survey representativeness

## Session Log

### Step 1: Read Instructions and Data Dictionary
- Read replication_instructions.docx
- Reviewed acs_data_dict.txt for variable coding
- Confirmed data structure: ACS repeated cross-sectional data 2008-2016 (excluding 2012)

### Step 2: Data Exploration
- Data file: prepared_data_numeric_version.csv
- Sample size: 17,382 observations (plus header)
- Variables include demographics, labor force status, state policies

### Step 3: Analysis Plan
1. Load and clean data
2. Compute summary statistics by treatment/control and before/after
3. Estimate basic DiD model: FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
4. Estimate DiD with covariates for robustness
5. Check parallel trends assumption
6. Generate visualizations
7. Write comprehensive report

### Step 4: Analytical Decisions

#### Decision 1: Use Survey Weights
- Rationale: ACS is a complex survey design; using PERWT ensures representative estimates

#### Decision 2: Cluster Standard Errors at State Level
- Rationale: Individuals within states may have correlated errors due to state-level policies and labor markets

#### Decision 3: Include Covariates
- Demographics: Age, Sex, Education
- Family characteristics: Family size, Marital status
- Geographic: State fixed effects
- Rationale: Improve precision and control for compositional changes over time

### Step 5: Analysis Execution

#### Command: Run analysis.py
```
cd "C:\Users\seraf\DACA Results Task 3\replication_13" && python analysis.py
```

#### Key Results:
- **Sample Size**: 17,382 observations
- **Treatment Group (Eligible 26-30)**: 11,382 observations
- **Control Group (31-35)**: 6,000 observations
- **Pre-DACA Period**: 9,527 observations
- **Post-DACA Period**: 7,855 observations

#### Descriptive Statistics (Weighted):
| Group | Pre-DACA FT Rate | Post-DACA FT Rate | Change |
|-------|------------------|-------------------|--------|
| Eligible (26-30) | 63.69% | 68.60% | +4.91 pp |
| Control (31-35) | 68.86% | 66.29% | -2.57 pp |
| **DiD Estimate** | | | **+7.48 pp** |

#### Regression Results Summary:
| Model | DiD Estimate | SE | p-value |
|-------|-------------|-----|---------|
| Basic OLS | 0.0643 | 0.0153 | <0.001 |
| Weighted (WLS) | 0.0748 | 0.0152 | <0.001 |
| WLS + Demographics | 0.0627 | 0.0142 | <0.001 |
| WLS + State FE | 0.0737 | 0.0152 | <0.001 |
| Full Model | 0.0624 | 0.0142 | <0.001 |
| Basic + Clustered SE | 0.0748 | 0.0203 | <0.001 |
| Full + Clustered SE | 0.0624 | 0.0218 | 0.004 |

#### Preferred Specification:
- **Model**: Full model with demographics, state FE, and clustered standard errors
- **DiD Estimate**: 0.0624 (6.24 percentage points)
- **Standard Error**: 0.0218 (clustered at state level)
- **95% CI**: [0.0197, 0.1052]
- **p-value**: 0.0042

#### Parallel Trends Assessment:
- Event study shows some pre-treatment differences
- Joint F-test for pre-trends: F = 3.02, p = 0.029
- Pre-treatment coefficients are negative (eligible group had relatively lower FT rates)
- Post-treatment effects emerge in 2016 (coef = 0.074, p = 0.018)

#### Heterogeneity Analysis:
- Males: DiD = 0.072 (SE = 0.017)
- Females: DiD = 0.053 (SE = 0.023)

#### Robustness Checks:
- Trimmed weights (1st-99th percentile): DiD = 0.065 (SE = 0.015)
- Unweighted analysis: DiD = 0.064 (SE = 0.015)

### Step 6: Figures Generated
1. figure1_parallel_trends.png - FT rates by group over time
2. figure2_event_study.png - Event study coefficients
3. figure3_did_visual.png - DiD visualization
4. figure4_coefficient_comparison.png - Coefficient comparison across models

### Step 7: Writing Replication Report
- Creating comprehensive LaTeX report (~20 pages)
- Including all tables, figures, and methodology details

### Step 8: Final Compilation

#### LaTeX Compilation
```
pdflatex -interaction=nonstopmode replication_report_13.tex
```
- Ran 3 times for proper cross-references
- Final PDF: 21 pages, 867,797 bytes

## Summary of Deliverables

### Required Output Files:
1. **replication_report_13.tex** - LaTeX source file (32,904 bytes)
2. **replication_report_13.pdf** - Compiled PDF report (867,797 bytes, 21 pages)
3. **run_log_13.md** - This run log file

### Additional Generated Files:
- figure1_parallel_trends.png - Parallel trends visualization
- figure2_event_study.png - Event study coefficients
- figure3_did_visual.png - DiD visualization
- figure4_coefficient_comparison.png - Coefficient comparison
- analysis.py - Python analysis script
- summary_statistics.csv - Descriptive statistics
- yearly_ft_rates.csv - Yearly FT rates
- event_study_results.csv - Event study results
- model_results.txt - Full regression output

## Key Findings Summary

| Metric | Value |
|--------|-------|
| Sample Size | 17,382 |
| Treatment Group | 11,382 (Eligible, ages 26-30) |
| Control Group | 6,000 (ages 31-35) |
| Pre-DACA Period | 2008-2011 |
| Post-DACA Period | 2013-2016 |
| **Preferred DiD Estimate** | **0.0624 (6.24 pp)** |
| Standard Error (Clustered) | 0.0218 |
| 95% Confidence Interval | [0.020, 0.105] |
| p-value | 0.004 |

## Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment. The preferred estimate suggests DACA eligibility increased the probability of full-time employment by approximately 6.2 percentage points. This effect is robust across multiple model specifications and represents a meaningful improvement in labor market outcomes for eligible individuals.

---
*Log completed: Session 13*

