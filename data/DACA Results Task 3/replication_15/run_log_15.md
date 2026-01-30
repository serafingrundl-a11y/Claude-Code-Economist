# DACA Replication Study - Run Log

## Study Information
- **Replication ID**: 15
- **Date**: January 27, 2026
- **Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the US

---

## Data Files Used

| File | Description | Location |
|------|-------------|----------|
| `prepared_data_numeric_version.csv` | Main analysis dataset | `data/` |
| `prepared_data_labelled_version.csv` | Labelled version (not used) | `data/` |
| `acs_data_dict.txt` | Variable documentation | `data/` |

---

## Session Log

### Step 1: Read Replication Instructions
- **Action**: Extracted content from `replication_instructions.docx`
- **Method**: Used unzip to extract XML content from docx file
- **Key specifications identified**:
  - Treatment group: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
  - Control group: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
  - Outcome: Full-time employment (FT=1 if usually working 35+ hours/week)
  - Pre-period: 2008-2011 (AFTER=0)
  - Post-period: 2013-2016 (AFTER=1)
  - Year 2012 excluded from data

### Step 2: Data Exploration
- **Command**: `head -5 prepared_data_numeric_version.csv`
- **Command**: `wc -l prepared_data_numeric_version.csv`
- **Findings**:
  - Total observations: 17,382
  - Variables: 105 columns
  - Key variables present: YEAR, ELIGIBLE, AFTER, FT, PERWT, AGE_IN_JUNE_2012

### Step 3: Software Environment Check
- **Python version**: 3.14.2
- **Key packages**:
  - pandas: 2.3.3
  - statsmodels: 0.14.6
  - scipy: 1.16.3
  - numpy: 2.3.5
  - matplotlib: 3.10.8

### Step 4: Create Analysis Script
- **File created**: `analysis.py`
- **Contents**:
  - Data loading and exploration
  - Summary statistics by group
  - Basic DiD calculation
  - Multiple regression specifications
  - Parallel trends assessment
  - Event study analysis
  - Subgroup analysis
  - Robustness checks
  - Visualization generation

### Step 5: Execute Analysis
- **Command**: `python analysis.py`
- **Runtime**: Completed successfully
- **Outputs generated**:
  - `regression_results.csv` - Summary of all model coefficients
  - `figure1_parallel_trends.png` - Time series by group
  - `figure2_event_study.png` - Dynamic treatment effects
  - `figure3_did_illustration.png` - DiD visualization
  - `figure4_ft_distribution.png` - Employment distribution
  - `figure5_coefficient_comparison.png` - Model comparison
  - `figure6_hours_distribution.png` - Hours worked histogram

### Step 6: Create LaTeX Report
- **File created**: `replication_report_15.tex`
- **Sections**:
  1. Introduction
  2. Background
  3. Data
  4. Methodology
  5. Results
  6. Discussion
  7. Conclusion
  8. Appendices (additional figures and tables)

### Step 7: Compile PDF
- **Command**: `pdflatex -interaction=nonstopmode replication_report_15.tex` (run 3 times)
- **Output**: `replication_report_15.pdf` (29 pages)

---

## Key Analytical Decisions

### 1. Model Specification Choice
**Decision**: Use OLS linear probability model for DiD
**Rationale**:
- Standard approach for DiD with binary outcomes
- Coefficients directly interpretable as percentage point changes
- Easier to include interaction terms
- Consistent with common practice in applied economics

### 2. Preferred Estimate Selection
**Decision**: Model 5 (DiD with covariates: sex, marital status, education)
**Rationale**:
- Controls for key demographic confounders
- Uses robust (HC1) standard errors for valid inference
- Does not over-control with excessive fixed effects
- Balances bias reduction with precision

### 3. Standard Error Treatment
**Decision**: Report multiple SE specifications (classical, robust HC1, clustered by state)
**Rationale**:
- Heteroskedasticity likely present (binary outcome)
- Clustering accounts for within-state correlation
- Robustness across SE specifications increases credibility

### 4. Treatment of Non-Workers
**Decision**: Keep individuals not in labor force with FT=0
**Rationale**:
- Explicitly required by instructions
- Captures extensive margin effects (labor force participation)
- More comprehensive measure of employment impact

### 5. Covariate Selection
**Decision**: Include sex, marital status, and education
**Rationale**:
- Strong predictors of employment outcomes
- May differ between treatment and control groups
- Available in data with minimal missing values
- Consistent with prior literature

### 6. Parallel Trends Assessment
**Decision**: Use multiple approaches (visual, statistical test, event study)
**Rationale**:
- No single test is definitive
- Visual inspection informative but subjective
- Formal test has limited power
- Event study shows dynamic pattern

---

## Key Results Summary

### Primary Finding
- **DiD Estimate (Preferred)**: 0.0537 (5.37 percentage points)
- **Standard Error**: 0.0142
- **95% CI**: [0.0259, 0.0814]
- **p-value**: 0.0001

### Interpretation
DACA eligibility is associated with a 5.4 percentage point increase in the probability of full-time employment for the treatment group (ages 26-30 at DACA implementation) relative to the control group (ages 31-35), after accounting for general time trends and demographic characteristics.

### Robustness
- Effect is statistically significant across all 7 model specifications
- Coefficient ranges from 0.0523 to 0.0645 across models
- Results robust to:
  - Different standard error adjustments
  - Year and state fixed effects
  - Person-weighted estimation

### Parallel Trends
- Pre-trends test p-value: 0.098 (not significant at 5%)
- Visual inspection shows roughly parallel trends 2008-2011
- Some evidence of pre-treatment differences in event study
- Conclusion: Assumption marginally supported, interpret with caution

---

## Files Generated

| File | Type | Description |
|------|------|-------------|
| `analysis.py` | Python | Main analysis script |
| `regression_results.csv` | CSV | Summary of regression coefficients |
| `figure1_parallel_trends.png` | PNG | Time series plot |
| `figure2_event_study.png` | PNG | Event study plot |
| `figure3_did_illustration.png` | PNG | DiD visualization |
| `figure4_ft_distribution.png` | PNG | Employment distribution |
| `figure5_coefficient_comparison.png` | PNG | Model comparison |
| `figure6_hours_distribution.png` | PNG | Hours distribution |
| `replication_report_15.tex` | LaTeX | Full report source |
| `replication_report_15.pdf` | PDF | Final report (29 pages) |
| `run_log_15.md` | Markdown | This log file |

---

## Sample Size Details

| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| **Total** | 9,527 | 7,855 | 17,382 |

---

## Technical Notes

1. **Data Format**: CSV with numeric coding for most variables
2. **Missing Values**: `NA` strings in some columns, handled by pandas
3. **Weights**: PERWT person weights available, used in weighted model
4. **Binary Variables**:
   - IPUMS variables: 1=No, 2=Yes
   - Custom variables (FT, AFTER, ELIGIBLE): 0=No, 1=Yes

---

## Quality Checks Performed

1. Verified sample sizes match expected distributions
2. Confirmed ELIGIBLE and AFTER variables correctly coded
3. Checked that FT rates are reasonable (60-70% range)
4. Verified no observations from 2012 in dataset
5. Confirmed regression coefficients consistent across specifications
6. Validated that figures render correctly in PDF

---

## Completion Status

All required deliverables have been generated:
- [x] `replication_report_15.tex` - LaTeX source file
- [x] `replication_report_15.pdf` - Compiled PDF report (29 pages)
- [x] `run_log_15.md` - This documentation file

**Analysis completed successfully.**
