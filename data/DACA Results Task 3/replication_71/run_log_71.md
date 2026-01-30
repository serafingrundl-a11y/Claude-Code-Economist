# Replication Run Log - ID 71

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment.

---

## Session Information
- **Date**: January 27, 2026
- **Working Directory**: `C:\Users\seraf\DACA Results Task 3\replication_71`
- **Analysis Software**: Python 3.x with pandas, numpy, statsmodels, scipy
- **Document Preparation**: LaTeX (pdflatex via MiKTeX)

---

## Step 1: Read Replication Instructions

### Command
```bash
python -c "from docx import Document; d = Document('replication_instructions.docx'); print('\n'.join([p.text for p in d.paragraphs]))"
```

### Key Information Extracted
- **Research Question**: Effect of DACA eligibility on full-time employment (35+ hours/week)
- **Treatment Group**: Ages 26-30 at time of policy (June 2012), ELIGIBLE = 1
- **Control Group**: Ages 31-35 at time of policy (June 2012), ELIGIBLE = 0
- **Pre-Period**: 2008-2011 (AFTER = 0)
- **Post-Period**: 2013-2016 (AFTER = 1)
- **Method**: Difference-in-Differences
- **Sample**: Ethnically Hispanic-Mexican, Mexican-born individuals

---

## Step 2: Explore Data Files

### Commands
```bash
# List files in directory
ls -la

# Preview data
head -5 data/prepared_data_labelled_version.csv
head -5 data/prepared_data_numeric_version.csv
wc -l data/prepared_data_numeric_version.csv
```

### Findings
- Total observations: 17,382
- Variables: 105 columns
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Key pre-constructed variables: FT, ELIGIBLE, AFTER

---

## Step 3: Data Analysis (Python)

### Analysis Script: `analysis.py`

Key components:
1. Data loading and exploration
2. Simple DiD calculation
3. OLS regression with robust standard errors
4. Model with covariates
5. Model with state fixed effects
6. Event study analysis
7. Pre-trends testing
8. Heterogeneity analysis by sex
9. Weighted analysis

### Command
```bash
python analysis.py
```

### Key Decisions Made

#### 1. Estimation Method
- **Decision**: Linear Probability Model (OLS) with robust standard errors
- **Rationale**: Simple, interpretable, and appropriate for DiD estimation. LPM coefficients directly interpretable as percentage point changes.

#### 2. Standard Error Specification
- **Decision**: Heteroskedasticity-robust standard errors (HC1) as primary specification
- **Rationale**: Outcome is binary (FT = 0 or 1), so heteroskedasticity is expected. Also tested clustered SE at state level for robustness.

#### 3. Sample Selection
- **Decision**: Use full provided sample without additional restrictions
- **Rationale**: Instructions explicitly stated the entire file is the intended analytic sample and not to drop individuals based on characteristics.

#### 4. Treatment of Non-Labor Force Participants
- **Decision**: Keep in sample with FT = 0
- **Rationale**: Instructions stated those not in the labor force are usually coded as 0 and to keep them in analysis.

#### 5. Covariate Selection for Extended Models
- **Decision**: Include sex (female), marital status (married), presence of children (has_children), and education level (categorical dummies)
- **Rationale**: These are theoretically relevant predictors of employment and available in the data.

#### 6. Reference Category for Education
- **Decision**: Less than High School as reference (though only 9 observations)
- **Rationale**: Standard practice; all other categories compared to lowest education level.

#### 7. Weighting
- **Decision**: Primary analysis unweighted; supplementary weighted analysis using PERWT
- **Rationale**: Both approaches defensible; unweighted estimates the average treatment effect in the sample, weighted estimates population-level effect.

---

## Step 4: Create Figures

### Script: `create_figures.py`

### Command
```bash
python create_figures.py
```

### Figures Generated
1. `figure1_trends.pdf` - Full-time employment trends over time
2. `figure2_eventstudy.pdf` - Event study coefficients
3. `figure3_did.pdf` - DiD visualization
4. `figure4_bysex.pdf` - Trends by sex
5. `figure5_descriptive.pdf` - Sample composition

---

## Step 5: LaTeX Report Compilation

### Commands
```bash
pdflatex -interaction=nonstopmode replication_report_71.tex
pdflatex -interaction=nonstopmode replication_report_71.tex  # Second pass for cross-refs
```

### Output
- `replication_report_71.pdf` (23 pages)

---

## Main Results Summary

### Sample Sizes
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treated (26-30) | 6,233 | 5,149 | 11,382 |
| **Total** | 9,527 | 7,855 | 17,382 |

### Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control | 0.670 | 0.645 | -0.025 |
| Treated | 0.626 | 0.666 | +0.039 |
| **DiD** | | | **+0.064** |

### Preferred Estimate (Basic DiD with Robust SE)
- **Coefficient**: 0.0643
- **Standard Error**: 0.0153
- **t-statistic**: 4.21
- **p-value**: < 0.0001
- **95% CI**: [0.034, 0.094]

### Robustness Checks
| Specification | DiD Estimate | SE | p-value |
|---------------|--------------|-----|---------|
| Basic | 0.064 | 0.015 | <0.001 |
| With Covariates | 0.052 | 0.014 | <0.001 |
| With State FE | 0.064 | 0.015 | <0.001 |
| Full Model | 0.052 | 0.014 | <0.001 |
| Clustered SE | 0.064 | 0.014 | <0.001 |
| Weighted | 0.075 | 0.018 | <0.001 |

### Pre-Trends Test
- Linear pre-trend interaction coefficient: 0.015
- SE: 0.009
- p-value: 0.098
- **Conclusion**: No significant differential pre-trends at 5% level

---

## Files Generated

1. `analysis.py` - Main analysis script
2. `create_figures.py` - Figure generation script
3. `year_means.csv` - Year-by-year means for plotting
4. `regression_results.csv` - Regression results summary
5. `figure1_trends.png/pdf` - Trends figure
6. `figure2_eventstudy.png/pdf` - Event study figure
7. `figure3_did.png/pdf` - DiD visualization
8. `figure4_bysex.png/pdf` - Sex heterogeneity figure
9. `figure5_descriptive.png/pdf` - Descriptive figure
10. `replication_report_71.tex` - LaTeX source
11. `replication_report_71.pdf` - Final report (23 pages)
12. `run_log_71.md` - This log file

---

## Analytical Decisions Summary

1. **Model**: Linear probability model (OLS) for difference-in-differences
2. **Standard Errors**: Heteroskedasticity-robust (HC1) as primary
3. **Sample**: Full provided sample used as-is
4. **Pre-ELIGIBLE Variable**: Used as provided; did not create alternative
5. **Covariates**: Sex, marital status, children, education included in robustness
6. **Fixed Effects**: State FE tested for robustness
7. **Clustering**: State-level clustering tested for robustness
8. **Weighting**: Unweighted primary; PERWT-weighted for sensitivity

---

## Conclusion

The replication analysis finds a statistically significant positive effect of DACA eligibility on full-time employment of approximately 6.4 percentage points. This result is robust across multiple specifications and supported by event study evidence consistent with the parallel trends assumption.
