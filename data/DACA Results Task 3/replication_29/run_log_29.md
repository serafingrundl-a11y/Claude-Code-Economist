# Run Log: DACA Replication Study (ID: 29)

## Overview
This document logs all commands and key decisions made during the replication of the DACA employment effects study.

---

## Session Information
- **Date**: January 27, 2026
- **Replication ID**: 29
- **Working Directory**: `C:\Users\seraf\DACA Results Task 3\replication_29`

---

## Step 1: Data Exploration

### Commands Executed
```bash
# Preview data structure
head -50 data/prepared_data_labelled_version.csv
head -50 data/prepared_data_numeric_version.csv
```

### Key Findings
- Total observations: 17,382
- Years in data: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 omitted)
- Key variables present: FT, ELIGIBLE, AFTER, demographic controls

---

## Step 2: Data Dictionary Review

### File Reviewed
- `data/acs_data_dict.txt`

### Key Variable Definitions
| Variable | Description |
|----------|-------------|
| ELIGIBLE | 1 = Treatment group (ages 26-30 at June 2012), 0 = Control (ages 31-35) |
| AFTER | 1 = Post-DACA (2013-2016), 0 = Pre-DACA (2008-2011) |
| FT | 1 = Full-time employment (35+ hrs/week), 0 = Not full-time |
| SEX | 1 = Male, 2 = Female (IPUMS coding) |
| MARST | 1 = Married spouse present |
| EDUC | Education level (numeric codes) |
| STATEFIP | State FIPS code |
| PERWT | Person weight for survey sampling |

---

## Step 3: Analysis Script Development

### File Created
- `analysis.py`

### Key Design Decisions

1. **Estimation Strategy**: Difference-in-Differences (DiD)
   - Treatment: ELIGIBLE = 1 (ages 26-30 at DACA implementation)
   - Control: ELIGIBLE = 0 (ages 31-35 at DACA implementation)
   - Pre-period: 2008-2011
   - Post-period: 2013-2016

2. **Model Specifications**:
   - Model 1: Basic DiD (no covariates)
   - Model 2: DiD + demographic controls (age, gender, marital status, education)
   - Model 3: DiD + state fixed effects
   - Model 4: DiD + year fixed effects
   - Model 5: DiD + state and year fixed effects
   - Model 6: DiD with clustered standard errors (by state)
   - Model 7: Weighted regression using PERWT

3. **Covariate Construction**:
   - FEMALE: Binary indicator derived from SEX == 2
   - MARRIED: Binary indicator derived from MARST == 1
   - HS_OR_MORE: Binary indicator for EDUC >= 6 (grade 12 or higher)

4. **Robustness Checks**:
   - Heterogeneity by gender
   - Event study (year-by-year effects)
   - Placebo test using pre-period only
   - Logit model with marginal effects

---

## Step 4: Analysis Execution

### Command
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_29"
python analysis.py
```

### Output Files Generated
| File | Description |
|------|-------------|
| `analysis_results.csv` | Summary statistics and key results |
| `model_comparison.csv` | Comparison of all model specifications |
| `event_study_results.csv` | Year-by-year event study coefficients |
| `event_study.png` | Event study visualization |
| `parallel_trends.png` | Parallel trends plot |
| `did_visualization.png` | DiD graphical illustration |

---

## Step 5: Key Results

### Main Finding
**Preferred Estimate** (Model 6: State + Year FE with Clustered SE):
- DiD Coefficient: **0.0541** (5.41 percentage points)
- Standard Error: 0.0150 (clustered by state)
- 95% CI: [0.0247, 0.0835]
- p-value: 0.0003

### Interpretation
DACA eligibility is associated with a 5.4 percentage point increase in the probability of full-time employment among Mexican-born, Hispanic individuals aged 26-30 compared to those aged 31-35.

### Sample Sizes
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treatment (26-30) | 6,233 | 5,149 | 11,382 |
| **Total** | 9,527 | 7,855 | **17,382** |

### Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control | 0.6697 | 0.6449 | -0.0248 |
| Treatment | 0.6263 | 0.6658 | +0.0394 |
| **DiD** | | | **+0.0643** |

### Model Comparison
| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD | 0.0643 | 0.0153 | <0.001 |
| + Demographics | 0.0555 | 0.0143 | <0.001 |
| + State FE | 0.0556 | 0.0143 | <0.001 |
| + Year FE | 0.0540 | 0.0143 | <0.001 |
| + State & Year FE | 0.0541 | 0.0143 | <0.001 |
| Clustered SE | 0.0541 | 0.0150 | <0.001 |
| Weighted | 0.0648 | 0.0142 | <0.001 |

### Robustness Checks
1. **By Gender**:
   - Male: 0.0522 (SE=0.0173, p=0.003)
   - Female: 0.0442 (SE=0.0229, p=0.054)

2. **Placebo Test**:
   - Coefficient: 0.0164 (p=0.400)
   - Supports parallel trends assumption

3. **Logit Marginal Effect**: 0.0614 (p<0.001)

---

## Step 6: Report Generation

### LaTeX Report
- **File**: `replication_report_29.tex`
- **Pages**: ~19 pages
- **Sections**:
  1. Introduction
  2. Data and Sample
  3. Empirical Strategy
  4. Results
  5. Robustness Checks
  6. Discussion
  7. Conclusion
  8. Technical Appendix

### PDF Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_29.tex
pdflatex -interaction=nonstopmode replication_report_29.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_29.tex  # Third pass for refs
```

### Output
- **File**: `replication_report_29.pdf`
- **Size**: ~522 KB
- **Pages**: 19

---

## Step 7: Files Produced

### Required Deliverables
| File | Status |
|------|--------|
| `replication_report_29.tex` | Created |
| `replication_report_29.pdf` | Created |
| `run_log_29.md` | Created |

### Supporting Files
| File | Description |
|------|-------------|
| `analysis.py` | Python analysis script |
| `analysis_results.csv` | Summary results |
| `model_comparison.csv` | Model comparison table |
| `event_study_results.csv` | Event study data |
| `event_study.png` | Event study figure |
| `parallel_trends.png` | Trends figure |
| `did_visualization.png` | DiD illustration |

---

## Key Analytical Decisions Summary

1. **Sample Definition**: Used entire provided sample (N=17,382); did not drop any observations

2. **Treatment Definition**: Used pre-constructed ELIGIBLE variable; ages 26-30 = treatment, ages 31-35 = control

3. **Outcome Variable**: Full-time employment (FT); includes those not in labor force as zeros

4. **Control Variables**:
   - Age (continuous)
   - Female (binary)
   - Married (binary)
   - High school or more education (binary)

5. **Fixed Effects**: State and year fixed effects in preferred specification

6. **Standard Errors**: Clustered at state level to account for within-state correlation

7. **Weights**: Reported both weighted and unweighted estimates; preferred unweighted

8. **Preferred Estimate**: 0.0541 (SE=0.015), representing a 5.4 percentage point increase in full-time employment probability

---

## Software Environment
- Python 3.x
- pandas
- numpy
- statsmodels
- matplotlib
- LaTeX (MiKTeX)

---

## Completion Time
Analysis and report generation completed in single session.

---

*End of Run Log*
