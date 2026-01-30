# Run Log for DACA Replication Study (Replication 66)

## Overview
This document logs all commands, key decisions, and analytical choices made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Date
January 27, 2026

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Phase 1: Data Exploration and Understanding

### Step 1.1: Read Instructions
- Extracted text from `replication_instructions.docx` using python-docx
- Understood research design: Difference-in-differences comparing ages 26-30 (treatment) to ages 31-35 (control) at time of DACA implementation (June 2012)

### Step 1.2: Explore Data Files
Files available:
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Labelled version (not used)
- `data/acs_data_dict.txt` - Data dictionary

### Step 1.3: Data Dictionary Review
Read data dictionary to understand variable coding:
- IPUMS binary variables: 1 = No, 2 = Yes
- Created variables (FT, AFTER, ELIGIBLE): 0 = No, 1 = Yes
- SEX: 1 = Male, 2 = Female
- MARST: 1 = Married spouse present, 6 = Never married

### Step 1.4: Initial Data Exploration
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
```

Dataset characteristics:
- Shape: 17,382 observations x 105 variables
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- ELIGIBLE: 11,382 treatment (ages 26-30), 6,000 control (ages 31-35)
- AFTER: 9,527 pre (2008-2011), 7,855 post (2013-2016)
- FT: 11,283 full-time (65%), 6,099 not full-time (35%)

---

## Phase 2: Key Analytical Decisions

### Decision 1: Use of ELIGIBLE Variable
**Decision:** Use the provided ELIGIBLE variable as-is, per instructions.
**Rationale:** Instructions explicitly state to use this variable and not create own eligibility variable.

### Decision 2: Sample Definition
**Decision:** Use entire provided dataset without further sample restrictions.
**Rationale:** Instructions state "This entire file constitutes the intended analytic sample for your analysis; do not further limit the sample by dropping individuals on the basis of their characteristics."

### Decision 3: Treatment of Those Not in Labor Force
**Decision:** Keep individuals not in labor force as FT=0.
**Rationale:** Instructions state "Those not in the labor force are included, usually as 0 values; keep these individuals in your analysis."

### Decision 4: Primary Estimation Method
**Decision:** Use Weighted Least Squares (WLS) with PERWT as the preferred specification.
**Rationale:** ACS is a complex survey design. Using person weights (PERWT) produces estimates representative of the target population. This is standard practice for ACS analysis.

### Decision 5: Standard Errors
**Decision:** Use heteroskedasticity-robust (HC1) standard errors.
**Rationale:** Standard practice for DiD estimation; accounts for heteroskedasticity in the binary outcome.

### Decision 6: Control Variables
**Decision:** Include demographic controls: sex, age, marital status, number of children, and education.
**Rationale:** These variables are likely predictors of full-time employment and may improve precision. Age is included because it varies across cohorts within each group.

### Decision 7: Fixed Effects
**Decision:** Present models with state and year fixed effects as robustness checks, but use model without fixed effects as preferred.
**Rationale:** State fixed effects control for time-invariant state characteristics. Year fixed effects absorb the AFTER main effect, controlling for common time trends. The simpler weighted model with controls is more interpretable.

---

## Phase 3: Analysis Implementation

### Step 3.1: Create Analysis Script (`analysis.py`)
Main analysis file containing:
1. Data loading and exploration
2. Simple DiD calculation
3. Regression-based DiD (multiple specifications)
4. Parallel trends tests
5. Event study analysis
6. Heterogeneous effects
7. Robustness checks (probit, logit, clustered SE)

### Step 3.2: Run Analysis
```bash
python analysis.py
```

### Key Results Summary:

#### Simple DiD Calculation:
- Control Pre: 0.6697
- Control Post: 0.6449
- Treatment Pre: 0.6263
- Treatment Post: 0.6658
- **DiD = (0.6658 - 0.6263) - (0.6449 - 0.6697) = 0.0643**

#### Regression Results:
| Model | Coefficient | SE | p-value |
|-------|------------|-----|---------|
| Basic DiD | 0.0643 | 0.0153 | 0.0000 |
| + Demographics | 0.0558 | 0.0142 | 0.0001 |
| + State FE | 0.0559 | 0.0142 | 0.0001 |
| + Year FE | 0.0543 | 0.0141 | 0.0001 |
| State + Year FE | 0.0544 | 0.0142 | 0.0001 |
| Weighted Basic | 0.0748 | 0.0181 | 0.0000 |
| **Weighted + Controls** | **0.0648** | **0.0167** | **0.0001** |
| Clustered SE | 0.0643 | 0.0141 | 0.0000 |

#### Parallel Trends Test:
- F-statistic: 2.05
- p-value: 0.104
- **Conclusion:** Fail to reject null hypothesis of parallel trends (p > 0.05)

#### Heterogeneous Effects:
- Male: 0.062 (p=0.0003)
- Female: 0.045 (p=0.0513)
- Not Married: 0.076 (p=0.0006)
- Married: 0.059 (p=0.0061)
- High School: 0.048 (p=0.0075)
- Some College: 0.108 (p=0.0047)

### Step 3.3: Create Figures (`create_figures.py`)
Generated 6 figures:
1. Parallel trends plot
2. Event study plot
3. DiD visualization (bar chart)
4. Model comparison coefficient plot
5. Heterogeneous effects plot
6. Sample size by year

```bash
python create_figures.py
```

All figures saved to `figures/` directory in PNG and PDF formats.

---

## Phase 4: Report Generation

### Step 4.1: Write LaTeX Report
Created `replication_report_66.tex` with:
- Abstract
- Introduction
- Background on DACA
- Data and Sample description
- Empirical Methodology
- Results
- Robustness checks
- Discussion
- Conclusion
- Appendix with additional tables

### Step 4.2: Compile PDF
```bash
pdflatex -interaction=nonstopmode replication_report_66.tex
pdflatex -interaction=nonstopmode replication_report_66.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_66.tex  # Third pass for final
```

Output: `replication_report_66.pdf` (25 pages)

---

## Phase 5: Preferred Estimate

### Preferred Model Specification:
Weighted Difference-in-Differences with Demographic Controls (Model 9)

### Variables Used:
- **Outcome:** FT (full-time employment, 35+ hours/week)
- **Treatment:** ELIGIBLE (ages 26-30 in June 2012)
- **Period:** AFTER (2013-2016 = post-DACA)
- **Controls:** Female, Married, NCHILD, AGE, EDUC_SC, EDUC_2YR, EDUC_BA
- **Weights:** PERWT (ACS person weights)

### Preferred Estimate:
- **Effect Size:** 0.0648 (6.48 percentage points)
- **Standard Error:** 0.0167
- **95% Confidence Interval:** [0.032, 0.098]
- **P-value:** 0.0001
- **Sample Size:** 17,382

### Interpretation:
DACA eligibility is associated with a 6.48 percentage point increase in the probability of full-time employment among eligible individuals aged 26-30 in June 2012, relative to the control group aged 31-35, controlling for demographic characteristics and using survey weights.

---

## Files Generated

1. **analysis.py** - Main analysis script
2. **create_figures.py** - Visualization script
3. **analysis_results.json** - Key results in JSON format
4. **balance_table.csv** - Balance table data
5. **event_study_results.csv** - Event study coefficients
6. **ft_by_year.csv** - Full-time employment by year
7. **figures/** - Directory with 6 figures (PNG and PDF)
8. **replication_report_66.tex** - LaTeX source
9. **replication_report_66.pdf** - Final report (25 pages)
10. **run_log_66.md** - This log file

---

## Software and Packages Used

- **Python 3.x**
  - pandas
  - numpy
  - statsmodels
  - scipy
  - matplotlib

- **LaTeX**
  - pdflatex (MiKTeX distribution)

---

## Replication Instructions

To reproduce this analysis:

1. Ensure Python and required packages are installed
2. Ensure LaTeX (pdflatex) is installed
3. Navigate to the replication directory
4. Run: `python analysis.py`
5. Run: `python create_figures.py`
6. Run: `pdflatex replication_report_66.tex` (three times for cross-references)

---

## Notes

- No external data sources were used beyond the provided dataset
- All original IPUMS variable names were preserved per instructions
- Analysis uses the full provided sample without additional restrictions
- Standard errors are robust to heteroskedasticity
- Survey weights (PERWT) are used in the preferred specification
