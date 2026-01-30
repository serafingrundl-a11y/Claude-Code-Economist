# Replication Run Log - Study ID: 82

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA employment effects study.

---

## Session Information
- **Date**: 2026-01-27
- **Analysis Software**: Python 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX Compiler**: pdflatex (MiKTeX)

---

## Step 1: Read Replication Instructions

**Action**: Read `replication_instructions.docx` to understand the research task.

**Key Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment?

**Design Specification**:
- Treatment group: Individuals aged 26-30 at DACA implementation (June 15, 2012)
- Control group: Individuals aged 31-35 at DACA implementation (would be eligible but for age)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded)
- Outcome: FT (full-time employment, =1 if usually working 35+ hours/week)

---

## Step 2: Data Exploration

**Data Files**:
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/acs_data_dict.txt` - Data dictionary

**Initial Data Check**:
```python
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(df.shape)  # (17382, 105)
```

**Key Variable Summary**:
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Outcome variable: FT (0=not full-time, 1=full-time)

---

## Step 3: Key Analytical Decisions

### Decision 1: Estimation Method
**Choice**: Linear Probability Model (LPM) estimated via Weighted Least Squares (WLS)
**Rationale**:
- LPM provides easily interpretable coefficients as probability changes
- WLS with PERWT weights ensures population-representative estimates
- Standard approach in labor economics for binary outcomes

### Decision 2: Standard Error Calculation
**Choice**: Heteroskedasticity-robust standard errors (HC1)
**Rationale**:
- LPM inherently has heteroskedastic errors
- Robust SE provide valid inference
- Also computed state-clustered SE as robustness check

### Decision 3: Fixed Effects
**Choice**: Include year and state fixed effects in preferred specification
**Rationale**:
- Year FE control for common time trends affecting all individuals
- State FE control for time-invariant state-level differences
- Standard in DiD designs with multiple periods and geographic units

### Decision 4: Control Variables
**Choice**: Include age, age-squared, gender, marital status, number of children, and education
**Rationale**:
- These variables are standard demographic controls in employment regressions
- May differ between treatment and control groups and relate to outcome
- Note: Including controls substantially changed results

### Decision 5: Variable Coding
**Transformations Made**:
- `TREAT_POST = ELIGIBLE * AFTER` (DiD interaction)
- `FEMALE = 1 if SEX == 2` (IPUMS codes male=1, female=2)
- `MARRIED = 1 if MARST in [1, 2]` (married, spouse present or absent)
- `AGE_SQ = AGE^2` (quadratic age term)

---

## Step 4: Analysis Execution

### Command: Run Main Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_82"
python analysis.py
```

### Models Estimated:

1. **Model 1**: Basic OLS DiD (unweighted)
   - Coefficient: 0.0643 (SE: 0.015, p<0.001)

2. **Model 2**: WLS DiD with PERWT weights
   - Coefficient: 0.0748 (SE: 0.018, p<0.001)

3. **Model 3**: WLS with Year FE
   - Coefficient: 0.0721 (SE: 0.018, p<0.001)

4. **Model 4**: WLS with Year FE + Demographics
   - Coefficient: 0.0215 (SE: 0.025, p=0.387)

5. **Model 5**: Full Model (Year FE + State FE + Demographics + Education) [PREFERRED]
   - Coefficient: 0.0201 (SE: 0.025, p=0.416)

6. **Model 6**: Full Model with State-Clustered SE
   - Coefficient: 0.0201 (SE: 0.022, p=0.366)

### Parallel Trends Test
- Pre-trend interaction coefficient: 0.0174 (SE: 0.011, p=0.113)
- Result: No significant differential pre-trend detected

### Event Study Results (reference: 2011):
- 2008: -0.0681 (SE: 0.035, p<0.10)
- 2009: -0.0499 (SE: 0.036, p>0.10)
- 2010: -0.0821 (SE: 0.036, p<0.05)
- 2013: +0.0158 (SE: 0.038, p>0.10)
- 2014: +0.0000 (SE: 0.038, p>0.10)
- 2015: +0.0014 (SE: 0.038, p>0.10)
- 2016: +0.0741 (SE: 0.038, p<0.10)

---

## Step 5: Figure Generation

### Command: Create Figures
```bash
python create_figures.py
```

### Figures Created:
1. `figure1_parallel_trends.pdf/png` - FT trends by treatment status over time
2. `figure2_event_study.pdf/png` - Event study coefficients with CIs
3. `figure3_did_visualization.pdf/png` - 2x2 DiD visualization
4. `figure4_sample_distribution.pdf/png` - Age and year distributions
5. `figure5_heterogeneity_gender.pdf/png` - Trends by gender

---

## Step 6: Report Compilation

### Command: Compile LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_82.tex
pdflatex -interaction=nonstopmode replication_report_82.tex
pdflatex -interaction=nonstopmode replication_report_82.tex
```

**Note**: Three passes needed for proper reference resolution.

**Output**: `replication_report_82.pdf` (19 pages)

---

## Step 7: Final Outputs

### Required Deliverables:
1. `replication_report_82.tex` - LaTeX source file
2. `replication_report_82.pdf` - Compiled PDF report (19 pages)
3. `run_log_82.md` - This run log

### Supporting Files:
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `analysis_results.pkl` - Saved analysis results
- `figure1_parallel_trends.pdf` - Parallel trends figure
- `figure2_event_study.pdf` - Event study figure
- `figure3_did_visualization.pdf` - DiD visualization
- `figure4_sample_distribution.pdf` - Sample distribution figure
- `figure5_heterogeneity_gender.pdf` - Gender heterogeneity figure

---

## Preferred Estimate Summary

| Metric | Value |
|--------|-------|
| Effect Size | 0.0201 (2.01 percentage points) |
| Standard Error (Robust) | 0.0247 |
| Standard Error (Clustered) | 0.0222 |
| 95% CI (Robust) | [-0.028, 0.069] |
| 95% CI (Clustered) | [-0.024, 0.064] |
| p-value (Robust) | 0.416 |
| p-value (Clustered) | 0.366 |
| Sample Size | 17,382 |

---

## Key Findings

1. **Simple DiD estimates are positive and significant**: Without controls, DACA eligibility associated with 6.4-7.5 pp increase in FT employment.

2. **Effects become insignificant with controls**: After controlling for demographics, education, and state FE, effect drops to ~2 pp and becomes statistically insignificant.

3. **Parallel trends assumption tentatively supported**: Formal test does not reject parallel pre-trends (p=0.113).

4. **Heterogeneous effects**: Larger effects for males (7.0 pp) vs females (4.9 pp); larger effects for more educated individuals.

---

## Session End

All deliverables verified present in `C:\Users\seraf\DACA Results Task 3\replication_82\`:
- [x] replication_report_82.tex
- [x] replication_report_82.pdf
- [x] run_log_82.md
