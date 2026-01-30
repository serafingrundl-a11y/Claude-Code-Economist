# Replication Run Log - Replication 17

## Date: January 27, 2026

---

## Task Overview
Replicate analysis of the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

### Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (outcome) for eligible Hispanic-Mexican Mexican-born individuals?

### Identification Strategy
- **Treatment Group**: Ages 26-30 at time of policy implementation (June 2012)
- **Control Group**: Ages 31-35 at time of policy implementation (otherwise eligible)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded)

---

## Step 1: Data Exploration

### Key Variables (from instructions):
- `ELIGIBLE`: 1 = treatment group (ages 26-30), 0 = comparison group (ages 31-35)
- `FT`: 1 = full-time work (35+ hours/week), 0 = not full-time
- `AFTER`: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
- `PERWT`: Person weight for population estimates

### Data Files:
- `prepared_data_numeric_version.csv` - Main analysis file
- `prepared_data_labelled_version.csv` - Labelled version for reference
- `acs_data_dict.txt` - Data dictionary

### Initial Data Inspection:
- Data appears to contain ACS data from 2008-2016 (excluding 2012)
- Sample restricted to Hispanic-Mexican Mexican-born individuals
- Contains treatment/control indicators and outcome variable pre-constructed

---

## Step 2: Analysis Plan

1. Load data and verify structure
2. Generate summary statistics by treatment/control and before/after
3. Compute simple difference-in-differences estimate
4. Run regression-based DiD with controls
5. Check parallel trends assumption
6. Conduct robustness checks
7. Create visualizations
8. Write up results

---

## Commands and Decisions Log

### Command 1: Data Loading and Exploration
```python
python analysis_17.py
```
- Loaded `prepared_data_numeric_version.csv`
- Verified 17,382 observations across years 2008-2011 and 2013-2016
- Confirmed key variables: ELIGIBLE, AFTER, FT, PERWT

### Key Data Findings:
- Total observations: 17,382
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (AFTER=0): 9,527 observations
- Post-period (AFTER=1): 7,855 observations
- Weighted sample size: 2,416,349

---

## Step 3: Analysis Results

### Simple DiD Calculation:
| Group | Pre-DACA | Post-DACA | Difference |
|-------|----------|-----------|------------|
| Treatment (26-30) | 0.6369 | 0.6860 | +0.0491 |
| Control (31-35) | 0.6886 | 0.6629 | -0.0257 |

**DiD Estimate = 0.0491 - (-0.0257) = 0.0748**

### Regression Results Summary:

| Model | DiD Estimate | SE | P-value |
|-------|--------------|-----|---------|
| 1. Basic DiD | 0.0748 | 0.0181 | <0.0001 |
| 2. DiD + Demographics (PREFERRED) | 0.0612 | 0.0167 | 0.0003 |
| 3. DiD + Year FE | 0.0583 | 0.0167 | 0.0005 |
| 4. DiD + Year & State FE | 0.0577 | 0.0166 | 0.0005 |
| 5. Unweighted | 0.0523 | 0.0142 | 0.0002 |
| 6. Clustered SE | 0.0612 | 0.0213 | 0.0041 |

### Parallel Trends Test:
- Pre-treatment interaction coefficient: 0.0174
- P-value: 0.1133
- **Conclusion**: Parallel trends assumption satisfied

### Robustness Checks:
1. **Logistic regression**: Log-odds coefficient = 0.2746, p < 0.001
2. **By gender**: Males (0.0716, p<0.001), Females (0.0527, p=0.061)
3. **Clustered SEs**: 95% CI [0.0194, 0.1030]

---

## Step 4: Key Decisions

1. **Choice of Preferred Model**: Model 2 (DiD with demographic controls) selected as preferred specification because it balances parsimony with important confounders while maintaining interpretability.

2. **Weighting**: Used person weights (PERWT) to make estimates representative of the population.

3. **Control Variables**: Included sex, marital status, presence of children, and education level as demographic controls.

4. **Standard Errors**: Used heteroskedasticity-robust (HC1) standard errors. Also reported clustered SEs for robustness.

5. **Sample**: Used entire provided sample without additional restrictions per instructions.

---

## Final Results

**Preferred Estimate**: 0.0612 (6.12 percentage points)
- Standard Error: 0.0167
- 95% CI: [0.0284, 0.0940]
- P-value: 0.0003
- Sample Size: 17,382

**Interpretation**: DACA eligibility is associated with a statistically significant 6.12 percentage point increase in the probability of full-time employment among eligible Hispanic-Mexican Mexican-born individuals.

---

## Step 5: Report Generation

### Command 2: LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_17.tex
```
- Compiled LaTeX report to PDF
- Three compilation passes for stable cross-references
- Final output: 25 pages

---

## Deliverables

All required deliverables have been generated:

1. **replication_report_17.tex** - LaTeX source file (approximately 37 KB)
2. **replication_report_17.pdf** - Compiled PDF report (25 pages, approximately 329 KB)
3. **run_log_17.md** - This run log documenting all commands and decisions

### Additional Output Files:
- `analysis_17.py` - Python analysis script
- `analysis_results.csv` - Key numerical results
- `yearly_means.csv` - Year-by-year employment rates
- `event_study.csv` - Event study coefficients
- `summary_stats.csv` - Summary statistics
- `model_summaries.txt` - Full regression output

---

## Software Used

- **Python 3.x** with packages: pandas, numpy, statsmodels, scipy
- **LaTeX** (MiKTeX distribution) with packages: booktabs, threeparttable, tcolorbox, hyperref, etc.

---

## Completion Time

Analysis completed: January 27, 2026
