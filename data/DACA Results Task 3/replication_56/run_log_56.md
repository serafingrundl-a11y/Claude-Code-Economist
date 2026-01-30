# Replication Run Log - Study 56

## Overview
Independent replication of DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

## Research Design
- **Treatment Group**: Individuals ages 26-30 at time of DACA implementation (June 15, 2012) who meet eligibility criteria (ELIGIBLE=1)
- **Control Group**: Individuals ages 31-35 at time of DACA implementation who would have been eligible but for age (ELIGIBLE=0)
- **Outcome**: Full-time employment (FT=1 if usually working 35+ hours/week)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 excluded)

## Key Variables
- **FT**: Full-time employment (1=yes, 0=no) - outcome variable
- **ELIGIBLE**: DACA eligibility status (1=treatment, 0=control)
- **AFTER**: Post-DACA period indicator (1=2013-2016, 0=2008-2011)
- **PERWT**: Person-level survey weights

## Session Log

### Step 1: Data Loading and Initial Exploration
- Read replication instructions from `replication_instructions.docx`
- Read data dictionary from `data/acs_data_dict.txt`
- Loaded data from `data/prepared_data_numeric_version.csv`
- Dataset contains 17,382 observations and 105 variables
- Treatment group (ELIGIBLE=1): 11,382 observations
- Control group (ELIGIBLE=0): 6,000 observations
- Pre-period (AFTER=0): 9,527 observations
- Post-period (AFTER=1): 7,855 observations

### Step 2: Exploratory Data Analysis
**Commands executed:**
```python
python analysis.py
```

**Key findings:**
- Overall full-time employment rate: 64.91%
- FT rate by group (unweighted):
  - Treatment Pre: 0.6263
  - Treatment Post: 0.6658
  - Control Pre: 0.6697
  - Control Post: 0.6449
- Simple DiD estimate (unweighted): 0.0643

### Step 3: Main Regression Analysis
**Commands executed:**
```python
python analysis_regression.py
```

**Models estimated:**
1. Basic OLS (no weights, no covariates): DiD = 0.0643 (SE = 0.0153)
2. Weighted (PERWT): DiD = 0.0748 (SE = 0.0152)
3. With covariates: DiD = 0.0621 (SE = 0.0142)
4. Year fixed effects: DiD = 0.0721 (SE = 0.0151)
5. Full model (year FE + covariates): DiD = 0.0593 (SE = 0.0142)
6. State fixed effects added: DiD = 0.0588 (SE = 0.0142)

**Preferred specification:**
- Model with covariates, survey weights, and state-clustered standard errors
- DiD Estimate: **0.0621**
- Clustered SE: **0.0212**
- t-statistic: 2.93
- p-value: **0.003**
- 95% CI: [0.021, 0.104]

### Step 4: Robustness Checks
**Commands executed:**
```python
python analysis_robustness.py
```

**Event Study Results (Reference: 2008):**
- Pre-treatment coefficients (2009, 2010): Not significant, supports parallel trends
- 2011: 0.0644 (p=0.014) - possible anticipation effect
- Post-treatment: Significant positive effects, especially 2016 (0.1244, p<0.001)

**Placebo Test (fake treatment in 2010):**
- Placebo DiD: 0.0172 (SE = 0.0243)
- p-value: 0.478 (not significant - passes placebo test)

**Heterogeneity Analysis:**
- Males: DiD = 0.0600 (SE = 0.0196, p = 0.002)
- Females: DiD = 0.0535 (SE = 0.0280, p = 0.056)
- Married: DiD = 0.0095 (SE = 0.0127, p = 0.451)
- Not Married: DiD = 0.1005 (SE = 0.0402, p = 0.012)

### Step 5: Figure Generation
**Commands executed:**
```python
python create_figures.py
```

**Figures created:**
1. `figure1_parallel_trends.png` - FT trends by treatment status
2. `figure2_event_study.png` - Event study coefficients
3. `figure3_did_visualization.png` - DiD 2x2 visualization
4. `figure4_age_distribution.png` - Age distribution by group
5. `figure5_heterogeneity.png` - Heterogeneity results
6. `figure6_model_comparison.png` - Model coefficient comparison

### Step 6: LaTeX Report Generation
**Commands executed:**
```bash
pdflatex -interaction=nonstopmode replication_report_56.tex
pdflatex -interaction=nonstopmode replication_report_56.tex  # Second pass for refs
pdflatex -interaction=nonstopmode replication_report_56.tex  # Third pass
```

**Output:**
- `replication_report_56.tex` (LaTeX source)
- `replication_report_56.pdf` (21 pages)

## Key Analytic Decisions

### Decision 1: Preferred Specification
**Choice:** Model with demographic covariates, survey weights, and state-clustered standard errors
**Rationale:**
- Survey weights ensure estimates are representative of the target population
- Covariates control for observable differences between treatment and control groups
- State-clustered SEs account for within-state correlation and are conservative

### Decision 2: Covariate Selection
**Covariates included:** Sex (female dummy), marital status, education (some college, BA+), children, age
**Rationale:** These are standard demographic controls that may affect employment and differ between age groups

### Decision 3: Standard Error Approach
**Choice:** State-clustered standard errors
**Rationale:**
- Treatment effects may be correlated within states due to local labor markets
- State policies (e.g., driver's license access) may vary
- More conservative than robust SEs

### Decision 4: Sample Restrictions
**Choice:** Used full provided sample with no additional restrictions
**Rationale:** Instructions specified "do not further limit the sample by dropping individuals on the basis of their characteristics"

## Final Results Summary

| Specification | Estimate | SE | 95% CI | p-value |
|--------------|----------|-----|--------|---------|
| Basic OLS | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |
| Weighted | 0.0748 | 0.0152 | [0.045, 0.105] | <0.001 |
| **Preferred** | **0.0621** | **0.0212** | **[0.021, 0.104]** | **0.003** |
| State FE | 0.0588 | 0.0142 | [0.031, 0.087] | <0.001 |

**Interpretation:** DACA eligibility increased full-time employment by approximately 6.2 percentage points (95% CI: 2.1-10.4 pp), a statistically significant effect at the 0.01 level.

## Files Generated

| File | Description |
|------|-------------|
| `analysis.py` | Initial data exploration |
| `analysis_regression.py` | Main DiD regression analysis |
| `analysis_robustness.py` | Robustness and heterogeneity tests |
| `create_figures.py` | Figure generation |
| `figure1_parallel_trends.png` | Parallel trends plot |
| `figure2_event_study.png` | Event study plot |
| `figure3_did_visualization.png` | DiD visualization |
| `figure4_age_distribution.png` | Age distribution |
| `figure5_heterogeneity.png` | Heterogeneity results |
| `figure6_model_comparison.png` | Model comparison |
| `replication_report_56.tex` | LaTeX report source |
| `replication_report_56.pdf` | Final PDF report (21 pages) |
| `run_log_56.md` | This run log |
