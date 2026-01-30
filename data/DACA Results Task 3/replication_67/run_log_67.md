# Run Log - Replication 67

## Project: DACA Impact on Full-Time Employment

### Date: 2026-01-27

---

## 1. Initial Setup and Data Review

### Files Received:
- `replication_instructions.docx` - Research task instructions
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Labelled version (for reference)
- `data/acs_data_dict.txt` - Data dictionary from IPUMS

### Research Question:
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome)?

### Research Design:
- **Treatment Group (ELIGIBLE=1)**: Ages 26-30 at time of policy (June 15, 2012)
- **Control Group (ELIGIBLE=0)**: Ages 31-35 at time of policy
- **Pre-treatment Period (AFTER=0)**: 2008-2011
- **Post-treatment Period (AFTER=1)**: 2013-2016
- **Outcome Variable**: FT (full-time employment, 1=yes, 0=no)
- **Method**: Difference-in-Differences (DiD)

### Key Variables:
- `FT`: Full-time employment (0/1, 35+ hours/week)
- `ELIGIBLE`: DACA eligibility indicator (1=treatment, 0=control)
- `AFTER`: Post-DACA indicator (1=2013-2016, 0=2008-2011)
- `YEAR`: Survey year
- `PERWT`: Person weight (for weighted estimates)
- Various demographic and state-level controls available

---

## 2. Analytic Decisions

### Decision 1: Estimation Strategy
**Choice**: Standard Difference-in-Differences regression using a linear probability model (WLS)
**Rationale**: The research design explicitly calls for comparing changes in the 26-30 age group (treated) relative to the 31-35 age group (control) from pre to post period. DiD is the natural estimator for this design.

### Decision 2: Use of Survey Weights
**Choice**: Use person weights (PERWT) for all estimates
**Rationale**: ACS is a complex survey; weighted estimates are necessary for population-level inference and to account for differential sampling probabilities.

### Decision 3: Standard Error Clustering
**Choice**: Cluster standard errors at the state level (STATEFIP)
**Rationale**: DACA is a federal policy, but state-level policies and labor markets may create correlation in outcomes within states. State-level clustering is conservative and accounts for within-state correlation. With 51 clusters (states + DC), cluster-robust inference is appropriate.

### Decision 4: Covariates
**Choice**: Include demographic controls and state/year fixed effects in preferred specification
**Rationale**: Improve precision and control for compositional changes between groups over time. Controls include:
- Sex (SEX, recoded to FEMALE indicator)
- Marital status (MARST, recoded to MARRIED indicator)
- Presence of children (NCHILD > 0, recoded to HAS_CHILDREN indicator)
- Age (AGE)
- State fixed effects (STATEFIP)
- Year fixed effects (YEAR)

### Decision 5: Sample
**Choice**: Use entire provided sample without further restrictions
**Rationale**: Instructions explicitly state "do not further limit the sample by dropping individuals on the basis of their characteristics." Sample is already restricted to Hispanic-Mexican, Mexican-born individuals meeting age and other DACA criteria.

### Decision 6: Reference Year for Event Study
**Choice**: Use 2011 as reference year
**Rationale**: 2011 is the last pre-treatment year, making it the natural reference point for comparing pre-trends and post-treatment effects.

---

## 3. Commands and Code Execution Log

### Step 1: Data Loading and Initial Exploration
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")  # Output: 17,382
```

### Step 2: Sample Size Verification
```
Unweighted sample sizes:
- Control (31-35), Pre-DACA: 3,294
- Control (31-35), Post-DACA: 2,706
- Treatment (26-30), Pre-DACA: 6,233
- Treatment (26-30), Post-DACA: 5,149
Total: 17,382

Weighted sample sizes (using PERWT):
- Control (31-35), Pre-DACA: 449,366
- Control (31-35), Post-DACA: 370,666
- Treatment (26-30), Pre-DACA: 868,160
- Treatment (26-30), Post-DACA: 728,157
Total: 2,416,349
```

### Step 3: Descriptive Statistics
```
Full-Time Employment Rates (Weighted):
- Control, Pre-DACA: 0.6886
- Control, Post-DACA: 0.6629 (change: -0.0257)
- Treatment, Pre-DACA: 0.6369
- Treatment, Post-DACA: 0.6860 (change: +0.0491)

Simple DiD = 0.0491 - (-0.0257) = 0.0748
```

### Step 4: Main DiD Analysis

#### Model 1: Basic DiD
```python
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                  data=df, weights=df['PERWT']).fit(
                  cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```
**Result**: DiD Estimate = 0.0748 (SE = 0.0203, p = 0.0002)

#### Model 2: DiD with Demographics
```python
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE',
                  data=df, weights=df['PERWT']).fit(
                  cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```
**Result**: DiD Estimate = 0.0647 (SE = 0.0210, p = 0.0021)

#### Model 3: DiD with State and Year Fixed Effects (PREFERRED)
```python
# Include state and year dummies in regression
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + [state dummies] + [year dummies]',
                  data=df, weights=df['PERWT']).fit(
                  cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```
**Result**: DiD Estimate = 0.0611 (SE = 0.0209, p = 0.0034)

### Step 5: Event Study Specification
```python
# Create year-specific treatment effects relative to 2011
event_vars = ['ELIGIBLE_X_2008', 'ELIGIBLE_X_2009', 'ELIGIBLE_X_2010',
              'ELIGIBLE_X_2013', 'ELIGIBLE_X_2014', 'ELIGIBLE_X_2015', 'ELIGIBLE_X_2016']
```

**Event Study Results (relative to 2011)**:
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2008 | -0.068** | 0.029 | [-0.126, -0.011] |
| 2009 | -0.050 | 0.037 | [-0.123, +0.024] |
| 2010 | -0.082*** | 0.030 | [-0.140, -0.024] |
| 2011 | 0.000 | --- | (Reference) |
| 2013 | +0.016 | 0.041 | [-0.064, +0.095] |
| 2014 | +0.000 | 0.028 | [-0.055, +0.055] |
| 2015 | +0.001 | 0.038 | [-0.074, +0.077] |
| 2016 | +0.074** | 0.030 | [+0.016, +0.133] |

### Step 6: Generate Figures
```python
python generate_figures.py
```
- figure1_parallel_trends.pdf/png
- figure2_event_study.pdf/png
- figure3_did_visual.pdf/png
- figure4_coefficient_comparison.pdf/png

### Step 7: Compile LaTeX Report
```bash
pdflatex -interaction=nonstopmode replication_report_67.tex
pdflatex -interaction=nonstopmode replication_report_67.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_67.tex  # Third pass to finalize
```

---

## 4. Results Summary

### Preferred Estimate (Model 3: DiD with State and Year Fixed Effects)
- **Effect Size**: 0.0611 (6.11 percentage points)
- **Standard Error**: 0.0209 (clustered at state level)
- **95% Confidence Interval**: [0.0202, 0.1020]
- **p-value**: 0.0034
- **Sample Size**: 17,382 observations

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 6.1 percentage points among Hispanic-Mexican, Mexican-born individuals who met the age and other eligibility criteria. This represents approximately a 9-10% increase relative to the pre-DACA employment rate of the treatment group (63.7%).

### Robustness
- Effect is statistically significant at the 1% level across all specifications
- Estimates range from 6.1 to 7.5 percentage points across models
- Event study shows some pre-trend differences (potential concern for parallel trends assumption)
- Strong positive effect emerges by 2016

---

## 5. Files Produced

### Required Deliverables
- `replication_report_67.tex` - LaTeX source for replication report (39,856 bytes)
- `replication_report_67.pdf` - Compiled PDF report (23 pages, 408,280 bytes)
- `run_log_67.md` - This log file

### Analysis Files
- `analysis_script.py` - Main Python analysis script
- `generate_figures.py` - Figure generation script
- `analysis_results.json` - Stored numerical results
- `model_summaries.txt` - Full regression output
- `yearly_ft_rates.csv` - Year-by-year employment rates
- `group_counts.csv` - Sample sizes by group

### Figures
- `figure1_parallel_trends.pdf/png` - Parallel trends visualization
- `figure2_event_study.pdf/png` - Event study coefficients
- `figure3_did_visual.pdf/png` - DiD visualization
- `figure4_coefficient_comparison.pdf/png` - Coefficient comparison across models

---

## 6. Notes and Caveats

1. **Parallel Trends**: Event study reveals some pre-existing trends (2008 and 2010 coefficients significantly different from zero), suggesting the parallel trends assumption may not hold perfectly. Treatment group shows improvement relative to control even before DACA.

2. **Delayed Effect**: Post-DACA effects are small and insignificant in 2013-2015 but become significant by 2016, consistent with gradual DACA uptake and cumulative effects.

3. **Sample Composition**: Treatment group is younger (by design), slightly more female, less likely to be married, and less likely to have children than control group. These differences are controlled for in Models 2 and 3.

4. **Age-Based Identification**: The comparison relies on the age cutoff for DACA eligibility (under 31 as of June 2012). Differential trends by age cohort could confound estimates if younger workers were affected differently by the Great Recession recovery.

---

*Log completed: 2026-01-27*
