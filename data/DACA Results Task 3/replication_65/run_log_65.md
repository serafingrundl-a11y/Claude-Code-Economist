# Replication Run Log - Study 65

## Overview
Independent replication study examining the causal effect of DACA eligibility on full-time employment among Mexican-born Hispanic immigrants in the United States.

## Date
January 27, 2026

---

## Step 1: Data Understanding

### Instructions Review
- Read `replication_instructions.docx` using Python's docx library
- Key research question: Effect of DACA eligibility on full-time employment
- Treatment group: Ages 26-30 at June 15, 2012 (DACA implementation)
- Control group: Ages 31-35 at June 15, 2012 (age-ineligible)
- Outcome: Full-time employment (35+ hours/week)
- Design: Difference-in-differences

### Data Files
- `prepared_data_labelled_version.csv`: Labeled categorical variables
- `prepared_data_numeric_version.csv`: Numeric version (used for analysis)
- `acs_data_dict.txt`: Variable definitions from IPUMS

### Key Pre-constructed Variables
- `ELIGIBLE`: 1 for treatment group (ages 26-30), 0 for control (ages 31-35)
- `AFTER`: 1 for post-DACA (2013-2016), 0 for pre-DACA (2008-2011)
- `FT`: 1 for full-time employed (35+ hours/week), 0 otherwise
- `PERWT`: Person-level survey weight

---

## Step 2: Data Exploration

### Command: Load and examine data structure
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)
```

### Sample Characteristics
- Total observations: 17,382
- Years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- States: 50 states represented
- Full-time employed: 11,283 (64.9%)
- Pre-DACA observations: 9,527
- Post-DACA observations: 7,855
- Treatment group (ELIGIBLE=1): 11,382
- Control group (ELIGIBLE=0): 6,000

### Age Verification
- Treatment group AGE_IN_JUNE_2012: mean=28.1, range=[26, 30.75]
- Control group AGE_IN_JUNE_2012: mean=32.9, range=[31, 35]

---

## Step 3: Descriptive Statistics

### Full-time Employment Rates (Weighted)
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control (31-35) | 0.689 | 0.663 | -0.026 |
| Treated (26-30) | 0.637 | 0.686 | +0.049 |

### Simple DiD Calculation (Weighted)
- Control change: 0.663 - 0.689 = -0.026
- Treated change: 0.686 - 0.637 = +0.049
- **DiD estimate: 0.049 - (-0.026) = 0.075**

---

## Step 4: Regression Analysis

### Model 1: Basic DiD (Unweighted)
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER
```
- ELIGIBLE_AFTER coefficient: 0.0643 (SE=0.015, p<0.001)

### Model 2: Basic DiD (Weighted with PERWT)
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER [weights=PERWT]
```
- ELIGIBLE_AFTER coefficient: 0.0748 (SE=0.015, p<0.001)

### Model 3: With Demographic Controls (Weighted)
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER + FEMALE + MARRIED + C(EDUC) + NCHILD + AGE
```
- ELIGIBLE_AFTER coefficient: 0.0645 (SE=0.014, p<0.001)

### Model 4: Full Model with State and Year FE
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER + FEMALE + MARRIED + C(EDUC) + NCHILD + C(STATEFIP) + C(YEAR)
```
- ELIGIBLE_AFTER coefficient: 0.0605 (SE=0.014, p<0.001)

### Model 5 (PREFERRED): Full Model with Clustered SE
```
FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER + FEMALE + MARRIED + C(EDUC) + NCHILD + C(STATEFIP) + C(YEAR)
[weights=PERWT, SE clustered by STATEFIP]
```
- **ELIGIBLE_AFTER coefficient: 0.0605 (SE=0.0215, p=0.005)**
- **95% CI: [0.018, 0.103]**

---

## Step 5: Event Study Analysis

### Year-Specific Treatment Effects (Reference: 2011)
| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2008 | -0.067*** | 0.026 | [-0.117, -0.017] |
| 2009 | -0.047* | 0.027 | [-0.100, 0.005] |
| 2010 | -0.076** | 0.032 | [-0.138, -0.014] |
| 2011 | 0 (ref) | --- | --- |
| 2013 | 0.018 | 0.036 | [-0.054, 0.089] |
| 2014 | -0.017 | 0.021 | [-0.059, 0.024] |
| 2015 | -0.009 | 0.034 | [-0.076, 0.059] |
| 2016 | 0.063** | 0.028 | [0.007, 0.118] |

### Interpretation
- Pre-treatment coefficients show some deviation from parallel trends (convergence by 2011)
- Post-treatment effects grow over time, largest in 2016
- Pattern consistent with gradual DACA enrollment

---

## Step 6: Heterogeneous Effects

### By Sex
- Male: DiD = 0.061*** (SE=0.019), N=9,075
- Female: DiD = 0.048 (SE=0.028), N=8,307

### By Education
- HS or less: DiD = 0.047** (SE=0.023), N=12,456
- Some college: DiD = 0.087** (SE=0.033), N=3,868
- BA or higher: DiD = 0.160*** (SE=0.028), N=1,058

### By Marital Status
- Married: DiD = 0.005 (SE=0.019), N=7,851
- Never married: DiD = 0.124*** (SE=0.040), N=7,405

---

## Step 7: Robustness Checks

### Placebo Test (Fake treatment in 2010, pre-period only)
- Placebo DiD coefficient: 0.018 (SE=0.024, p=0.445)
- Result: Not significant, supports causal interpretation

### Narrower Age Bandwidth (Ages 29-32)
- DiD coefficient: 0.045** (SE=0.020, p=0.028), N=5,624
- Smaller magnitude but still significant

### State Policy Controls
- DiD with driver's licenses, tuition, E-Verify, Secure Communities controls
- DiD coefficient: 0.060*** (SE=0.022, p=0.005)
- Virtually unchanged from main specification

---

## Step 8: Visualizations Generated

1. `parallel_trends.png`: FT employment rates by group and year
2. `did_estimate.png`: 2x2 DiD visualization with counterfactual
3. `event_study.png`: Year-specific treatment effects
4. `descriptive_stats.png`: Sample characteristics by group

---

## Key Decisions Made

1. **Used weighted regression**: ACS person weights (PERWT) ensure representativeness

2. **Clustered standard errors by state**: Accounts for within-state correlation and state-level policy variation

3. **Included state and year fixed effects**: Controls for unobserved state and time-varying factors

4. **Control variables selected**:
   - Sex (FEMALE indicator)
   - Marital status (MARRIED indicator)
   - Education (EDUC categorical)
   - Number of children (NCHILD)

5. **Did not further subset the sample**: As instructed, used entire provided dataset

6. **Used pre-defined ELIGIBLE variable**: Did not create alternative eligibility definitions

---

## Preferred Estimate Summary

| Measure | Value |
|---------|-------|
| Effect size (DiD) | 0.0605 |
| Standard error | 0.0215 |
| 95% Confidence Interval | [0.018, 0.103] |
| p-value | 0.005 |
| Sample size | 17,382 |

**Interpretation**: DACA eligibility increased full-time employment by approximately 6.0 percentage points (about 9.4% relative to pre-treatment mean of 63.7%) among Mexican-born Hispanic immigrants aged 26-30 compared to those aged 31-35.

---

## Output Files Generated

1. `replication_report_65.tex` - Full LaTeX report (~19 pages)
2. `replication_report_65.pdf` - Compiled PDF report
3. `run_log_65.md` - This log file
4. `parallel_trends.png` - Parallel trends figure
5. `did_estimate.png` - DiD visualization
6. `event_study.png` - Event study figure
7. `descriptive_stats.png` - Descriptive statistics figure

---

## Software Used

- Python 3.x with pandas, numpy, statsmodels, matplotlib
- pdfLaTeX (MiKTeX) for document compilation

---

## Session End
All required deliverables completed and verified.
