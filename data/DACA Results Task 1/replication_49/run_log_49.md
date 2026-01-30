# DACA Replication Study Run Log

## Date: 2026-01-25

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hours per week)?

---

## Data Source
- American Community Survey (ACS) from IPUMS USA
- Years: 2006-2016 (one-year ACS files)
- Data file: data.csv (~33.8 million observations)
- Data dictionary: acs_data_dict.txt

---

## Key Variable Definitions

### Sample Restrictions
1. **Hispanic-Mexican ethnicity**: HISPAN == 1 (Mexican)
2. **Born in Mexico**: BPL == 200 (Mexico)
3. **Working age**: AGE 16-64 (standard working age population)
4. **Non-citizens (potential undocumented)**: CITIZEN == 3 (Not a citizen)

### DACA Eligibility Criteria (as of June 15, 2012)
1. Arrived unlawfully in the US before their 16th birthday
   - Proxy: YRIMMIG - BIRTHYR < 16
2. Had not yet had their 31st birthday as of June 15, 2012
   - Born after June 15, 1981 → BIRTHYR >= 1982 (conservative)
3. Lived continuously in the US since June 15, 2007
   - YRIMMIG <= 2007
4. Were present in the US on June 15, 2012
   - Assumed for ACS respondents in US

### Outcome Variable
- **Full-time employment**: UHRSWORK >= 35

### Treatment and Control Groups
- **Treatment group**: Non-citizen Hispanic-Mexicans born in Mexico who meet DACA eligibility criteria
- **Control group**: Non-citizen Hispanic-Mexicans born in Mexico who do NOT meet DACA eligibility criteria (arrived age 16+, or arrived after 2007, or born before 1982)

### Time Periods
- **Pre-treatment**: 2006-2011 (DACA announced June 2012, so 2012 is transitional)
- **Post-treatment**: 2013-2016

---

## Analytical Approach: Difference-in-Differences

### Identification Strategy
We employ a difference-in-differences (DiD) design comparing:
1. Changes in full-time employment for DACA-eligible individuals (treatment)
2. Against changes for non-DACA-eligible Hispanic-Mexican non-citizens (control)

### Main Specification
```
FullTime_it = β₀ + β₁ Eligible_i + β₂ Post_t + β₃ (Eligible_i × Post_t) + X_it'γ + ε_it
```

Where:
- FullTime_it = 1 if person i in year t works 35+ hours/week
- Eligible_i = 1 if person meets DACA eligibility criteria
- Post_t = 1 if year >= 2013
- β₃ = DiD estimate (causal effect of DACA eligibility)
- X_it = control variables (age, sex, education, state, year fixed effects)

### Control Variables
- Age (continuous and squared)
- Sex (binary)
- Education level (categorical: less than HS, some HS, HS grad, some college, college+)
- Marital status
- State fixed effects (STATEFIP)
- Year fixed effects

---

## Commands and Execution Log

### Step 1: Data Preparation
```python
# Load required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data in chunks (33.8M rows total)
data_path = "data/data.csv"
chunks = []
chunksize = 1000000
for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

### Step 2: Sample Selection
```python
# Apply sample restrictions
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]  # Working age
df = df[df['CITIZEN'] == 3]  # Non-citizens only
df = df[df['YEAR'] != 2012]  # Exclude transitional year
```

**Sample Sizes After Each Filter:**
- Hispanic-Mexican + Mexico-born: 991,261
- After age 16-64 filter: 851,090
- After non-citizen filter: 618,640
- After excluding 2012: 561,470 (final sample)

### Step 3: Create Analysis Variables
```python
# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility indicator
df['daca_eligible'] = (
    (df['age_at_immigration'] < 16) &
    (df['age_at_immigration'] >= 0) &
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0)
).astype(int)

# Post-treatment indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['daca_post'] = df['daca_eligible'] * df['post']

# Full-time employment outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

### Step 4: Run Regression Analysis
```python
# Model 4 (preferred): Full model with year and state fixed effects
# Standard errors clustered by state
model4 = sm.OLS(y, X4).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

---

## Key Decisions and Justifications

1. **Excluding 2012**: DACA was announced June 15, 2012, and applications began August 15, 2012. Since ACS does not identify month of interview, 2012 observations mix pre- and post-treatment. Excluding 2012 provides cleaner identification.

2. **Using CITIZEN == 3 for undocumented proxy**: We cannot distinguish documented vs undocumented non-citizens. Following the instructions, we assume non-citizens without citizenship papers are potentially undocumented.

3. **Age restriction to 16-64**: Standard working age population. DACA eligible individuals are necessarily young (born after 1981), but we apply this restriction uniformly.

4. **Control group selection**: We use Hispanic-Mexican non-citizens born in Mexico who do not meet one or more DACA criteria. This provides a similar comparison group in terms of ethnicity and immigration status.

5. **Birth year cutoff**: Using BIRTHYR >= 1982 to approximate being under 31 as of June 15, 2012. This is conservative as those born in late 1981 may still qualify.

6. **Immigration age calculation**: Using YRIMMIG - BIRTHYR to calculate age at immigration. If this value is < 16, they arrived before age 16.

7. **Clustering standard errors by state**: To account for potential serial correlation within states over time and heteroskedasticity.

---

## Analysis Results

### Sample Composition
- Total observations: 561,470
- DACA eligible: 80,300 (14.3%)
- Control group: 481,170 (85.7%)
- Pre-treatment observations: 345,792 (61.6%)
- Post-treatment observations: 215,678 (38.4%)

### Simple 2x2 Difference-in-Differences
|                | Pre-DACA | Post-DACA | Difference |
|----------------|----------|-----------|------------|
| DACA Eligible  | 0.426    | 0.495     | +0.069     |
| Control Group  | 0.603    | 0.579     | -0.025     |
| **DiD Estimate** |        |           | **+0.093** |

### Regression Results Summary

| Model | Specification | DiD Coefficient | Std. Error | p-value |
|-------|---------------|-----------------|------------|---------|
| 1 | Basic (no controls) | 0.093 | 0.005 | <0.001 |
| 2 | + Demographics | 0.043 | 0.006 | <0.001 |
| 3 | + Year FE | 0.038 | 0.005 | <0.001 |
| 4 | + State FE (preferred) | **0.037** | **0.005** | **<0.001** |

### Preferred Estimate (Model 4)
- **Effect Size**: 0.0371 (3.71 percentage points)
- **Standard Error**: 0.0053
- **95% Confidence Interval**: [0.0266, 0.0476]
- **P-value**: < 0.001
- **Sample Size**: 561,470
- **R-squared**: 0.216

### Robustness Checks

| Specification | DiD Estimate | Std. Error | p-value |
|---------------|--------------|------------|---------|
| Any employment (alternative outcome) | 0.047 | 0.010 | <0.001 |
| Males only | 0.032 | 0.005 | <0.001 |
| Females only | 0.033 | 0.008 | <0.001 |
| Stricter birth year (1983+) | 0.044 | 0.005 | <0.001 |
| Placebo test (2010+ vs 2006-2009) | 0.017 | 0.004 | <0.001 |

### Event Study Coefficients (reference: 2011)

| Year | Coefficient | Std. Error | Significance |
|------|-------------|------------|--------------|
| 2006 | -0.025 | 0.008 | *** |
| 2007 | -0.020 | 0.006 | *** |
| 2008 | -0.005 | 0.009 | |
| 2009 | +0.003 | 0.008 | |
| 2010 | +0.006 | 0.011 | |
| 2011 | [reference] | -- | |
| 2013 | +0.011 | 0.009 | |
| 2014 | +0.025 | 0.014 | * |
| 2015 | +0.045 | 0.009 | *** |
| 2016 | +0.045 | 0.010 | *** |

---

## Files Generated

1. **analysis.py** - Main analysis script
2. **create_figures.py** - Figure generation script
3. **results_summary.csv** - Key results in CSV format
4. **summary_statistics.csv** - Summary statistics by eligibility
5. **figure1_trends.png** - Employment trends by eligibility
6. **figure2_eventstudy.png** - Event study plot
7. **figure3_agedist.png** - Age distribution
8. **figure4_gender.png** - Employment by gender
9. **figure5_did.png** - DiD visualization
10. **replication_report_49.tex** - LaTeX report source
11. **replication_report_49.pdf** - Final PDF report (22 pages)
12. **run_log_49.md** - This log file

---

## Interpretation

DACA eligibility is associated with a statistically significant **3.71 percentage point increase** in the probability of full-time employment (SE = 0.0053, 95% CI: [0.027, 0.048], p < 0.001).

This effect represents approximately an **8.7% relative increase** from the pre-DACA baseline of 42.6% for eligible individuals.

The effect is:
- Robust across specifications (Models 1-4)
- Similar for males and females
- Larger when using stricter eligibility criteria
- Grows over time (from 1.1 pp in 2013 to 4.5 pp in 2015-2016)

**Caution**: The placebo test shows a significant pre-trend (1.7 pp, p < 0.001), and early event study coefficients are negative and significant, suggesting potential parallel trends violations. Results should be interpreted with caution.

---

## Execution Times

- Data loading and filtering: ~2 minutes
- Main regression analysis: ~5 minutes
- Figure generation: ~3 minutes
- LaTeX compilation: ~30 seconds

Total execution time: ~10 minutes

---

## Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- matplotlib
- pdflatex (MiKTeX)
