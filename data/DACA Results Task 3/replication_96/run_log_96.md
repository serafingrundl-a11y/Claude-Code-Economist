# DACA Replication Study - Run Log (ID: 96)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment.

**Date:** January 27, 2026
**Research Question:** What was the causal impact of DACA eligibility on the probability of full-time employment among Hispanic-Mexican, Mexican-born individuals in the US?

---

## 1. Data Exploration

### 1.1 Initial File Discovery
Examined the working directory structure:
- `replication_instructions.docx` - Research instructions
- `data/prepared_data_numeric_version.csv` - Main analysis dataset
- `data/prepared_data_labelled_version.csv` - Labelled version of dataset
- `data/acs_data_dict.txt` - Data dictionary

### 1.2 Data Dictionary Review
Read the ACS data dictionary to understand variable definitions:
- Key IPUMS variables: YEAR, SEX, AGE, MARST, EDUC, EMPSTAT, UHRSWORK, etc.
- Custom variables: FT, AFTER, ELIGIBLE

### 1.3 Data Loading
Loaded `prepared_data_numeric_version.csv`:
- **Total observations:** 17,382
- **Years:** 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- **Columns:** 105

---

## 2. Key Variables

| Variable | Description | Coding |
|----------|-------------|--------|
| FT | Full-time employment | 1 = 35+ hours/week, 0 = otherwise |
| ELIGIBLE | Treatment group | 1 = ages 26-30 in June 2012, 0 = ages 31-35 |
| AFTER | Post-DACA period | 1 = 2013-2016, 0 = 2008-2011 |
| PERWT | Person weight | ACS survey weights |

### Sample Distribution:
```
ELIGIBLE=1, AFTER=0: 6,233 (Treatment, Pre)
ELIGIBLE=1, AFTER=1: 5,149 (Treatment, Post)
ELIGIBLE=0, AFTER=0: 3,294 (Control, Pre)
ELIGIBLE=0, AFTER=1: 2,706 (Control, Post)
```

---

## 3. Analytical Decisions

### 3.1 Identification Strategy
- **Method:** Difference-in-Differences (DiD)
- **Treatment group:** Individuals aged 26-30 in June 2012 (ELIGIBLE=1)
- **Control group:** Individuals aged 31-35 in June 2012 (ELIGIBLE=0)
- **Pre-period:** 2008-2011 (AFTER=0)
- **Post-period:** 2013-2016 (AFTER=1)

### 3.2 Outcome Variable
- Used provided FT variable (1 = full-time employment, 0 = otherwise)
- Kept individuals not in labor force as FT=0 per instructions

### 3.3 Weighting
- Used ACS person weights (PERWT) for all analyses
- Produces nationally representative estimates

### 3.4 Standard Errors
- Used heteroskedasticity-robust standard errors (HC1)
- Did not cluster standard errors (no panel dimension)

### 3.5 Fixed Effects
- Year fixed effects: Control for common time trends
- State fixed effects: Control for time-invariant state characteristics

### 3.6 Control Variables
- SEX (recoded to FEMALE indicator)
- MARST (recoded to MARRIED indicator)
- NCHILD (number of children)
- EDUC (categorized: Less than HS, HS, Some College, BA+)

---

## 4. Commands Executed

### 4.1 Python Analysis Script
Created `analysis.py` containing:

```python
# Load data
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data_path = "data/prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

# Model 2: Year Fixed Effects
model2 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

# Model 3: Demographic Controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)
model3 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + FEMALE + MARRIED + NCHILD + C(EDUC_cat)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')

# Model 4: State Fixed Effects (Preferred)
model4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + FEMALE + MARRIED + NCHILD + C(EDUC_cat)',
                  data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')
```

### 4.2 Run Analysis
```bash
cd "C:\Users\seraf\DACA Results Task 3\replication_96"
python analysis.py
```

### 4.3 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_96.tex
pdflatex -interaction=nonstopmode replication_report_96.tex  # Second pass for references
pdflatex -interaction=nonstopmode replication_report_96.tex  # Third pass
```

---

## 5. Results Summary

### 5.1 Simple DiD Estimate
```
Pre-period:  Treatment=0.637, Control=0.689, Diff=-0.052
Post-period: Treatment=0.686, Control=0.663, Diff=+0.023
DiD Estimate: 0.075 (7.5 percentage points)
```

### 5.2 Regression Results

| Model | Specification | DiD Coef | SE | p-value |
|-------|---------------|----------|-----|---------|
| 1 | Basic | 0.0748 | 0.0181 | <0.001 |
| 2 | + Year FE | 0.0721 | 0.0181 | <0.001 |
| 3 | + Covariates | 0.0617 | 0.0167 | <0.001 |
| 4 | + State FE (Preferred) | 0.0611 | 0.0166 | <0.001 |

### 5.3 Preferred Estimate (Model 4)
- **Effect Size:** 0.0611 (6.11 percentage points)
- **Standard Error:** 0.0166
- **95% CI:** [0.0285, 0.0937]
- **p-value:** 0.0002
- **Sample Size:** 17,379

### 5.4 Heterogeneity by Sex
- Males: DiD = 0.072 (SE = 0.020), p < 0.001
- Females: DiD = 0.053 (SE = 0.028), p = 0.061

---

## 6. Files Produced

### 6.1 Analysis Files
- `analysis.py` - Python analysis script
- `analysis_results.json` - Saved regression results

### 6.2 Figures
- `figure1_trends.png` - FT rates by group over time
- `figure2_event_study.png` - Event study coefficients
- `figure3_difference.png` - Treatment-control gap over time

### 6.3 Report Files
- `replication_report_96.tex` - LaTeX source
- `replication_report_96.pdf` - Final report (18 pages)

### 6.4 Log Files
- `run_log_96.md` - This file

---

## 7. Key Interpretation

DACA eligibility is associated with a **6.1 percentage point increase** in the probability of full-time employment among eligible Mexican-born Hispanic individuals, compared to the control group of individuals who were too old to qualify.

This represents approximately a **10% relative increase** from the baseline full-time employment rate of 63.7% in the pre-period.

The effect is:
- Statistically significant at the 1% level
- Robust across multiple specifications
- Larger for males than females
- Consistent with theoretical predictions about legal work authorization

---

## 8. Limitations Noted

1. **Parallel trends:** Some evidence of non-parallel trends in pre-period (2010 coefficient statistically different from zero)
2. **DACA receipt not observed:** Analysis based on eligibility, not actual DACA receipt
3. **Potential spillovers:** Control group may be affected if DACA changed overall labor market
4. **External validity:** Results specific to Mexican-born Hispanic individuals aged 26-35

---

## 9. Session Information

- **Platform:** Windows 11
- **Python:** 3.x with pandas, numpy, statsmodels, matplotlib
- **LaTeX:** MiKTeX 25.12
- **Date completed:** January 27, 2026
