# Run Log - Replication 58

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA full-time employment analysis.

## Date
January 27, 2026

## Data Files
- Input: `data/prepared_data_numeric_version.csv` (17,382 observations, 105 variables)
- Data dictionary: `data/acs_data_dict.txt`
- Instructions: `replication_instructions.docx`

## Output Files
- Report: `replication_report_58.tex`, `replication_report_58.pdf`
- Figures: `figure1_trends.png`, `figure2_event_study.png`
- This log: `run_log_58.md`

---

## Session Log

### 1. Data Exploration

**Read instructions:**
```python
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]
```

**Explored data structure:**
```python
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)
```

**Key variable identification:**
- YEAR: 2008-2011 (pre), 2013-2016 (post) - 2012 excluded
- ELIGIBLE: 1 = ages 26-30 at June 2012, 0 = ages 31-35
- AFTER: 1 = 2013-2016, 0 = 2008-2011
- FT: 1 = full-time (35+ hrs/wk), 0 = otherwise
- PERWT: Person-level survey weight

### 2. Descriptive Analysis

**Cross-tabulation:**
```python
pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
# AFTER        0     1    All
# ELIGIBLE
# 0         3294  2706   6000
# 1         6233  5149  11382
# All       9527  7855  17382
```

**Weighted FT rates by group/period:**
- ELIGIBLE=0, AFTER=0: 0.6886 (n=3294)
- ELIGIBLE=0, AFTER=1: 0.6629 (n=2706)
- ELIGIBLE=1, AFTER=0: 0.6369 (n=6233)
- ELIGIBLE=1, AFTER=1: 0.6860 (n=5149)

### 3. Difference-in-Differences Analysis

**Simple DiD calculation (weighted):**
```
Treatment (ELIGIBLE=1): 0.6860 - 0.6369 = +0.0491
Control (ELIGIBLE=0): 0.6629 - 0.6886 = -0.0257
DiD Estimate: 0.0491 - (-0.0257) = 0.0748
```

**Model 1: Basic DiD (weighted, HC1 SEs)**
```python
import statsmodels.api as sm
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
model = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')
```
Results:
- const: 0.6886
- ELIGIBLE: -0.0517
- AFTER: -0.0257
- ELIGIBLE_x_AFTER: **0.0748** (SE=0.0181, p<0.001)

**Model 2: With demographic controls**
```python
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)
df['educ_some_college'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_two_year'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba_plus'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
```
DiD estimate: **0.0722** (SE=0.0234, p=0.002)

**Model 3: With year fixed effects**
```python
for y in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'year_{y}'] = (df['YEAR'] == y).astype(int)
```
DiD estimate: **0.0601** (SE=0.0167, p<0.001)

**Model 4: With state fixed effects**
DiD estimate: **0.0594** (SE=0.0166, p<0.001)

**Model 5: With state-clustered standard errors (preferred specification)**
```python
model = sm.WLS(y, X, weights=w).fit(cov_type='cluster', cov_kwds={'groups': groups})
```
DiD estimate: **0.0601** (SE=0.0205, p=0.003)
95% CI: [0.020, 0.100]

### 4. Event Study Analysis

Created year-specific treatment interactions (reference: 2011):
```python
for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_x_{y}'] = df['ELIGIBLE'] * (df['YEAR'] == y).astype(int)
```

Results (relative to 2011):
- 2008: -0.0639 (p=0.016) *
- 2009: -0.0466 (p=0.081)
- 2010: -0.0763 (p=0.014) *
- 2011: 0 (reference)
- 2013: +0.0158 (p=0.668)
- 2014: -0.0127 (p=0.549)
- 2015: -0.0094 (p=0.781)
- 2016: +0.0617 (p=0.033) *

**Note:** Some evidence of differential pre-trends in 2008 and 2010.

### 5. Subgroup Analysis

**By sex:**
- Males (n=9,075): DiD = 0.0716 (SE=0.0195, p=0.0002)
- Females (n=8,307): DiD = 0.0527 (SE=0.0290, p=0.070)

### 6. Visualization

**Figure 1: FT Employment Trends**
```python
import matplotlib.pyplot as plt
# Plot weighted FT rates by year for eligible and control groups
plt.savefig('figure1_trends.png', dpi=150)
```

**Figure 2: Event Study**
```python
# Plot year-specific treatment effects with 95% CIs
plt.savefig('figure2_event_study.png', dpi=150)
```

### 7. Report Generation

**LaTeX report compiled:**
```bash
pdflatex -interaction=nonstopmode replication_report_58.tex
pdflatex -interaction=nonstopmode replication_report_58.tex  # Second pass for references
```
Output: 20-page PDF report

---

## Key Analytical Decisions

1. **Sample**: Used full provided sample without additional restrictions

2. **Outcome**: Full-time employment (FT) as provided, including non-labor force as zeros

3. **Treatment definition**: Used pre-constructed ELIGIBLE variable

4. **Weighting**: All regressions weighted by PERWT (person-level survey weight)

5. **Standard errors**: State-clustered robust SEs (50 clusters) to account for within-state correlation

6. **Controls included**:
   - Age (linear)
   - Female indicator
   - Married indicator
   - Education dummies (reference: High School Degree)
   - Year fixed effects

7. **Preferred specification**: Model with demographic controls, year FE, survey weights, state-clustered SEs

8. **Preferred estimate**:
   - DiD = **0.0601** (6.01 percentage points)
   - SE = 0.0205
   - p-value = 0.003
   - 95% CI: [0.020, 0.100]
   - N = 17,382

---

## Limitations Noted

1. Pre-trend concerns (2008, 2010 show significant differences from 2011)
2. Age-based confounding between groups
3. Repeated cross-section (not panel data)
4. Sample limited to Mexican-born, Hispanic individuals

---

## Software

- Python 3.x with pandas, numpy, statsmodels, matplotlib
- MiKTeX for LaTeX compilation

---

## Files Created

1. `replication_report_58.tex` - LaTeX source
2. `replication_report_58.pdf` - Final report (20 pages)
3. `figure1_trends.png` - Employment trends figure
4. `figure2_event_study.png` - Event study figure
5. `run_log_58.md` - This log file
