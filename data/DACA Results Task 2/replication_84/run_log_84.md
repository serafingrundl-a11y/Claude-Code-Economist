# Run Log - DACA Replication Study (ID: 84)

## Overview
This log documents all commands executed and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Mexican-born non-citizens.

---

## 1. Initial Setup and Data Exploration

### Date: January 26, 2026

### 1.1 Read Replication Instructions
- Extracted and read `replication_instructions.docx` using Python's `python-docx` library
- Key research question: Effect of DACA eligibility on full-time employment (35+ hours/week)
- Treatment group: Ages 26-30 at June 15, 2012
- Control group: Ages 31-35 at June 15, 2012
- Method: Difference-in-differences

### 1.2 Data Files Available
```
data/data.csv           - Main ACS data (2006-2016)
data/acs_data_dict.txt  - Data dictionary for IPUMS variables
data/state_demo_policy.csv - Optional state-level data (not used)
```

### 1.3 Data Loading
```python
import pandas as pd
df = pd.read_csv('data/data.csv')
# Result: 33,851,424 observations, 54 variables
# Years: 2006-2016 (11 years of ACS data)
```

---

## 2. Sample Selection Decisions

### 2.1 Target Population
Identified DACA-eligible population using the following criteria:

1. **Hispanic-Mexican Ethnicity + Mexican Birthplace**
   - Variables: HISPAN = 1 (Mexican) AND BPL = 200 (Mexico)
   - Rationale: Research question specifies "ethnically Hispanic-Mexican Mexican-born"
   - Result: 991,261 observations

2. **Non-Citizenship Status**
   - Variable: CITIZEN = 3 (Not a citizen)
   - Rationale: Proxy for undocumented status per instructions
   - Result: 701,347 observations (after combining with criterion 1)

3. **Arrived Before Age 16**
   - Calculation: YRIMMIG - BIRTHYR < 16 (where YRIMMIG > 0)
   - Rationale: DACA eligibility requirement
   - Result: 195,023 observations (after combining with criteria 1-3 below)

4. **In U.S. Since 2007**
   - Variable: YRIMMIG <= 2007
   - Rationale: DACA required continuous presence since June 15, 2007

5. **Age Groups**
   - Treatment: Age 26-30 in mid-2012
   - Control: Age 31-35 in mid-2012
   - Age in 2012 calculated as: AGE - (YEAR - 2012)
   - Result: 49,019 observations

6. **Exclude 2012**
   - Rationale: DACA announced June 15, 2012; ACS doesn't indicate month
   - Result: 44,725 final observations

### Sample Breakdown:
- Treatment group: 26,591
- Control group: 18,134
- Pre-period (2006-2011): 29,326
- Post-period (2013-2016): 15,399

---

## 3. Variable Construction

### 3.1 Outcome Variable
```python
fulltime = (UHRSWORK >= 35).astype(int)
```
- Binary indicator: 1 if usually works 35+ hours/week, 0 otherwise
- Pre-treatment mean (treatment): 62.53%
- Pre-treatment mean (control): 67.05%

### 3.2 Treatment Variables
```python
treated = (age_in_2012 >= 26) & (age_in_2012 <= 30)
post = (YEAR >= 2013)
treat_post = treated * post  # DiD interaction
```

### 3.3 Control Variables
```python
female = (SEX == 2).astype(int)
married = (MARST == 1).astype(int)
has_children = (NCHILD > 0).astype(int)
hs_grad = (EDUC == 6).astype(int)
some_college = (EDUC >= 7) & (EDUC <= 9).astype(int)
college_plus = (EDUC >= 10).astype(int)
```

---

## 4. Statistical Analysis

### 4.1 Models Estimated

| Model | Specification | DiD Estimate | SE |
|-------|--------------|--------------|-----|
| 1 | Basic OLS | 0.0551 | 0.0098 |
| 2 | WLS with survey weights | 0.0620 | 0.0097 |
| 3 | + Year fixed effects | 0.0610 | 0.0096 |
| 4 | + Demographics | 0.0445 | 0.0089 |
| 5 | + State fixed effects | 0.0438 | 0.0089 |
| 6 | + Robust SE (HC1) | **0.0438** | **0.0105** |

### 4.2 Preferred Specification (Model 6)
```python
model6 = smf.wls(
    'fulltime ~ treated + C(year_factor) + C(STATEFIP) + treat_post + '
    'female + married + has_children + hs_grad + some_college + college_plus',
    data=dfa_clean,
    weights=dfa_clean['PERWT']
).fit(cov_type='HC1')
```

**Key Results:**
- DiD coefficient: 0.0438 (4.38 percentage points)
- Robust SE: 0.0105
- t-statistic: 4.15
- p-value: < 0.001
- 95% CI: [0.023, 0.064]

### 4.3 Event Study
Year-by-treatment interactions (relative to 2011):
- 2006: -0.005 (n.s.)
- 2007: -0.013 (n.s.)
- 2008: 0.019 (n.s.)
- 2009: 0.017 (n.s.)
- 2010: 0.019 (n.s.)
- 2013: 0.060 ***
- 2014: 0.070 ***
- 2015: 0.043 **
- 2016: 0.095 ***

Pre-treatment coefficients are small and insignificant, supporting parallel trends assumption.

### 4.4 Heterogeneity Analysis
- By gender: Males (6.1 pp) > Females (3.0 pp)
- By education: HS+ (7.3 pp) > Less than HS (4.5 pp)

### 4.5 Alternative Outcomes
- Any employment: 5.6 pp increase (p < 0.001)
- Labor force participation: 3.9 pp increase (p < 0.001)

---

## 5. Commands Executed

### Python Analysis Script (analysis.py)
```bash
cd "C:\Users\seraf\DACA Results Task 2\replication_84"
python analysis.py
```
- Runtime: ~5 minutes
- Output: Console results + results.json

### Figure Generation (create_figures.py)
```bash
python create_figures.py
```
- Created: figure1_parallel_trends.png/pdf
- Created: figure2_event_study.png/pdf
- Created: figure3_did_visual.png/pdf
- Created: figure4_gender_heterogeneity.png/pdf

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_84.tex
pdflatex -interaction=nonstopmode replication_report_84.tex  # Second pass for refs
```
- Output: replication_report_84.pdf (31 pages)

---

## 6. Key Decisions and Justifications

### 6.1 Sample Selection
- **Decision:** Exclude 2012 from analysis
- **Justification:** DACA announced June 15, 2012; ACS data doesn't indicate collection month, making 2012 observations ambiguous regarding pre/post classification

### 6.2 Age Groups
- **Decision:** Treatment = 26-30, Control = 31-35 (in mid-2012)
- **Justification:** Per instructions, treatment group should be those eligible (under 31), control should be those just above cutoff but otherwise eligible

### 6.3 Non-Citizen Proxy
- **Decision:** Use CITIZEN = 3 as proxy for undocumented
- **Justification:** Per instructions to "assume that anyone who is not a citizen and who has not received immigration papers is undocumented"

### 6.4 Eligibility Criteria
- **Decision:** Require YRIMMIG <= 2007 and arrival before age 16
- **Justification:** Approximates DACA's continuous presence requirement and childhood arrival criterion

### 6.5 Preferred Specification
- **Decision:** Use WLS with year FE, state FE, demographics, robust SE
- **Justification:**
  - Survey weights ensure population representativeness
  - Year FE absorb time trends
  - State FE control for state-level confounders
  - Demographics improve comparability
  - Robust SE account for heteroskedasticity

### 6.6 Outcome Definition
- **Decision:** Full-time = UHRSWORK >= 35
- **Justification:** Standard BLS definition of full-time work; matches research question specification

---

## 7. Files Generated

| File | Description |
|------|-------------|
| analysis.py | Main analysis script |
| create_figures.py | Figure generation script |
| results.json | Key results in JSON format |
| figure1_parallel_trends.png/pdf | Trends by treatment status |
| figure2_event_study.png/pdf | Event study coefficients |
| figure3_did_visual.png/pdf | Pre/post bar chart |
| figure4_gender_heterogeneity.png/pdf | Gender heterogeneity |
| replication_report_84.tex | LaTeX source |
| replication_report_84.pdf | Final report (31 pages) |
| run_log_84.md | This log file |

---

## 8. Summary of Main Finding

**DACA eligibility increased full-time employment by 4.4 percentage points (95% CI: 2.3-6.4 pp, p < 0.001) among Mexican-born non-citizens who met the program's eligibility criteria.**

This represents a 7.0% increase relative to the pre-treatment employment rate of 62.5% for the treatment group. The finding is supported by:
- Parallel pre-trends (event study)
- Robustness across specifications
- Consistent effects by subgroup (gender, education)
- Similar effects on related outcomes (any employment, LFP)

---

## 9. Software Versions

- Python 3.x with pandas, numpy, statsmodels, matplotlib
- pdflatex (MiKTeX distribution)
- Windows 10/11

---

*Log completed: January 26, 2026*
