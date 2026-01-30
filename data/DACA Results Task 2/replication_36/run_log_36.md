# DACA Replication Study - Run Log (ID: 36)

## Date: January 26, 2026

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA employment effects study.

---

## 1. Initial Setup and Data Examination

### 1.1 Reading Instructions
- Read `replication_instructions.docx` using Python docx library
- Key research question: Effect of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico
- Treatment group: Ages 26-30 as of June 15, 2012
- Control group: Ages 31-35 as of June 15, 2012

### 1.2 Data Files Identified
```
data/
├── data.csv              # Main ACS data file (6.26 GB, 33.8M observations)
├── acs_data_dict.txt     # IPUMS data dictionary
└── state_demo_policy.csv # Optional state-level data (not used)
```

### 1.3 Data Dictionary Review
Examined key variables from acs_data_dict.txt:
- YEAR: Survey year (2006-2016)
- HISPAN: Hispanic origin (1=Mexican)
- BPL: Birthplace (200=Mexico)
- CITIZEN: Citizenship status (3=Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR, BIRTHQTR: Birth year and quarter
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status (1=Employed)
- PERWT: Person weight

---

## 2. Analysis Decisions

### 2.1 Sample Definition

**Decision 1: Identifying DACA-eligible population**
- Used HISPAN=1 (Mexican Hispanic origin) AND BPL=200 (born in Mexico)
- Rationale: Per instructions, focus on "ethnically Hispanic-Mexican Mexican-born people"

**Decision 2: Proxy for undocumented status**
- Used CITIZEN=3 (not a citizen)
- Rationale: ACS does not directly identify documentation status. Per instructions: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes."
- Limitation: This includes some legal permanent residents and visa holders

**Decision 3: DACA eligibility criteria**
- Arrived before 16th birthday: (YRIMMIG - BIRTHYR) < 16
- Continuous US residence since 2007: YRIMMIG <= 2007
- Rationale: These are core DACA requirements per the instructions

**Decision 4: Age calculation for June 15, 2012**
- Age = 2012 - BIRTHYR for Q1/Q2 births (had birthday by June 15)
- Age = 2012 - BIRTHYR - 1 for Q3/Q4 births (hadn't had birthday yet)
- Rationale: More precise age determination using BIRTHQTR

### 2.2 Outcome Variable

**Decision 5: Full-time employment definition**
- Full-time = (UHRSWORK >= 35) AND (EMPSTAT == 1)
- Rationale: Per instructions, "usually working 35 hours per week or more"
- Binary indicator (0/1)

### 2.3 Time Periods

**Decision 6: Pre and post periods**
- Pre-DACA: 2006-2011
- Post-DACA: 2013-2016
- Excluded 2012 (cannot distinguish pre/post implementation)
- Rationale: Per instructions, "Examine the effects on full-time employment in the years 2013-2016"

### 2.4 Estimation Strategy

**Decision 7: Difference-in-differences design**
- Treatment: Treated=1 if age 26-30 on June 15, 2012
- Control: Treated=0 if age 31-35 on June 15, 2012
- Post indicator: Post=1 for years 2013-2016
- DiD interaction: Treated × Post

**Decision 8: Weighting**
- All regressions use PERWT (ACS person weights)
- Rationale: Ensures population-representative estimates

**Decision 9: Standard errors**
- Heteroskedasticity-robust (HC1) standard errors
- Rationale: Appropriate for cross-sectional data with possible heteroskedasticity

**Decision 10: Model specifications**
- Model 1: Basic DiD (no controls)
- Model 2: DiD + demographic controls (female, age, age², education, married, children)
- Model 3: DiD + demographic controls + year fixed effects (preferred)
- Event study: Year-specific treatment effects

---

## 3. Commands Executed

### 3.1 Data Loading and Exploration
```python
# Load data
df = pd.read_csv("data/data.csv")
# Total observations: 33,851,424
# Years: 2006-2016
```

### 3.2 Sample Filtering
```python
# Step 1: Hispanic-Mexican born in Mexico
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]
# N = 991,261

# Step 2: Non-citizens (proxy for undocumented)
df_mex = df_mex[df_mex['CITIZEN'] == 3]
# N = 701,347

# Step 3: Calculate age at June 15, 2012
def calc_age_june_2012(row):
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']
    if birth_qtr in [1, 2]:
        return 2012 - birth_year
    else:
        return 2012 - birth_year - 1

df_mex['age_june_2012'] = df_mex.apply(calc_age_june_2012, axis=1)

# Step 4: Valid immigration year
df_mex = df_mex[df_mex['YRIMMIG'] > 0]

# Step 5: Arrived before age 16
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex = df_mex[df_mex['age_at_immig'] < 16]
# N = 205,327

# Step 6: Continuous residence since 2007
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007]
# N = 195,023

# Step 7: Treatment and control groups
df_mex['treated'] = ((df_mex['age_june_2012'] >= 26) & (df_mex['age_june_2012'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_june_2012'] >= 31) & (df_mex['age_june_2012'] <= 35)).astype(int)
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)]
# N = 47,418

# Step 8: Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
# Final N = 43,238
```

### 3.3 Outcome Variable
```python
df_analysis['fulltime'] = ((df_analysis['UHRSWORK'] >= 35) &
                           (df_analysis['EMPSTAT'] == 1)).astype(int)
```

### 3.4 Regression Models
```python
import statsmodels.formula.api as smf

# Model 1: Basic DiD
model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Model 2: DiD with covariates
model2 = smf.wls('fulltime ~ treated + post + did + female + age_centered + age_sq + educ_hs + married + has_children',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Model 3: DiD with year fixed effects (preferred)
model3 = smf.wls('fulltime ~ treated + C(YEAR) + did + female + age_centered + age_sq + educ_hs + married + has_children',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
```

### 3.5 Figure Generation
```python
# Event study plot
plt.savefig("figure1_event_study.png")
plt.savefig("figure1_event_study.pdf")

# Parallel trends plot
plt.savefig("figure2_parallel_trends.png")
plt.savefig("figure2_parallel_trends.pdf")

# DiD diagram
plt.savefig("figure3_did_diagram.png")
plt.savefig("figure3_did_diagram.pdf")

# Age distribution
plt.savefig("figure4_age_distribution.png")
plt.savefig("figure4_age_distribution.pdf")
```

### 3.6 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_36.tex
pdflatex -interaction=nonstopmode replication_report_36.tex
pdflatex -interaction=nonstopmode replication_report_36.tex
```

---

## 4. Results Summary

### 4.1 Sample Sizes
- Total ACS observations: 33,851,424
- Hispanic-Mexican born in Mexico: 991,261
- Non-citizens: 701,347
- DACA-eligible (arrived <16, residence since 2007): 195,023
- Analysis sample (ages 26-35 in 2012, excluding 2012): 43,238
  - Treatment group (26-30): 25,470
  - Control group (31-35): 17,768

### 4.2 DiD Estimates

| Model | Estimate | Std. Error | 95% CI | p-value |
|-------|----------|------------|--------|---------|
| Simple DiD | 0.0642 | - | - | - |
| Model 1 (Basic) | 0.0642 | 0.012 | [0.041, 0.088] | <0.001 |
| Model 2 (Covariates) | 0.0680 | 0.015 | [0.038, 0.098] | <0.001 |
| Model 3 (Year FE) | 0.0251 | 0.016 | [-0.006, 0.057] | 0.116 |

### 4.3 Preferred Estimate
- **Model 3 (Year FE + Covariates)**
- Effect: 2.5 percentage points
- SE: 0.016
- 95% CI: [-0.6, 5.7] percentage points
- p-value: 0.116 (not statistically significant)
- Sample size: 43,238

---

## 5. Output Files Created

### 5.1 Data and Results
- `summary_stats.csv`: Summary statistics by group/period
- `regression_results.csv`: Model comparison table
- `event_study_results.csv`: Event study coefficients
- `detailed_summary.csv`: Detailed group characteristics
- `model_summaries.txt`: Full regression output
- `yearly_means.csv`: Full-time employment by year/group

### 5.2 Figures
- `figure1_event_study.png/pdf`: Event study plot
- `figure2_parallel_trends.png/pdf`: Parallel trends visualization
- `figure3_did_diagram.png/pdf`: DiD illustration
- `figure4_age_distribution.png/pdf`: Sample age distribution

### 5.3 Report
- `replication_report_36.tex`: LaTeX source (23 pages)
- `replication_report_36.pdf`: Final compiled report

### 5.4 Code
- `daca_analysis.py`: Main analysis script
- `generate_figures.py`: Figure generation script

---

## 6. Key Limitations Noted

1. **Proxy for undocumented status**: Non-citizenship includes legal residents
2. **Intent-to-treat**: Estimates eligibility effect, not actual DACA receipt
3. **Parallel trends**: Some pre-treatment coefficients are negative (not zero)
4. **Repeated cross-section**: Not same individuals over time
5. **Specification sensitivity**: Year FE substantially reduce the estimate

---

## 7. Software Versions

- Python: 3.x
- pandas: 2.3.3
- numpy: standard
- statsmodels: standard
- matplotlib: standard
- LaTeX: MiKTeX 25.12, pdfTeX 3.141592653

---

## 8. Session Timeline

1. Read instructions and data dictionary
2. Designed sample selection criteria
3. Wrote and executed main analysis script (daca_analysis.py)
4. Generated figures (generate_figures.py)
5. Wrote LaTeX report (replication_report_36.tex)
6. Compiled PDF (3 passes for cross-references)
7. Created this run log

---

*End of Run Log*
