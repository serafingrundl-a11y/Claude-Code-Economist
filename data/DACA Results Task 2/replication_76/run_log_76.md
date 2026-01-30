# Run Log - Replication 76: DACA Impact on Full-Time Employment

## Date: 2026-01-26

---

## 1. Initial Setup and Data Review

### 1.1 Replication Instructions Summary
- **Research Question**: Estimate the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US.
- **Treatment Group**: Eligible individuals aged 26-30 at policy implementation (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at policy implementation (otherwise would have been eligible)
- **Design**: Difference-in-Differences (DiD)
- **Pre-period**: 2006-2011 (excluding 2012 due to implementation mid-year)
- **Post-period**: 2013-2016

### 1.2 Data Files Available
- `data.csv`: Main ACS data file (6.26 GB)
- `acs_data_dict.txt`: Data dictionary with variable definitions
- `state_demo_policy.csv`: Optional state-level supplemental data (not used)

### 1.3 Key Variables Identified
| Variable | Description | Usage |
|----------|-------------|-------|
| YEAR | Census year | Time variable |
| PERWT | Person weight | Sampling weights |
| AGE | Age | Age group construction |
| BIRTHYR | Birth year | DACA eligibility |
| BIRTHQTR | Birth quarter | DACA eligibility refinement |
| HISPAN/HISPAND | Hispanic origin | Sample restriction (Mexican=1) |
| BPL/BPLD | Birthplace | Sample restriction (Mexico=200/20000) |
| CITIZEN | Citizenship status | DACA eligibility (non-citizen=3) |
| YRIMMIG | Year of immigration | DACA eligibility |
| UHRSWORK | Usual hours worked/week | Outcome (>=35 = full-time) |
| EMPSTAT | Employment status | Employed indicator |
| SEX | Sex | Covariate |
| EDUC/EDUCD | Education | Covariate |
| MARST | Marital status | Covariate |
| STATEFIP | State FIPS code | Geographic control |

---

## 2. DACA Eligibility Criteria

Per the instructions and DACA program rules:
1. **Hispanic-Mexican ethnicity**: HISPAN == 1 (Mexican)
2. **Born in Mexico**: BPL == 200 (or BPLD == 20000)
3. **Not a citizen**: CITIZEN == 3
4. **Arrived before 16th birthday**: YRIMMIG - BIRTHYR < 16
5. **Continuous residence since June 15, 2007**: YRIMMIG <= 2007
6. **Age requirement** (for treatment/control):
   - Treatment: Ages 26-30 on June 15, 2012 (born 1982-1986, with birth quarter consideration)
   - Control: Ages 31-35 on June 15, 2012 (born 1977-1981)

---

## 3. Analysis Commands and Decisions

### 3.1 Data Loading
```python
# Command: Load data with pandas, selecting relevant columns
import pandas as pd
cols = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'HISPAN', 'HISPAND',
        'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT', 'EMPSTATD',
        'SEX', 'EDUC', 'EDUCD', 'MARST', 'STATEFIP', 'LABFORCE']
df = pd.read_csv('data/data.csv', usecols=cols)
```

### 3.2 Sample Construction Decisions

**Decision 1**: Treatment group based on age at June 15, 2012
- Ages 26-30 at policy date means birth years 1982-1986
- For precise age calculation: someone born in Q3/Q4 1981 would be 30 on June 15, 2012
- Conservative approach: Use birth years 1982-1986 for treatment (strictly 26-30)

**Decision 2**: Control group
- Ages 31-35 at policy date means birth years 1977-1981
- Use birth years 1977-1981 for control group

**Decision 3**: Handling 2012 data
- DACA implemented June 15, 2012; ACS collected throughout year
- Cannot distinguish pre/post in 2012; exclude 2012 from analysis

**Decision 4**: Immigration timing
- Must have immigrated by 2007 to meet continuous residence requirement
- Must have immigrated before age 16

### 3.3 Sample Construction Flow
```
Total observations loaded: 33,851,424
After Hispanic-Mexican filter: 2,945,521
After Mexico birthplace filter: 991,261
After non-citizen filter: 701,347
After immigration year filter (<=2007): 654,693
After arrived before age 16 filter: 195,023
After treatment/control group filter: 49,019
After excluding 2012: 44,725
```

---

## 4. Statistical Analysis

### 4.1 Primary Model: Difference-in-Differences

Basic DiD specification:
```
FullTime_it = beta_0 + beta_1*Treat_i + beta_2*Post_t + beta_3*(Treat_i * Post_t) + epsilon_it
```

Where:
- FullTime_it = 1 if UHRSWORK >= 35, 0 otherwise
- Treat_i = 1 if aged 26-30 on June 15, 2012
- Post_t = 1 if year >= 2013
- beta_3 = DiD estimate (effect of DACA eligibility)

### 4.2 Extended Model with Covariates

```
FullTime_it = beta_0 + beta_1*Treat_i + beta_3*(Treat_i * Post_t)
            + X_it'*gamma + State_FE + Year_FE + epsilon_it
```

Covariates (X):
- Sex (female indicator)
- Education (categorical)
- Marital status (categorical)
- State fixed effects
- Year fixed effects

---

## 5. Results Summary

### 5.1 Preferred Estimate (Model 6)
- **DiD Estimate**: 0.0472 (4.72 percentage points)
- **Robust Standard Error**: 0.0105
- **95% Confidence Interval**: [0.0266, 0.0679]
- **P-value**: < 0.001
- **Sample Size**: 44,725

### 5.2 Interpretation
DACA eligibility increased the probability of full-time employment (35+ hours/week) by 4.72 percentage points among eligible Mexican-born, Hispanic-Mexican non-citizens. This represents a relative increase of approximately 7.6% from the pre-treatment baseline of 62.5%.

### 5.3 Model Progression
| Model | Controls | Estimate | SE |
|-------|----------|----------|-----|
| 1 | Basic (unweighted) | 0.0551 | 0.0098 |
| 2 | Basic (weighted) | 0.0620 | 0.0097 |
| 3 | + Demographics | 0.0491 | 0.0089 |
| 4 | + State FE | 0.0484 | 0.0089 |
| 5 | + Year FE | 0.0472 | 0.0089 |
| 6 | + Robust SE | 0.0472 | 0.0105 |

### 5.4 Descriptive Statistics
Full-time employment rates by group and period:
- Control, Pre-period: 67.1%
- Control, Post-period: 64.1% (change: -3.0 pp)
- Treatment, Pre-period: 62.5%
- Treatment, Post-period: 65.8% (change: +3.3 pp)
- **Raw DiD**: +6.3 pp

---

## 6. Robustness Checks

### 6.1 Alternative Outcome: Any Employment
- DiD Estimate: 0.0452
- SE: 0.0100
- P-value: < 0.001
- Result: Effect on overall employment is similar to full-time effect

### 6.2 Placebo Test (Ages 36-40 vs 41-45)
- DiD Estimate: 0.0132
- SE: 0.0141
- P-value: 0.349
- Result: No significant effect in placebo group, supporting validity of design

### 6.3 Heterogeneity by Gender
- Men only: 0.0482 (SE: 0.0123, p < 0.001)
- Women only: 0.0317 (SE: 0.0177, p = 0.073)
- Result: Larger effect for men, marginally significant for women

### 6.4 Event Study Coefficients (relative to 2011)
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | 0.006 | 0.022 | 0.792 |
| 2007 | -0.011 | 0.022 | 0.631 |
| 2008 | 0.017 | 0.022 | 0.452 |
| 2009 | 0.013 | 0.023 | 0.586 |
| 2010 | 0.013 | 0.023 | 0.557 |
| 2011 | 0.000 | (ref) | - |
| 2013 | 0.047 | 0.024 | 0.050 |
| 2014 | 0.054 | 0.024 | 0.027 |
| 2015 | 0.034 | 0.024 | 0.164 |
| 2016 | 0.081 | 0.024 | 0.001 |

Result: Pre-trends are flat and insignificant; effects appear in post-period

---

## 7. Files Generated

| Filename | Description |
|----------|-------------|
| run_log_76.md | This log file |
| analysis.py | Python analysis script |
| results_summary.txt | Summary of main results |
| summary_statistics.csv | Descriptive statistics |
| event_study_coefficients.csv | Event study data |
| replication_report_76.tex | LaTeX report (~20 pages) |
| replication_report_76.pdf | Compiled PDF report |

---

## 8. Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Treatment ages | 26-30 (born 1982-1986) | Per instructions; just below age 31 cutoff |
| Control ages | 31-35 (born 1977-1981) | Per instructions; just above age 31 cutoff |
| Exclude 2012 | Yes | Cannot distinguish pre/post in mid-year implementation |
| Immigration cutoff | YRIMMIG <= 2007 | Continuous residence requirement |
| Full-time definition | UHRSWORK >= 35 | Standard BLS definition |
| Undocumented proxy | CITIZEN == 3 | ACS cannot distinguish documented vs undocumented |
| Preferred model | Model 6 | Full controls with robust SE |
| Weighting | Yes (PERWT) | Population representativeness |

---

## Appendix: Session Commands

### Commands Executed:
1. Read replication_instructions.docx via Python docx library
2. Listed data folder contents
3. Read acs_data_dict.txt for variable definitions
4. Created run_log_76.md
5. Created analysis.py
6. Executed analysis.py (runtime: ~3 minutes)
7. Created replication_report_76.tex
8. Compiled PDF with pdflatex

### Software Used:
- Python 3.x with pandas, numpy, statsmodels, scipy
- pdflatex for LaTeX compilation

---

## End of Log
