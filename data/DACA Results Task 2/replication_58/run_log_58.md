# Run Log: DACA Replication Study (ID: 58)

## Overview
This document logs all commands and key decisions made during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexico-born individuals.

---

## Date: January 26, 2026

---

## 1. Initial Setup and Data Exploration

### 1.1 Reading Instructions
- Read `replication_instructions.docx` using Python's python-docx library
- Key requirements identified:
  - Research question: Effect of DACA on full-time employment (35+ hours/week)
  - Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
  - Control: Ages 31-35 at DACA implementation
  - Pre-period: 2006-2011
  - Post-period: 2013-2016 (excluding 2012)

### 1.2 Data Files
- Main data: `data/data.csv` (6.27 GB, 33.8 million observations)
- Data dictionary: `data/acs_data_dict.txt`
- Optional state-level data: `data/state_demo_policy.csv` (not used)

### 1.3 Key Variables Identified from Data Dictionary
| Variable | Description | Values Used |
|----------|-------------|-------------|
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | Used for eligibility |
| BIRTHYR | Birth year | Age calculation |
| BIRTHQTR | Birth quarter | Age calculation |
| UHRSWORK | Usual hours worked per week | Outcome (>=35 = full-time) |
| EMPSTAT | Employment status | 1 = Employed |
| PERWT | Person weight | Survey weighting |
| EDUC | Educational attainment | Covariate |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1=Married |

---

## 2. Sample Construction Decisions

### 2.1 Population Definition
**Decision:** Restrict to Hispanic-Mexican (HISPAN=1), Mexico-born (BPL=200), non-citizens (CITIZEN=3)

**Rationale:**
- Instructions specify "ethnically Hispanic-Mexican Mexican-born people"
- Non-citizen status used as proxy for undocumented status per instructions
- Cannot distinguish documented vs undocumented non-citizens in ACS

### 2.2 DACA Eligibility Criteria
**Decision:** Apply observable eligibility criteria to both treatment and control groups:
1. Arrived before 16th birthday: YRIMMIG <= BIRTHYR + 15
2. Continuous residence since June 2007: YRIMMIG <= 2007

**Rationale:**
- Ensures comparability between treatment and control groups
- These are the criteria observable in ACS data
- Education and criminal history criteria cannot be observed

### 2.3 Age Calculation
**Decision:** Calculate age as of June 15, 2012 using BIRTHYR and BIRTHQTR

**Formula:**
```python
age = 2012 - BIRTHYR
if BIRTHQTR in [3, 4]:  # Jul-Dec birth
    age -= 1  # Had not yet had birthday by June 15
```

**Rationale:**
- June 15 falls in Q2 (Apr-Jun)
- People born in Q1-Q2 had already had their birthday
- People born in Q3-Q4 had not yet had their birthday

### 2.4 Treatment and Control Groups
**Decision:**
- Treatment: Ages 26-30 as of June 15, 2012 + all eligibility criteria
- Control: Ages 31-35 as of June 15, 2012 + all eligibility criteria except age

**Rationale:**
- Per instructions: "eligible people who were ages 26-30 at the time when the policy went into place comprise the treated group"
- Control group "would have been eligible if not for their age"

### 2.5 Time Periods
**Decision:**
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Exclude 2012

**Rationale:**
- 2012 excluded because ACS does not record interview month
- Cannot distinguish pre/post DACA within 2012

---

## 3. Analysis Commands

### 3.1 Data Loading
```python
# Memory-efficient chunked loading due to large file size
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

chunks = []
for chunk in pd.read_csv(data_path, usecols=cols_needed, chunksize=1000000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    chunks.append(filtered)
df = pd.concat(chunks)
```

### 3.2 Sample Statistics
- Total ACS observations: 33,851,424
- After filtering (Hispanic-Mexican, Mexico-born, non-citizen): 701,347
- Meeting eligibility criteria (except age): 195,023
- Treatment group: 27,903
- Control group: 19,515
- Final analysis sample (excluding 2012): 43,238

### 3.3 Outcome Variable
```python
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
```

### 3.4 Difference-in-Differences Models

**Model 1: Basic OLS**
```python
model_basic = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
```

**Model 2: Weighted (Preferred)**
```python
model_weighted = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_analysis,
                         weights=df_analysis['PERWT']).fit()
```

**Model 3: With Covariates**
```python
model_covariates = smf.wls(
    'fulltime ~ treat + post + treat_post + female + married + educ_hs + age_centered',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit()
```

**Model 4: Year Fixed Effects**
```python
model_yearfe = smf.wls(
    'fulltime ~ treat + treat_post + year_2007 + ... + year_2016',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit()
```

**Model 5: State Fixed Effects**
```python
model_statefe = smf.wls(
    'fulltime ~ treat + post + treat_post + C(STATEFIP)',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
```

**Model 6: Full Model**
```python
model_full = smf.wls(
    'fulltime ~ treat + treat_post + female + married + educ_hs + age_centered + year_FEs',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
```

---

## 4. Key Results

### 4.1 Main Effect (Preferred Specification: Weighted DiD)
| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0590 |
| Standard Error | 0.0098 |
| 95% CI Lower | 0.0398 |
| 95% CI Upper | 0.0782 |
| P-value | < 0.001 |
| Sample Size | 43,238 |

### 4.2 Weighted Means (2x2 Table)
| Group | Pre (2006-2011) | Post (2013-2016) | Change |
|-------|-----------------|------------------|--------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| **DiD** | | | **0.059** |

### 4.3 Robustness Checks
| Specification | Coefficient | P-value |
|--------------|-------------|---------|
| Basic OLS | 0.052 | <0.001 |
| Weighted | 0.059 | <0.001 |
| With Covariates | 0.047 | <0.001 |
| Year FE | 0.057 | <0.001 |
| State FE | 0.057 | <0.001 |
| Full Model | 0.047 | <0.001 |
| Placebo (2009) | 0.006 | 0.610 |
| Alt Ages (24-28 vs 33-37) | 0.101 | <0.001 |
| Males Only | 0.046 | <0.001 |
| Females Only | 0.047 | 0.002 |

---

## 5. Output Files Generated

| File | Description |
|------|-------------|
| `daca_analysis.py` | Main analysis script |
| `results_summary.csv` | Key results for reporting |
| `regression_results.csv` | All model coefficients |
| `yearly_fulltime_rates.csv` | Year-by-year employment rates |
| `replication_report_58.tex` | LaTeX source for report |
| `replication_report_58.pdf` | Final PDF report (~20 pages) |
| `run_log_58.md` | This log file |

---

## 6. Key Analytical Decisions Summary

1. **Sample restriction:** Hispanic-Mexican (HISPAN=1), Mexico-born (BPL=200), non-citizen (CITIZEN=3)

2. **Eligibility criteria:** Arrived before age 16, continuous residence since 2007

3. **Age groups:** Treatment 26-30, Control 31-35 at June 15, 2012

4. **Outcome:** Full-time = UHRSWORK >= 35 hours/week

5. **Time periods:** Pre 2006-2011, Post 2013-2016, exclude 2012

6. **Preferred model:** Weighted DiD using PERWT survey weights

7. **Standard errors:** Heteroskedasticity-robust (HC1) for fixed effects models

8. **Interpretation:** 5.9 percentage point increase in probability of full-time employment due to DACA eligibility

---

## 7. Software and Versions

- Python 3.14
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- pdflatex/MiKTeX (LaTeX compilation)

---

## 8. Verification

- Manual DiD calculation matches regression coefficient (0.059)
- Placebo test shows no pre-trend (p = 0.61)
- Results robust across all specifications
- Effect similar for males and females

---

*End of Run Log*
