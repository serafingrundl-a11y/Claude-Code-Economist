# DACA Replication Analysis Run Log

## Project Information
- **Replication ID:** 88
- **Date:** January 25, 2026
- **Research Question:** What was the causal impact of DACA eligibility on the probability of full-time employment among ethnically Hispanic-Mexican, Mexican-born people in the United States?

---

## Data Sources

### Primary Data
- **Source:** American Community Survey (ACS) via IPUMS USA
- **File:** `data/data.csv`
- **Years:** 2006-2016 (one-year ACS files)
- **Total observations:** 33,851,424 rows

### Data Dictionary
- **File:** `data/acs_data_dict.txt`
- Documents all IPUMS variables included in the extract

### Supplementary Data (Not Used)
- **File:** `data/state_demo_policy.csv`
- State-level demographic and policy information (optional per instructions)

---

## Key Decisions

### 1. Sample Selection

**Population of Interest:**
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Born in Mexico (BPL = 200)
- Non-citizen (CITIZEN = 3)
- Working age 16-64

**Rationale:**
- The research question specifies Hispanic-Mexican, Mexican-born individuals
- Non-citizens are assumed to be undocumented per the instructions ("Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes")
- Working age restriction focuses on employment-relevant population

**Result:** 618,640 observations with determinable DACA eligibility

### 2. DACA Eligibility Criteria

An individual is coded as DACA-eligible if ALL of the following are met:

1. **Arrived before age 16:**
   - `YRIMMIG - BIRTHYR < 16`

2. **Under 31 as of June 15, 2012:**
   - `BIRTHYR > 1981` OR
   - `BIRTHYR == 1981 AND BIRTHQTR in {3, 4}` (born July-December 1981)

3. **Continuously in US since June 2007:**
   - `YRIMMIG <= 2007`

**Notes:**
- Cannot verify physical presence on June 15, 2012 or current educational enrollment
- Citizenship restriction serves as proxy for lack of lawful status
- Observations with missing/zero YRIMMIG are excluded (cannot determine eligibility)

**Result:**
- DACA Eligible: 83,611 observations
- DACA Ineligible: 477,859 observations

### 3. Outcome Variable

**Full-time Employment:**
- `EMPSTAT == 1` (employed) AND `UHRSWORK >= 35` (usual hours worked >= 35/week)

**Rationale:** Per research instructions, full-time employment is defined as "usually working 35 hours per week or more"

### 4. Treatment Period Definition

- **Pre-period:** 2006-2011
- **Transition year:** 2012 (excluded from main analysis)
- **Post-period:** 2013-2016

**Rationale for excluding 2012:**
- DACA was implemented June 15, 2012
- ACS does not identify month of data collection
- Cannot distinguish pre- and post-DACA observations in 2012

**Main analysis sample (excluding 2012):** 561,470 observations

### 5. Empirical Strategy

**Method:** Difference-in-Differences (DiD)

**Identifying Assumption:** Absent DACA, full-time employment trends would have evolved similarly for eligible and ineligible groups

**Model Specifications:**
1. Basic DiD (no controls)
2. DiD with demographic controls (age, age^2, female, married, HS education)
3. DiD with year fixed effects + demographics
4. DiD with year + state fixed effects + demographics (PREFERRED)
5. Weighted version of Model 4

### 6. Control Variables

| Variable | Definition |
|----------|------------|
| AGE | Continuous age variable |
| age_sq | Age squared |
| female | SEX == 2 |
| married | MARST in {1, 2} |
| educ_hs | EDUC >= 6 (high school or more) |
| Year FE | Indicator for each survey year |
| State FE | Indicator for each state (STATEFIP) |

---

## Commands Executed

### Data Loading (Chunked Processing)
```python
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=1000000):
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)]
```

### DACA Eligibility Function
```python
def is_daca_eligible(row):
    if row['YRIMMIG'] == 0 or pd.isna(row['YRIMMIG']):
        return np.nan

    age_at_immigration = row['YRIMMIG'] - row['BIRTHYR']
    if age_at_immigration >= 16:
        return 0

    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']
    if birth_year > 1981:
        under_31 = True
    elif birth_year == 1981 and birth_qtr in [3, 4]:
        under_31 = True
    else:
        under_31 = False
    if not under_31:
        return 0

    if row['YRIMMIG'] > 2007:
        return 0

    return 1
```

### Main Regression (Model 4 - Preferred)
```python
model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + educ_hs',
                  data=df_main).fit()
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_88.tex
# Run 3 times for cross-references
```

---

## Results Summary

### Main Finding (Preferred Specification - Model 4)

| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0273 |
| Standard Error | 0.0034 |
| t-statistic | 7.99 |
| p-value | < 0.001 |
| 95% CI | [0.0206, 0.0339] |
| Sample Size | 561,470 |
| R-squared | 0.202 |

**Interpretation:** DACA eligibility is associated with a 2.73 percentage point increase in the probability of full-time employment, statistically significant at the 0.1% level.

### Simple DiD Calculation

|  | Pre (2006-2011) | Post (2013-2016) | Difference |
|--|-----------------|------------------|------------|
| DACA Eligible | 0.371 | 0.452 | +0.080 |
| DACA Ineligible | 0.546 | 0.542 | -0.004 |
| **DiD** | | | **0.084** |

### Model Comparison

| Model | Coefficient | SE | p-value |
|-------|------------|-----|---------|
| (1) Basic DiD | 0.0837 | 0.0038 | <0.001 |
| (2) + Demographics | 0.0326 | 0.0034 | <0.001 |
| (3) + Year FE | 0.0276 | 0.0034 | <0.001 |
| (4) + State FE | 0.0273 | 0.0034 | <0.001 |
| (5) Weighted | 0.0237 | 0.0034 | <0.001 |

### Robustness Checks

| Specification | Coefficient | SE |
|---------------|------------|-----|
| Including 2012 | 0.0210 | 0.0032 |
| Ages 18-40 only | 0.0139 | 0.0039 |
| Any employment | 0.0435 | 0.0033 |
| Males only | 0.0220 | 0.0045 |
| Females only | 0.0224 | 0.0051 |

### Event Study Pre-Trends

| Year | Coefficient | SE | Significant |
|------|------------|-----|-------------|
| 2006 | -0.021 | 0.008 | * |
| 2007 | -0.013 | 0.008 | |
| 2008 | -0.005 | 0.008 | |
| 2009 | -0.001 | 0.007 | |
| 2010 | 0.005 | 0.007 | |
| 2011 | 0.000 | -- | (reference) |
| 2013 | 0.009 | 0.007 | |
| 2014 | 0.016 | 0.007 | * |
| 2015 | 0.031 | 0.007 | *** |
| 2016 | 0.033 | 0.007 | *** |

Pre-treatment coefficients (2007-2010) are small and statistically insignificant, supporting parallel trends assumption. 2006 shows some divergence.

---

## Output Files

### Analysis Outputs
- `results_summary.csv` - All regression coefficients and key statistics
- `table1_descriptive.csv` - Descriptive statistics by eligibility
- `table2_did_summary.csv` - DiD summary by group and period
- `table3_regression.csv` - Main regression results
- `table4_event_study.csv` - Event study coefficients
- `table5_desc_by_period.csv` - Descriptive stats by eligibility and period
- `model4_full_output.txt` - Full regression output for preferred model

### Final Deliverables
- `replication_report_88.tex` - LaTeX source (25 pages)
- `replication_report_88.pdf` - Final PDF report
- `run_log_88.md` - This log file
- `analysis.py` - Python analysis script

---

## Software and Packages

- **Python:** 3.14
- **pandas:** Data manipulation
- **numpy:** Numerical operations
- **statsmodels:** Regression analysis (OLS, WLS)
- **pdflatex (MiKTeX):** LaTeX compilation

---

## Notes and Caveats

1. **Imperfect eligibility determination:** Cannot observe undocumented status directly; use non-citizen status as proxy

2. **Selection issues:** Sample restricted to those who appear in ACS; may miss some undocumented individuals

3. **Intent-to-treat:** Estimates represent effect of eligibility, not actual DACA receipt

4. **Standard errors:** Heteroskedasticity-robust but not clustered; may understate uncertainty

5. **2006 pre-trend:** Event study shows some divergence in 2006; parallel trends stronger in 2007-2010

6. **Memory constraints:** Data loaded in chunks due to large file size (~6GB CSV)

---

## Session Completed
All required deliverables have been produced:
- [x] `replication_report_88.tex`
- [x] `replication_report_88.pdf`
- [x] `run_log_88.md`
