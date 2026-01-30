# DACA Replication Study - Run Log

## Study Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of full-time employment (>=35 hrs/wk) in 2013-2016?

**Key Dates:**
- DACA enacted: June 15, 2012
- Applications began: August 15, 2012
- Analysis period: 2013-2016 (post-DACA)
- Pre-period: 2006-2011 (pre-DACA)

## Session Start
Date: 2026-01-24

---

## Step 1: Data Exploration

### Data Files Available:
- `data/data.csv` - Main ACS data file (~6GB, 33.85 million rows)
- `data/acs_data_dict.txt` - Data dictionary from IPUMS
- `data/state_demo_policy.csv` - Optional state-level data (not used)

### Key Variables Identified:

**Sample Identification:**
- YEAR: Census year (2006-2016 available)
- SAMPLE: IPUMS sample identifier

**DACA Eligibility Variables:**
- BIRTHYR: Birth year (for age calculation)
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- BPL/BPLD: Birthplace (200 = Mexico)
- HISPAN/HISPAND: Hispanic origin (1 = Mexican)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration

**Outcome Variable:**
- UHRSWORK: Usual hours worked per week (>=35 = full-time)
- EMPSTAT: Employment status (1 = Employed)

**Control Variables:**
- SEX: 1=Male, 2=Female
- AGE: Age in years
- EDUCD: Detailed education code
- MARST: Marital status
- STATEFIP: State FIPS code

**Survey Weights:**
- PERWT: Person weight for population estimates

---

## Step 2: Analysis Design Decisions

### DACA Eligibility Criteria (from instructions):
1. Arrived in US before 16th birthday
2. Had not reached 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

### Operationalization Decisions:

**Treatment Group (DACA-eligible):**
- Born in Mexico (BPL = 200)
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Non-citizen (CITIZEN = 3)
- Age at immigration < 16: (YRIMMIG - BIRTHYR) < 16
- Under 31 as of June 15, 2012: BIRTHYR >= 1982, OR (BIRTHYR = 1981 AND BIRTHQTR >= 3)
- In US since 2007: YRIMMIG <= 2007

**Control Group (too old for DACA):**
- Same criteria as treatment except:
- BIRTHYR <= 1980 (definitely over 31 by June 2012)
- Still arrived before age 16 and in US since 2007

**Rationale for Control Group Choice:**
The "too old" control group shares key characteristics with the treatment group (childhood arrival, long-term US presence, non-citizen status, Mexican origin) but was excluded from DACA solely due to birth year. This provides a cleaner comparison than alternative control groups.

**Time Periods:**
- Pre-DACA: 2006-2011 (6 years)
- Post-DACA: 2013-2016 (4 years)
- 2012 excluded: DACA implemented mid-year (June 15), ACS does not report interview month

### Identification Strategy: Difference-in-Differences

**Regression Specification:**
```
Y_ist = α + β₁*Eligible_i + β₂*Post_t + β₃*(Eligible_i × Post_t) + X_i'γ + θ_s + δ_t + ε_ist
```

Where:
- Y = 1 if full-time employed (UHRSWORK >= 35), 0 otherwise
- Eligible = 1 if DACA-eligible, 0 otherwise
- Post = 1 if year >= 2013, 0 if year <= 2011
- X = individual controls (age, age², female, married, education)
- θ_s = state fixed effects
- δ_t = year fixed effects
- β₃ = difference-in-differences estimate (parameter of interest)

---

## Step 3: Data Processing

### Commands Executed:

```python
# Load data in chunks (file is ~6GB)
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
               'EDUC', 'EDUCD', 'MARST', 'EMPSTAT', 'UHRSWORK', 'LABFORCE']

# Sequential filters applied:
df = pd.read_csv('data/data.csv', usecols=cols_needed)  # 33,851,424 rows
df = df[df['HISPAN'] == 1]                               # 2,945,521 rows
df = df[df['BPL'] == 200]                                # 991,261 rows
df = df[df['CITIZEN'] == 3]                              # 701,347 rows
df = df[df['YEAR'] != 2012]                              # 636,722 rows
```

### Sample Construction Results:
- Total ACS observations (2006-2016): 33,851,424
- After Hispanic-Mexican filter: 2,945,521
- After Mexico birthplace filter: 991,261
- After non-citizen filter: 701,347
- After excluding 2012: 636,722
- DACA eligible (all criteria): 118,374
- Control group (too old): 53,761
- Final analysis sample (treatment + control, ages 18-64): **122,695**

---

## Step 4: Variable Construction

### Treatment Variable:
```python
# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility criteria
df['arrived_before_16'] = (df['age_at_immig'] < 16) & (df['age_at_immig'] >= 0)
df['under_31_june2012'] = (df['BIRTHYR'] >= 1982) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

df['daca_eligible'] = (df['arrived_before_16'] & df['under_31_june2012'] & df['in_us_since_2007']).astype(int)
```

### Outcome Variable:
```python
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
```

### Control Variables:
```python
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['age_sq'] = df['AGE'] ** 2
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)
df['educ_hs'] = (df['EDUCD'] >= 62).astype(int)
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)
df['daca_x_post'] = df['daca_eligible'] * df['post']
```

---

## Step 5: Analysis Results

### Descriptive Statistics by Group and Period:

| Group | Period | Full-time Rate | Employed Rate | Mean Age | Female | HS+ | College+ | Married | N |
|-------|--------|----------------|---------------|----------|--------|-----|----------|---------|---|
| Control | Pre | 65.3% | 68.2% | 37.9 | 40.7% | 41.8% | 3.0% | 63.0% | 34,327 |
| Control | Post | 61.2% | 67.9% | 43.6 | 41.9% | 40.9% | 3.1% | 61.8% | 18,017 |
| Eligible | Pre | 51.1% | 58.9% | 22.2 | 44.2% | 56.9% | 1.7% | 26.9% | 37,715 |
| Eligible | Post | 54.8% | 66.4% | 25.2 | 45.4% | 64.8% | 3.3% | 33.7% | 32,636 |

### Main Regression Results:

| Model | Controls | DID Coef | Std Error | 95% CI | N |
|-------|----------|----------|-----------|--------|---|
| 1 | None | 0.0781*** | 0.0058 | [0.067, 0.090] | 122,695 |
| 2 | Demographics | 0.0155*** | 0.0056 | [0.004, 0.026] | 122,695 |
| 3 | + Education | 0.0100* | 0.0056 | [-0.001, 0.021] | 122,695 |
| 4 | + State FE | 0.0090 | 0.0056 | [-0.002, 0.020] | 122,695 |
| 5 | + Year FE | -0.0022 | 0.0056 | [-0.013, 0.009] | 122,695 |
| **6 (Preferred)** | **+ Weighted** | **0.0006** | **0.0055** | **[-0.010, 0.011]** | **122,695** |

### Preferred Estimate Summary:
- **Effect size:** 0.0006 (0.06 percentage points)
- **Standard error:** 0.0055
- **95% CI:** [-0.0103, 0.0114]
- **p-value:** 0.9145
- **Interpretation:** Not statistically significant; cannot reject null hypothesis of no effect

### Robustness Checks:

| Specification | DID Coef | Std Error | 95% CI |
|---------------|----------|-----------|--------|
| Alternative control (recent arrivals) | -0.0253*** | 0.0068 | [-0.039, -0.012] |
| Men only | -0.0190*** | 0.0069 | [-0.032, -0.006] |
| Women only | 0.0151* | 0.0089 | [-0.002, 0.033] |
| Any employment (outcome) | 0.0044 | 0.0054 | [-0.006, 0.015] |

### Event Study Results (Reference: 2011):

| Year | Coefficient | Std Error | Significant? |
|------|-------------|-----------|--------------|
| 2006 | 0.0326 | 0.0119 | Yes |
| 2007 | 0.0264 | 0.0117 | Yes |
| 2008 | 0.0419 | 0.0117 | Yes |
| 2009 | 0.0255 | 0.0116 | Yes |
| 2010 | 0.0270 | 0.0115 | Yes |
| 2011 | 0.0000 | -- | Reference |
| 2013 | 0.0201 | 0.0115 | No |
| 2014 | 0.0181 | 0.0115 | No |
| 2015 | 0.0244 | 0.0118 | Yes |
| 2016 | 0.0413 | 0.0118 | Yes |

**Parallel trends concern:** Pre-treatment coefficients (2006-2010) are all positive and statistically significant, indicating treatment and control groups had different employment trends before DACA.

---

## Step 6: Key Decisions and Rationale

1. **Excluded 2012:** DACA was implemented June 15, 2012. ACS does not report interview month, so pre/post status is ambiguous for 2012 observations.

2. **Control group = "too old":** Selected individuals who meet all DACA criteria except age (born 1980 or earlier). Alternative control groups (recent arrivals, documented immigrants) have other confounding characteristics.

3. **Age restriction 18-64:** Standard working-age population. Excludes children and elderly.

4. **Survey weights:** Used PERWT for population-representative estimates in preferred specification.

5. **State and year fixed effects:** Included to control for time-invariant state characteristics and common time trends.

6. **Linear probability model:** Used OLS/WLS for interpretability. Could consider logit/probit in robustness.

---

## Step 7: Limitations Identified

1. **Parallel trends violation:** Event study shows pre-existing differential trends between treatment and control groups.

2. **Age differences:** Treatment group is ~15-20 years younger than control group on average, despite both groups being restricted to ages 18-64.

3. **DACA eligibility measurement error:** Cannot distinguish documented from undocumented non-citizens; cannot verify continuous presence or education requirements.

4. **Intent-to-treat:** Estimates eligibility effects, not effects of actually receiving DACA.

---

## Deliverables

1. **analysis.py** - Python script with all data processing and analysis
2. **summary_statistics.csv** - Descriptive statistics by group and period
3. **regression_results.csv** - Regression coefficients across specifications
4. **event_study_results.csv** - Event study coefficients by year
5. **replication_report_01.tex** - LaTeX source for report
6. **replication_report_01.pdf** - Compiled PDF report (21 pages)
7. **run_log_01.md** - This file

---

## Session End
Date: 2026-01-24

## Final Result Summary

**Preferred Estimate:**
- Effect of DACA eligibility on full-time employment: **0.06 percentage points**
- Standard Error: **0.55 percentage points**
- 95% Confidence Interval: **[-1.03, 1.14] percentage points**
- p-value: **0.91**
- Sample Size: **122,695**
- Conclusion: **No statistically significant effect detected**
