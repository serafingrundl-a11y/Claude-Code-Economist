# Run Log for DACA Replication Study - Replication 11

## Overview
This log documents the independent replication of a study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (usually working 35+ hours per week)?

## Identification Strategy
- **Treatment Group**: Individuals aged 26-30 at the time DACA was implemented (June 15, 2012)
- **Control Group**: Individuals aged 31-35 at the time DACA was implemented
- **Method**: Difference-in-Differences (DiD) comparing pre-treatment (2006-2011) to post-treatment (2013-2016) periods
- **Note**: 2012 is excluded due to inability to distinguish pre/post DACA observations within that year

## Data Sources
- American Community Survey (ACS) 2006-2016 from IPUMS USA
- Data file: data.csv (33,851,425 rows)
- Data dictionary: acs_data_dict.txt
- Optional state-level policy data: state_demo_policy.csv

---

## Session Log

### Step 1: Data Exploration
**Date/Time**: Session Start

**Files Examined**:
- `replication_instructions.docx` - Contains research question and methodology guidance
- `data.csv` - Main ACS data (33,851,424 observations + header)
- `acs_data_dict.txt` - Variable definitions and codes
- `state_demo_policy.csv` - State-level policy information

**Key Variables Identified**:
- `YEAR` - Survey year (2006-2016)
- `BIRTHYR` - Birth year
- `BIRTHQTR` - Birth quarter (1-4)
- `HISPAN` - Hispanic origin (1 = Mexican)
- `BPL` - Birthplace (200 = Mexico)
- `CITIZEN` - Citizenship status (3 = Not a citizen)
- `YRIMMIG` - Year of immigration
- `UHRSWORK` - Usual hours worked per week (35+ = full-time)
- `EMPSTAT` - Employment status (1 = Employed)
- `PERWT` - Person weight for population estimates

### Step 2: Define DACA Eligibility Criteria
**DACA eligibility requirements**:
1. Arrived in the US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Not a citizen (CITIZEN = 3 in ACS)
5. Mexican-born (BPL = 200) and Hispanic-Mexican (HISPAN = 1)

**Age Group Definitions**:
- Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control: Born 1977-1981 (ages 31-35 on June 15, 2012)

**Key Decision**: Use BIRTHYR to calculate age at DACA implementation. Given that ACS does not have exact interview dates and we cannot precisely determine age on June 15, 2012, I use birth year ranges:
- Treatment: BIRTHYR in [1982, 1986] - approximately ages 26-30
- Control: BIRTHYR in [1977, 1981] - approximately ages 31-35

### Step 3: Data Cleaning and Sample Construction

**Sample Restrictions Applied (in order)**:
1. Exclude year 2012 (cannot distinguish pre/post DACA)
2. Restrict to Hispanic-Mexican ethnicity (HISPAN = 1)
3. Restrict to Mexican-born (BPL = 200)
4. Restrict to non-citizens (CITIZEN = 3) as proxy for undocumented status
5. Restrict to ages 26-35 at DACA implementation (age_at_daca = 2012 - BIRTHYR)
6. Require valid immigration year (YRIMMIG > 0)
7. Restrict to arrived before age 16 (YRIMMIG - BIRTHYR < 16)
8. Restrict to in US since 2007 (YRIMMIG <= 2007)

**Final Sample Size**: 44,725 observations
- Treatment group (26-30): 26,591 (59.5%)
- Control group (31-35): 18,134 (40.5%)
- Pre-DACA (2006-2011): 29,326 (65.6%)
- Post-DACA (2013-2016): 15,399 (34.4%)

### Step 4: Variable Construction

**Outcome Variable**:
- `fulltime`: Binary indicator = 1 if EMPSTAT == 1 (employed) AND UHRSWORK >= 35

**Treatment Variables**:
- `treat`: Binary = 1 if age_at_daca in [26, 30]
- `post`: Binary = 1 if YEAR >= 2013
- `treat_post`: Interaction term = treat * post (DiD estimator)

**Control Variables**:
- `female`: Binary = 1 if SEX == 2
- `married`: Binary = 1 if MARST == 1 (married, spouse present)
- Education: Categorical using EDUC variable

### Step 5: Difference-in-Differences Analysis

**Simple DiD (Weighted Means)**:
- Treatment Pre: 55.97%
- Treatment Post: 62.00%
- Treatment Change: +6.03 pp
- Control Pre: 61.11%
- Control Post: 59.82%
- Control Change: -1.29 pp
- **DiD Estimate: +7.31 percentage points**

**Regression Models**:

| Model | Specification | Coefficient | SE | p-value |
|-------|--------------|-------------|-----|---------|
| 1 | Basic (unweighted) | 0.0592 | 0.0100 | <0.001 |
| 2 | Basic (weighted) | 0.0731 | 0.0099 | <0.001 |
| 3 | + Demographics | 0.0576 | 0.0092 | <0.001 |
| 4 | + Year FE | 0.0567 | 0.0092 | <0.001 |
| 5 | + State FE + Robust SE | 0.0562 | 0.0110 | <0.001 |

**Preferred Estimate (Model 5)**:
- Effect: 5.62 percentage points
- Standard Error: 0.0110 (robust)
- 95% CI: [3.48, 7.77] percentage points
- t-statistic: 5.14
- p-value: <0.0001

### Step 6: Robustness Checks

**A. Alternative Age Bandwidth (27-29 vs 32-34)**:
- Effect: 0.0535 (SE: 0.0139)
- N: 26,792
- Conclusion: Similar to main estimate

**B. Heterogeneity by Gender**:
- Males: 0.0624 (SE: 0.0136), N = 25,058
- Females: 0.0313 (SE: 0.0174), N = 19,667
- Conclusion: Larger effect for males

**C. Pre-Trends Test**:
- Treatment x Pre-Trend: 0.0032 (SE: 0.0038)
- p-value: 0.408
- Conclusion: No evidence of differential pre-trends

**D. Event Study**:
Year-specific effects (relative to 2011):
- 2006: -0.008 (SE: 0.023)
- 2007: -0.008 (SE: 0.023)
- 2008: +0.014 (SE: 0.023)
- 2009: +0.008 (SE: 0.024)
- 2010: +0.014 (SE: 0.024)
- 2013: +0.063 (SE: 0.025)
- 2014: +0.063 (SE: 0.025)
- 2015: +0.041 (SE: 0.025)
- 2016: +0.070 (SE: 0.025)
- Conclusion: Pre-treatment effects near zero, post-treatment effects positive

**E. Alternative Outcome (Any Employment)**:
- Effect: 0.0436 (SE: 0.0101)
- Conclusion: Positive effect on overall employment as well

---

## Key Decisions

1. **Year 2012 Exclusion**: Excluded because ACS does not record interview month, making it impossible to distinguish pre- vs post-DACA observations.

2. **Age Calculation**: Used BIRTHYR to calculate age at DACA implementation. Did not use BIRTHQTR for more precise age calculation due to ambiguity in ACS interview timing.

3. **Undocumented Proxy**: Used CITIZEN = 3 (Not a citizen) as proxy for undocumented status, per instructions.

4. **DACA Eligibility Criteria**: Applied all verifiable criteria:
   - Arrived before age 16
   - In US since 2007
   - Non-citizen

5. **Weighting**: Used PERWT (person weights) in all main analyses for population-representative estimates.

6. **Standard Errors**: Used heteroskedasticity-robust standard errors (HC1) for inference.

7. **Fixed Effects**: Included state and year fixed effects to control for state-specific and time-specific shocks.

---

## Commands Executed

```python
# Data loading (chunked to manage memory)
for chunk in pd.read_csv(data_path, usecols=use_cols, chunksize=500000):
    # Apply sample restrictions
    chunk = chunk[chunk['YEAR'] != 2012]
    chunk = chunk[chunk['HISPAN'] == 1]
    chunk = chunk[chunk['BPL'] == 200]
    chunk = chunk[chunk['CITIZEN'] == 3]
    chunk['age_at_daca'] = 2012 - chunk['BIRTHYR']
    chunk = chunk[(chunk['age_at_daca'] >= 26) & (chunk['age_at_daca'] <= 35)]
    chunk = chunk[chunk['YRIMMIG'] > 0]
    chunk['age_at_immig'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['age_at_immig'] < 16]
    chunk = chunk[chunk['YRIMMIG'] <= 2007]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

# Create analysis variables
df['treat'] = ((df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 30)).astype(int)
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['treat_post'] = df['treat'] * df['post']
df['fulltime'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)

# Main regression (preferred specification)
model = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

---

## Output Files

1. `analysis.py` - Main Python analysis script
2. `analysis_output.txt` - Full output from analysis
3. `replication_report_11.tex` - LaTeX replication report
4. `replication_report_11.pdf` - Compiled PDF report
5. `run_log_11.md` - This log file
