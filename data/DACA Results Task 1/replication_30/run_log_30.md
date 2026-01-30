# DACA Replication Analysis - Run Log

## Date: 2026-01-25

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

---

## Data Sources

### Primary Data
- **File**: `data/data.csv` (6.27 GB)
- **Source**: American Community Survey (ACS) from IPUMS USA
- **Years**: 2006-2016 (one-year ACS samples)

### Data Dictionary
- **File**: `data/acs_data_dict.txt`

### Supplemental State Data (Optional - Not Used)
- **File**: `data/state_demo_policy.csv`

---

## Key Analytical Decisions

### 1. Sample Selection

| Step | Description | Observations |
|------|-------------|--------------|
| 1 | Load ACS data 2006-2016 | ~34 million (full file) |
| 2 | Filter to Hispanic-Mexican (HISPAN = 1) AND Mexico-born (BPL = 200) | 991,261 |
| 3 | Restrict to ages 16-64 | 851,090 |
| 4 | Exclude 2012 (transition year) | 771,888 |

**Rationale for sample restrictions:**
- Hispanic-Mexican and Mexico-born: The vast majority of DACA-eligible individuals are of Mexican origin; restricting to this population ensures more comparable treatment and control groups
- Ages 16-64: Standard working-age population
- Excluding 2012: DACA was announced on June 15, 2012, and ACS does not identify interview month, making it impossible to distinguish pre- and post-DACA observations within that year

### 2. DACA Eligibility Definition (Treatment Variable)

An individual is classified as DACA-eligible if ALL of the following criteria are met:

1. **Non-citizen**: CITIZEN = 3 ("Not a citizen")
   - Following instructions, we treat non-citizens without naturalization as potentially undocumented

2. **Arrived before age 16**: (YRIMMIG - BIRTHYR) < 16 AND (YRIMMIG - BIRTHYR) >= 0
   - Age at immigration calculated from year of immigration minus birth year

3. **Under 31 as of June 15, 2012**:
   - BIRTHYR >= 1982, OR
   - (BIRTHYR = 1981 AND BIRTHQTR in {3, 4})
   - Conservative approach: those born in Q1-Q2 of 1981 may have turned 31 before June 2012

4. **In US since June 15, 2007**: YRIMMIG <= 2007 AND YRIMMIG > 0
   - Must have immigrated in 2007 or earlier

### 3. Outcome Variable

**Full-time employment**: UHRSWORK >= 35
- UHRSWORK = Usual hours worked per week
- Full-time defined as 35+ hours per week (standard BLS definition)

### 4. Identification Strategy

**Difference-in-Differences (DiD)**
- Treatment group: DACA-eligible individuals (meet all criteria above)
- Control group: Non-eligible Mexican-born Hispanic-Mexican individuals
- Pre-period: 2006-2011
- Post-period: 2013-2016

**Key assumption**: Parallel trends - absent DACA, full-time employment would have evolved similarly for both groups

### 5. Control Variables

- Age and age squared
- Female indicator (SEX = 2)
- Married indicator (MARST in {1, 2})
- Education dummies:
  - Less than high school (EDUC < 6) - reference category
  - High school (EDUC = 6)
  - Some college (EDUC in {7, 8, 9})
  - College or higher (EDUC >= 10)
- Year fixed effects
- State fixed effects (STATEFIP)

### 6. Estimation Method

- Weighted Least Squares (WLS) using PERWT (person weights)
- Heteroskedasticity-robust standard errors (HC1)

---

## Commands Executed

### Data Loading and Processing
```python
# Read data in chunks to manage memory
needed_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK']

chunks = []
chunk_size = 1000000
for i, chunk in enumerate(pd.read_csv(data_path, usecols=needed_cols, chunksize=chunk_size)):
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
```

### Variable Construction
```python
# DACA eligibility
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['noncitizen'] = (df['CITIZEN'] == 3).astype(int)
df['arrived_before_16'] = (df['age_at_immig'] < 16) & (df['age_at_immig'] >= 0)
df['under_31_june2012'] = (
    (df['BIRTHYR'] >= 1982) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([3, 4])))
)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)

df['daca_eligible'] = (
    df['noncitizen'] &
    df['arrived_before_16'] &
    df['under_31_june2012'] &
    df['in_us_since_2007']
).astype(int)

# Outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# DiD variables
df['post'] = (df['YEAR'] >= 2013).astype(int)
df['treated'] = df['daca_eligible']
df['did'] = df['treated'] * df['post']
```

### Regression Models
```python
# Model 1: Basic DiD
model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 2: With demographic controls
model2 = smf.wls('fulltime ~ treated + post + did + AGE + age_sq + female + married +
                  educ_hs + educ_somecoll + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 3: With year fixed effects
model3 = smf.wls(f'fulltime ~ treated + did + [demographics] + [year_dummies]',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 4: With state fixed effects (PREFERRED)
model4 = smf.wls(f'fulltime ~ treated + did + [demographics] + [year_dummies] + [state_dummies]',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

---

## Results Summary

### Sample Sizes

| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) | Total |
|-------|----------------------|------------------------|-------|
| Non-eligible | 418,621 | 270,916 | 689,537 |
| DACA-eligible | 46,080 | 36,271 | 82,351 |
| Total | 464,701 | 307,187 | 771,888 |

### Full-Time Employment Rates

| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| DACA-eligible | 45.4% | 52.2% | +6.9 pp |
| Non-eligible | 63.9% | 61.8% | -2.1 pp |
| **Raw DiD** | | | **+9.0 pp** |

### Regression Results

| Model | DiD Estimate | SE | p-value |
|-------|--------------|-----|---------|
| (1) Basic DiD | 0.0896 | 0.0045 | <0.001 |
| (2) + Demographics | 0.0259 | 0.0041 | <0.001 |
| (3) + Year FE | 0.0203 | 0.0041 | <0.001 |
| (4) + State FE | **0.0196** | **0.0041** | **<0.001** |

### PREFERRED ESTIMATE (Model 4)

- **Effect**: 1.96 percentage points (0.0196)
- **Standard Error**: 0.0041
- **95% Confidence Interval**: [0.0116, 0.0276]
- **t-statistic**: 4.789
- **p-value**: < 0.001
- **Sample Size**: 771,888

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 2 percentage points, representing a 4.3% increase relative to the pre-treatment mean of 45.4% among DACA-eligible individuals.

### Robustness Checks

| Specification | DiD Estimate | SE | Significant? |
|---------------|--------------|-----|--------------|
| Non-citizens only | 0.0301 | 0.0042 | Yes |
| Young workers (18-35) | 0.0013 | 0.0049 | No |
| Males only | 0.0192 | 0.0054 | Yes |
| Females only | 0.0132 | 0.0061 | Yes |
| Placebo (2009) | 0.0025 | 0.0054 | No (p=0.644) |

### Event Study Results (Reference: 2011)

| Year | Coefficient | SE | 95% CI |
|------|-------------|-----|--------|
| 2006 | 0.005 | 0.010 | [-0.014, 0.024] |
| 2007 | 0.006 | 0.009 | [-0.013, 0.024] |
| 2008 | 0.014 | 0.009 | [-0.004, 0.032] |
| 2009 | 0.015 | 0.009 | [-0.003, 0.033] |
| 2010 | 0.017 | 0.009 | [-0.001, 0.034] |
| 2013 | 0.014 | 0.009 | [-0.003, 0.032] |
| 2014 | 0.024 | 0.009 | [0.006, 0.041] |
| 2015 | 0.039 | 0.009 | [0.022, 0.056] |
| 2016 | 0.042 | 0.009 | [0.025, 0.060] |

**Note**: Some evidence of pre-trends in 2008-2010, which warrants cautious interpretation. Post-DACA effects strengthen over time, particularly in 2015-2016.

---

## Output Files

1. **analysis.py** - Main analysis script
2. **results.pkl** - Pickled results dictionary
3. **replication_report_30.tex** - LaTeX source for report
4. **replication_report_30.pdf** - Final PDF report (22 pages)
5. **run_log_30.md** - This log file

---

## Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy
- LaTeX (pdflatex via MiKTeX)

---

## Notes and Limitations

1. **Treatment definition**: We measure DACA *eligibility*, not actual DACA receipt. Take-up was approximately 50-60% of eligible population, so estimates reflect intent-to-treat effects.

2. **Undocumented status**: ACS does not directly identify undocumented immigrants. Following instructions, we treat non-citizens as potentially undocumented.

3. **Parallel trends**: Event study shows some pre-trends in 2008-2010, possibly reflecting differential effects of Great Recession.

4. **Control group comparability**: Despite sample restrictions, treatment and control groups differ substantially in age (22.7 vs 40.0 years).

5. **Young workers subsample**: Null result for ages 18-35 may reflect control group being more affected by recession recovery timing.

---

## Conclusion

The analysis finds a statistically significant positive effect of DACA eligibility on full-time employment of approximately 2 percentage points. This represents meaningful labor market benefits from the program's work authorization provisions.
