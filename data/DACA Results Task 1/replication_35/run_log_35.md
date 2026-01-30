# Run Log - DACA Effect on Full-Time Employment Replication

## Project Overview
- **Research Question**: Effect of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US
- **Data Source**: American Community Survey (ACS) 2006-2016 via IPUMS
- **Methodology**: Difference-in-Differences with individual-level controls and state/year fixed effects

## Key Decisions and Rationale

### 1. Sample Selection
- **Population**: Hispanic-Mexican ethnicity (HISPAN=1), born in Mexico (BPL=200)
- **Age restriction**: Working-age population (16-65 years)
- **Citizenship**: Non-citizens only (CITIZEN=3) - necessary because citizens cannot be DACA-eligible
- **Rationale**: Focus on the target population most relevant to DACA policy; non-citizen restriction ensures clean comparison between eligible and non-eligible groups

### 2. Treatment Definition (DACA Eligibility)
Based on actual DACA requirements applied to the data:
1. Arrived in US before age 16 (age_at_immig < 16)
2. Under 31 as of June 15, 2012 (BIRTHYR > 1981)
3. In US since at least June 15, 2007 (YRIMMIG <= 2007)
4. Not a US citizen (CITIZEN == 3)

**Rationale**: These criteria follow the actual DACA eligibility requirements. The instruction notes that non-citizens without immigration papers should be assumed undocumented, which aligns with our approach.

### 3. Control Group Definition
Non-citizens who fail at least one of the DACA eligibility criteria:
- Arrived after age 16, OR
- Too old (born 1981 or earlier), OR
- Arrived in US after 2007

**Rationale**: This creates a comparison group of similar Mexican immigrants who were not eligible for DACA benefits.

### 4. Time Period Handling
- **Pre-period**: 2006-2011 (6 years)
- **Post-period**: 2013-2016 (4 years)
- **Excluded**: 2012 (DACA implemented June 15, 2012; cannot distinguish pre/post within year)

**Rationale**: ACS does not provide month of survey, so 2012 observations cannot be classified as pre or post treatment.

### 5. Outcome Variable
- Full-time employment: UHRSWORK >= 35 hours per week
- Binary indicator (1 if full-time, 0 otherwise)

**Rationale**: Follows standard BLS definition of full-time work; directly measures the labor market outcome most relevant to DACA's work authorization benefit.

### 6. Estimation Strategy
Linear probability model (LPM) with:
- Treatment indicator (DACA eligible)
- Post-period indicator (Year >= 2013)
- Difference-in-differences interaction term
- Person weights (PERWT)
- Heteroskedasticity-robust standard errors (HC1)

**Rationale**: LPM provides easily interpretable marginal effects; weighting ensures population representativeness; robust SEs account for heteroskedasticity in binary outcomes.

### 7. Control Variables
- Age and age-squared (capture lifecycle effects)
- Sex (male indicator)
- Marital status (married indicator)
- Years in US (immigrant assimilation)

**Rationale**: These demographics affect employment and may differ across treatment/control groups; including them reduces omitted variable bias.

### 8. Fixed Effects
- Year fixed effects (absorb common macroeconomic shocks)
- State fixed effects (absorb time-invariant state-level differences)

**Rationale**: Control for aggregate employment trends and cross-state variation in labor markets.

## Commands Executed

### Data Loading and Preprocessing
```python
# Load data in chunks due to large file size (6.3 GB)
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

for chunk in pd.read_csv(data_path, usecols=cols_needed, chunksize=500000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
```

### Sample Restrictions
```python
# Exclude 2012
df = df[df['YEAR'] != 2012]

# Working age
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 65)]

# Non-citizens only
df = df[df['CITIZEN'] == 3]

# Valid immigration year
df = df[df['YRIMMIG'] > 0]
```

### Variable Construction
```python
# DACA eligibility
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['daca_eligible'] = (
    (df['age_at_immig'] < 16) &
    (df['BIRTHYR'] > 1981) &
    (df['YRIMMIG'] <= 2007) &
    (df['CITIZEN'] == 3)
).astype(int)

# Outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-period indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# DiD interaction
df['did'] = df['daca_eligible'] * df['post']
```

### Regression Analysis
```python
# Weighted least squares with robust SEs
model = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

# Clustered SEs by state
model_clustered = sm.WLS(y, X, weights=weights).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
```

## Sample Sizes

| Stage | N |
|-------|---|
| Original ACS data (2006-2016) | 33,851,424 |
| Hispanic-Mexican, Mexican-born | 991,261 |
| After excluding 2012 | 898,879 |
| Working age (16-65) | 778,727 |
| Non-citizens only | 564,667 |

**Final Analysis Sample**: 564,667 observations

### By Treatment/Control and Period:
| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| DACA Eligible (Treatment) | 45,433 | 36,075 |
| Not Eligible (Control) | 302,048 | 181,111 |

## Key Results

### Main Findings

**Preferred Specification (Model 4 - Full model with state and year FE):**
- **DiD Coefficient**: 0.0359 (3.59 percentage points)
- **Standard Error**: 0.00425
- **95% CI**: [0.028, 0.044]
- **p-value**: < 0.001

**Interpretation**: DACA eligibility is associated with a 3.6 percentage point increase in the probability of full-time employment, statistically significant at conventional levels.

### Model Comparison
| Model | Coefficient | SE | 95% CI |
|-------|-------------|-------|--------|
| Basic DiD | 0.1009 | 0.0046 | [0.092, 0.110] |
| + Demographics | 0.0441 | 0.0043 | [0.036, 0.052] |
| + Year FE | 0.0366 | 0.0043 | [0.028, 0.045] |
| + State FE | 0.0359 | 0.0043 | [0.028, 0.044] |
| Clustered SE | 0.0359 | 0.0042 | [0.028, 0.044] |

### Robustness Checks

1. **Clustered Standard Errors**: Results robust to state-level clustering

2. **Pre-trends**: Some evidence of differential trends in pre-period (potentially concerning)
   - 2009: 0.029 (p=0.003)
   - 2010: 0.033 (p=0.001)
   - 2011: 0.023 (p=0.019)

3. **Subgroup Analysis**:
   - Males: 0.041 (SE: 0.006)
   - Females: 0.039 (SE: 0.006)

4. **Event Study**: Shows gradual increase in treatment effect post-2012
   - 2013: 0.018
   - 2014: 0.029
   - 2015: 0.044
   - 2016: 0.046

## Caveats and Limitations

1. **Pre-trends**: The pre-treatment period shows some differential trends between treatment and control groups, which could indicate violation of parallel trends assumption.

2. **Cannot observe documentation status**: We assume all non-citizens without naturalization are potentially undocumented, but this may include some legal permanent residents.

3. **Repeated cross-section**: Cannot follow individuals over time; effects reflect composition changes as well as individual responses.

4. **Selection into eligibility**: Year of immigration is endogenous; those who immigrated earlier may differ systematically from later arrivals.

5. **General equilibrium effects**: Does not capture potential spillovers to non-eligible workers.

## Files Generated

1. `daca_analysis.py` - Main analysis script
2. `regression_results.csv` - Summary of regression coefficients
3. `descriptive_stats.csv` - Summary statistics by group
4. `event_study_results.csv` - Year-by-year treatment effects
5. `pretrend_results.csv` - Pre-trend test results
6. `replication_report_35.tex` - LaTeX report
7. `replication_report_35.pdf` - Final PDF report
8. `run_log_35.md` - This log file

## Software Environment

- Python 3.14
- pandas
- numpy
- statsmodels
- LaTeX (for report compilation)
