# Replication Run Log (Replication 36)

## Project Overview
**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**Treatment Period**: DACA implemented June 15, 2012. Examining effects on full-time employment in years 2013-2016.

---

## Key Decisions Log

### Decision 1: Data Source
- **Choice**: Using provided ACS data from IPUMS (data.csv) covering 2006-2016
- **Rationale**: Instructions specify to use ACS one-year files from IPUMS, data already provided

### Decision 2: Sample Restriction
- **Choice**: Restrict to Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200)
- **Rationale**: Research question specifies "ethnically Hispanic-Mexican Mexican-born people"
- **Result**: 991,261 total observations; 755,660 in analysis sample (working-age, excluding 2012)

### Decision 3: DACA Eligibility Criteria (Treatment Definition)
Per the instructions, DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

**Operationalization**:
- Age at arrival < 16: (YRIMMIG - BIRTHYR) < 16
- Under 31 as of June 15, 2012: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR in [1,2])
- In US since June 15, 2007: YRIMMIG <= 2007
- Not a citizen: CITIZEN = 3 (Not a citizen)
- Assume non-citizens without naturalization are undocumented (per instructions)

### Decision 4: Outcome Variable
- **Variable**: UHRSWORK (usual hours worked per week)
- **Definition**: Full-time employment = UHRSWORK >= 35

### Decision 5: Estimation Strategy
- **Choice**: Difference-in-Differences (DiD)
- **Treatment Group**: DACA-eligible individuals (meeting all criteria above)
- **Control Group**: Similar Hispanic-Mexican Mexican-born individuals who are NOT DACA-eligible
- **Pre-Period**: 2006-2011 (before DACA)
- **Post-Period**: 2013-2016 (after DACA)
- **Note**: 2012 excluded due to ambiguity (DACA implemented mid-year)

### Decision 6: Age Restrictions for Analysis Sample
- **Choice**: Restrict to working-age population (18-64)
- **Rationale**: Focus on individuals who would be expected to be in the labor force

### Decision 7: Control Variables
- Age, age squared
- Sex (female indicator)
- Marital status (married indicator)
- Education level (high school, some college, college+ indicators; less than HS as reference)
- State fixed effects
- Year fixed effects

### Decision 8: Standard Errors
- **Choice**: Cluster standard errors at state level
- **Rationale**: Account for within-state correlation over time

---

## Commands and Code Execution Log

### Step 1: Data Loading and Exploration
```python
# Loaded ACS data with relevant columns
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'LABFORCE', 'UHRSWORK']
# Filtered to Hispanic-Mexican (HISPAN == 1) AND Mexican-born (BPL == 200)
# Result: 991,261 observations
```

### Step 2: Sample Construction
- Total observations: 33,851,424
- After filtering to Hispanic-Mexican, Mexican-born: 991,261
- After restricting to ages 18-64: 833,282
- After excluding 2012: 755,660

### Step 3: Variable Creation
```python
# DACA eligibility (time-invariant)
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
crit1 = df['age_at_immig'] < 16  # Arrived before age 16
crit2 = (df['BIRTHYR'] >= 1982) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([1, 2])))  # Under 31 in 2012
crit3 = df['YRIMMIG'] <= 2007  # In US since 2007
crit4 = df['CITIZEN'] == 3  # Not a citizen
df['daca_eligible'] = (crit1 & crit2 & crit3 & crit4).astype(int)

# Outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Treatment timing
df['post_daca'] = (df['YEAR'] >= 2013).astype(int)
df['daca_x_post'] = df['daca_eligible'] * df['post_daca']
```

### Step 4: Descriptive Statistics
Sample sizes:
- Pre-DACA DACA-eligible: 38,108
- Pre-DACA Control: 415,942
- Post-DACA DACA-eligible: 32,991
- Post-DACA Control: 268,619

Full-time employment rates:
- Pre-DACA eligible: 50.91%
- Post-DACA eligible: 54.68%
- Pre-DACA control: 62.16%
- Post-DACA control: 60.28%

Simple DiD: 0.0565

### Step 5: Main Analysis Results

| Model | Controls | DiD Coef | SE | 95% CI | p-value | N |
|-------|----------|----------|----|----|---------|---|
| 1 | None | 0.0565 | 0.0039 | [0.049, 0.064] | <0.001 | 755,660 |
| 2 | Demographics | 0.0148 | 0.0035 | [0.008, 0.022] | <0.001 | 755,660 |
| 3 | Demo + Educ | 0.0125 | 0.0035 | [0.006, 0.019] | <0.001 | 755,660 |
| 4 | Full + State/Year FE | 0.0065 | 0.0035 | [-0.000, 0.013] | 0.063 | 755,660 |
| 5 | Full + Clustered SE | 0.0065 | 0.0050 | [-0.003, 0.016] | 0.191 | 755,660 |
| 6 | Weighted + Clustered | 0.0089 | 0.0040 | [0.001, 0.017] | 0.028 | 755,660 |

**Preferred Specification (Model 5)**:
- Effect size: 0.0065 (0.65 percentage points)
- Standard error (clustered): 0.0050
- 95% CI: [-0.0032, 0.0163]
- p-value: 0.1906
- Not statistically significant at conventional levels

### Step 6: Robustness Checks

| Robustness Check | DiD Coef | Clustered SE | N |
|------------------|----------|--------------|---|
| Age 16-40 | 0.0024 | 0.0039 | 431,062 |
| Males only | 0.0032 | 0.0038 | 399,807 |
| Females only | 0.0009 | 0.0083 | 355,853 |
| Include 2012 | -0.0009 | 0.0035 | 833,282 |
| Placebo (2009) | -0.0162 | 0.0037 | 454,050 |

Note: Placebo test is significant, which may indicate pre-trends.

---

## File Outputs
- `analysis.py` - Main analysis script
- `analysis_results.pkl` - Pickled results dictionary
- `event_study_results.csv` - Event study coefficients
- `descriptive_stats.csv` - Descriptive statistics table
- `event_study.png` - Event study figure
- `trends.png` - Parallel trends figure
- `replication_report_36.tex` - Main LaTeX report
- `replication_report_36.pdf` - Compiled PDF report
- `run_log_36.md` - This log file

---

## Notes
- Using Python (pandas, statsmodels) for analysis
- Person weights (PERWT) used in weighted specification
- ACS is repeated cross-section, not panel data
- Event study shows some pre-trends which complicate causal interpretation
- Main effect (0.65 pp increase in full-time employment) is small and not statistically significant with clustered SEs

