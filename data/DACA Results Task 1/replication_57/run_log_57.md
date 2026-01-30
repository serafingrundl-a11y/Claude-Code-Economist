# Replication Run Log - DACA Full-Time Employment Analysis

## Session Start: 2026-01-25

---

## 1. Initial Setup and Data Exploration

### Task Understanding
- **Research Question**: Among ethnically Hispanic-Mexican Mexican-born people in the US, what was the causal impact of DACA eligibility (treatment) on full-time employment probability (outcome)?
- **Full-time employment defined as**: Usually working 35+ hours per week (UHRSWORK >= 35)
- **DACA implementation date**: June 15, 2012
- **Analysis period**: Examine effects in 2013-2016

### DACA Eligibility Criteria:
1. Arrived in US before their 16th birthday
2. Had not yet turned 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Non-citizen assumed to be undocumented if no immigration papers

### Key Variables (from data dictionary):
- **YEAR**: Survey year (2006-2016 available)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican, 100-107 for detailed Mexican)
- **BPL/BPLD**: Birthplace (200=Mexico, 20000=Mexico detailed)
- **CITIZEN**: Citizenship (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Birth quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **AGE**: Age at survey
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status (1=Employed)
- **PERWT**: Person weight for survey representativeness

### Data Files:
- `data/data.csv`: Main ACS data file (2006-2016 ACS samples)
- `data/acs_data_dict.txt`: Variable codebook
- `data/state_demo_policy.csv`: Optional state-level data

---

## 2. Analysis Strategy: Difference-in-Differences

### Design Overview:
- **Treatment group**: DACA-eligible Hispanic-Mexican immigrants born in Mexico
- **Control group**: Non-DACA-eligible Hispanic-Mexican immigrants born in Mexico
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA implementation)
- **Note**: 2012 excluded as DACA implemented mid-year (June 15)

### Identification Strategy:
The treatment/control distinction comes from age-based eligibility:
- Must be born after June 15, 1981 (to be under 31 on June 15, 2012)
- Must have arrived before age 16

---

## 3. Commands and Decisions Log

### 3.1 Data Loading

```python
df = pd.read_csv('data/data.csv')
# Total observations loaded: 33,851,424
# Years in data: 2006-2016
```

**Decision**: Used all available years from ACS data (2006-2016).

### 3.2 Sample Restrictions

```python
# 1. Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
# Result: 2,945,521 observations

# 2. Restrict to born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
# Result: 991,261 observations

# 3. Restrict to non-citizens (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
# Result: 701,347 observations

# 4. Exclude 2012 (DACA implementation year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
# Result: 636,722 observations

# 5. Require valid immigration year (YRIMMIG > 0)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
# Result: 636,722 observations

# 6. Restrict to working age (18-55)
df_sample = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 55)].copy()
# Final sample: 507,423 observations
```

**Key Decisions**:
- Used HISPAN=1 (Mexican) rather than HISPAND detailed codes
- Used non-citizen status as proxy for undocumented (as instructed)
- Excluded 2012 because DACA was implemented mid-year (June 15)
- Restricted to ages 18-55 for meaningful employment analysis

### 3.3 DACA Eligibility Construction

```python
# Calculate age at arrival
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Criteria 1: Arrived before age 16
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16).astype(int)

# Criteria 2: Born after June 15, 1981 (under 31 on June 15, 2012)
# Using BIRTHYR and BIRTHQTR for more precision
df_sample['under_31_june2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
).astype(int)

# Criteria 3: In US since June 15, 2007 (arrived by 2007)
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

# DACA eligible = all criteria met
df_sample['daca_eligible'] = (
    (df_sample['arrived_before_16'] == 1) &
    (df_sample['under_31_june2012'] == 1) &
    (df_sample['in_us_since_2007'] == 1)
).astype(int)
```

**Key Decisions**:
- For the age cutoff (under 31 on June 15, 2012), used BIRTHQTR to be more precise
- Since June 15 falls in Q2, individuals born in Q3 or Q4 of 1981 would still be under 31
- Criteria for education/military service and criminal history not observable in ACS data

### 3.4 Outcome Variable Definition

```python
# Full-time employment: Employed AND works 35+ hours/week
df_sample['fulltime'] = (
    (df_sample['EMPSTAT'] == 1) &
    (df_sample['UHRSWORK'] >= 35)
).astype(int)
```

**Decision**: Used UHRSWORK >= 35 as the standard definition of full-time work.

### 3.5 Treatment Period Definition

```python
# Post-DACA: 2013-2016, Pre-DACA: 2006-2011
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
```

---

## 4. Main Analysis Results

### 4.1 Sample Sizes by Group

|              | Pre-DACA | Post-DACA | Total   |
|--------------|----------|-----------|---------|
| Not Eligible | 277,261  | 158,815   | 436,076 |
| Eligible     | 38,248   | 33,099    | 71,347  |
| Total        | 315,509  | 191,914   | 507,423 |

### 4.2 Full-Time Employment Rates

|              | Pre-DACA | Post-DACA |
|--------------|----------|-----------|
| Not Eligible | 0.5570   | 0.5584    |
| Eligible     | 0.4412   | 0.4990    |

**Simple DiD**: 0.0564

### 4.3 Regression Results

| Model | Controls | DiD Coefficient | SE | p-value | N |
|-------|----------|-----------------|-----|---------|--------|
| 1 | None | 0.0564 | 0.0041 | <0.001 | 507,423 |
| 2 | +Demographics | 0.0301 | 0.0039 | <0.001 | 507,423 |
| 3 | +Year FE | 0.0227 | 0.0039 | <0.001 | 507,423 |
| 4 | +State FE | 0.0223 | 0.0038 | <0.001 | 507,423 |

**Preferred Estimate (Model 4)**:
- Effect Size: 0.0223 (2.23 percentage points)
- 95% CI: [0.0148, 0.0299]
- Standard Error: 0.0038
- p-value: < 0.0001
- Sample Size: 507,423

### 4.4 Robustness Checks

| Specification | DiD Coefficient | SE | N |
|---------------|-----------------|-----|-------|
| Alternative age (20-45) | 0.0305 | 0.0043 | 397,006 |
| Any employment outcome | 0.0416 | 0.0038 | 507,423 |
| Weighted (PERWT) | 0.0271 | 0.0046 | 507,423 |
| Males only | 0.0224 | 0.0052 | 275,552 |
| Females only | 0.0304 | 0.0056 | 231,871 |

### 4.5 Event Study Coefficients (Reference: 2011)

| Year | Coefficient | SE | p-value |
|------|-------------|------|---------|
| 2006 | -0.0016 | 0.0090 | 0.863 |
| 2007 | 0.0034 | 0.0087 | 0.697 |
| 2008 | 0.0108 | 0.0087 | 0.216 |
| 2009 | 0.0077 | 0.0086 | 0.369 |
| 2010 | 0.0094 | 0.0083 | 0.258 |
| 2011 | 0.0000 | --- | --- |
| 2013 | 0.0107 | 0.0081 | 0.185 |
| 2014 | 0.0235 | 0.0081 | 0.004 |
| 2015 | 0.0380 | 0.0081 | <0.001 |
| 2016 | 0.0391 | 0.0081 | <0.001 |

**Interpretation**: Pre-treatment coefficients are all small and statistically insignificant, supporting parallel trends assumption. Post-treatment effects emerge in 2014 and grow through 2016.

### 4.6 Placebo Test

Fake treatment year: 2009 (using only pre-DACA data 2006-2011)
- Placebo DiD: 0.0017
- SE: 0.0052
- p-value: 0.739

**Interpretation**: No spurious effects detected in pre-treatment period.

---

## 5. Output Files Generated

### Analysis Scripts:
- `analysis.py`: Main analysis script (DiD regressions, robustness checks)
- `create_figures.py`: Figure generation script

### Results Files:
- `analysis_results.json`: All numerical results in JSON format
- `summary_stats.csv`: Summary statistics by group

### Figures:
- `figure1_event_study.png/pdf`: Event study coefficients
- `figure2_trends.png/pdf`: Employment trends by group
- `figure3_did_bars.png/pdf`: DiD visualization
- `figure4_coefficients.png/pdf`: Coefficient comparison across specifications

### Report:
- `replication_report_57.tex`: LaTeX source
- `replication_report_57.pdf`: Final compiled report (22 pages)

---

## 6. Key Analytical Decisions Summary

1. **Sample Definition**: Restricted to Hispanic-Mexican, Mexican-born, non-citizen, working-age (18-55) population

2. **Treatment Definition**: DACA eligibility based on:
   - Arrival before age 16
   - Born after June 15, 1981
   - In US since 2007 (arrived by 2007)

3. **Outcome Definition**: Full-time employment = EMPSTAT==1 AND UHRSWORK>=35

4. **Identification Strategy**: Difference-in-differences comparing eligible vs. ineligible groups before and after 2012

5. **Time Periods**: Pre-DACA (2006-2011), Post-DACA (2013-2016), excluding 2012

6. **Standard Errors**: Heteroskedasticity-robust (HC1)

7. **Preferred Specification**: Model 4 with demographic controls + year FE + state FE

---

## 7. Conclusion

The analysis finds that DACA eligibility increased full-time employment probability by approximately **2.23 percentage points** (SE = 0.0038, p < 0.001). This represents about a 5% increase relative to the pre-treatment mean for the eligible group. The finding is robust across multiple specifications and passes placebo tests. Event study analysis confirms parallel pre-trends and shows effects emerging gradually after 2012.

---

## Session End: 2026-01-25
