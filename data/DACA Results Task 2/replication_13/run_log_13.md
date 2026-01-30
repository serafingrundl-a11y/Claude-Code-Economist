# DACA Replication Study - Run Log

## Overview
This document logs all commands, key decisions, and analytical choices made during the replication of the DACA employment effects study.

---

## 1. Data Exploration

### Files Used
- **Main data file**: `data/data.csv` (6.3 GB, ACS data 2006-2016)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Supplemental state data**: `data/state_demo_policy.csv` (not used in analysis)

### Initial Data Inspection
```bash
head -5 data/data.csv
```

The data contains 54 columns including key variables:
- YEAR, STATEFIP, PERWT (survey identifiers/weights)
- BIRTHYR, BIRTHQTR, AGE (demographic)
- HISPAN, BPL, CITIZEN, YRIMMIG (ethnicity/immigration)
- UHRSWORK, EMPSTAT (employment)
- EDUC, MARST, SEX (controls)

---

## 2. Key Analytical Decisions

### 2.1 Sample Definition

**Decision 1: Exclude 2012**
- DACA was implemented June 15, 2012
- ACS does not record month of survey collection
- Cannot distinguish pre/post observations within 2012
- **Action**: Exclude 2012 entirely; use 2006-2011 as pre-period, 2013-2016 as post-period

**Decision 2: Hispanic-Mexican Ethnicity**
- IPUMS variable: HISPAN = 1 (Mexican)
- Consistent with research question focus on Hispanic-Mexican population

**Decision 3: Born in Mexico**
- IPUMS variable: BPL = 200 (Mexico)
- Aligns with research question specifying Mexican-born individuals

**Decision 4: Non-citizen Status as Proxy for Undocumented**
- IPUMS variable: CITIZEN = 3 (Not a citizen)
- Per instructions: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented for DACA purposes"
- This is an approximation; ACS cannot directly identify undocumented status

**Decision 5: DACA Eligibility Criteria**
- Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
- Continuous presence since June 15, 2007: YRIMMIG <= 2007
- Age requirement operationalized via birth year (see below)

**Decision 6: Treatment and Control Groups**
- Treatment: Ages 26-30 as of June 15, 2012 → Born 1982-1986
- Control: Ages 31-35 as of June 15, 2012 → Born 1977-1981
- Control group would have been eligible except for being too old

### 2.2 Outcome Variable

**Decision 7: Full-time Employment Definition**
- IPUMS variable: UHRSWORK >= 35
- Based on research question: "usually working 35 hours per week or more"
- Binary indicator: 1 if full-time, 0 otherwise

### 2.3 Model Specification

**Decision 8: Weighted Least Squares**
- Use PERWT (person weights) in WLS to produce population-representative estimates
- ACS sampling design requires weighting for valid inference

**Decision 9: Fixed Effects**
- Year fixed effects: Control for common shocks across all groups
- State fixed effects: Control for time-invariant state-level differences
- Preferred model includes both

**Decision 10: Control Variables**
- Female (SEX = 2)
- Married (MARST in {1, 2})
- Education: Less than HS (EDUC < 6), HS (EDUC = 6), Some college (EDUC in {7,8,9}), College+ (EDUC >= 10)
- Years in US (YRSUSA1)

**Decision 11: Standard Error Estimation**
- Preferred: Heteroskedasticity-robust (HC1) standard errors
- Robustness check: Standard errors clustered by state

---

## 3. Commands Executed

### 3.1 Analysis Script (analysis.py)
```python
# Key data loading with chunked reading (memory efficiency)
for chunk in pd.read_csv('data/data.csv', usecols=needed_cols, chunksize=500000):
    # Apply filters immediately
    chunk = chunk[chunk['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]
    chunk = chunk[chunk['HISPAN'] == 1]
    chunk = chunk[chunk['BPL'] == 200]
    chunk = chunk[chunk['CITIZEN'] == 3]
    chunk = chunk[chunk['BIRTHYR'].isin(range(1977, 1987))]
```

### 3.2 Main Regression Models
```python
# Model 1: Basic DiD
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()

# Model 2: Weighted
model2 = smf.wls('fulltime ~ treated + post + treated_post', data=df, weights=df['PERWT']).fit()

# Model 3: With controls
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us', data=df, weights=df['PERWT']).fit()

# Model 4: State FE
model4 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(STATEFIP)', data=df, weights=df['PERWT']).fit()

# Model 5: Year + State FE (PREFERRED)
model5 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecol + educ_college + years_in_us + C(STATEFIP)', data=df, weights=df['PERWT']).fit(cov_type='HC1')
```

### 3.3 Event Study
```python
# Create year-specific treatment effects (2011 is reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_x_{year}'] = df['treated'] * (df['YEAR'] == year).astype(int)

event_formula = 'fulltime ~ treated + C(YEAR) + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + female + married + educ_hs + educ_somecol + educ_college + years_in_us'
```

### 3.4 Figure Generation (create_figures.py)
```bash
python create_figures.py
```
Generated:
- figure1_trends.pdf
- figure2_eventstudy.pdf
- figure3_coefficients.pdf

### 3.5 LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_13.tex
pdflatex -interaction=nonstopmode replication_report_13.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_13.tex  # Third pass for references
```

---

## 4. Results Summary

### Sample Size
- Total observations: 44,725
- Treatment group (born 1982-1986): 26,591
- Control group (born 1977-1981): 18,134

### Pre-Treatment Full-Time Employment Rates (Weighted)
- Treatment group: 62.5%
- Control group: 67.1%

### Post-Treatment Full-Time Employment Rates (Weighted)
- Treatment group: 65.8%
- Control group: 64.1%

### Raw DiD Estimate (Weighted)
- Treatment change: +3.3 pp
- Control change: -2.9 pp
- **Raw DiD: +6.2 pp**

### Preferred Estimate (Model 5)
- **DiD Coefficient: 0.0472 (4.72 percentage points)**
- Robust Standard Error: 0.0105
- 95% CI: [0.0266, 0.0679]
- p-value: < 0.001

### Robustness Checks
1. Clustered SE by state: 0.0472 (SE: 0.0112)
2. Placebo test (2010): 0.0032 (SE: 0.0113, p = 0.78) - not significant
3. Male subsample: 0.0493 (SE: 0.0106)
4. Female subsample: 0.0367 (SE: 0.0147)

### Event Study (Pre-Treatment Coefficients)
- 2006: +0.005 (SE: 0.018) - not significant
- 2007: -0.012 (SE: 0.018) - not significant
- 2008: +0.018 (SE: 0.018) - not significant
- 2009: +0.010 (SE: 0.019) - not significant
- 2010: +0.016 (SE: 0.019) - not significant

All pre-treatment coefficients are small and statistically insignificant, supporting the parallel trends assumption.

---

## 5. Output Files

### Analysis Outputs
- `summary_statistics.csv` - Descriptive statistics by group/period
- `regression_results.csv` - Main regression coefficients
- `event_study_results.csv` - Event study coefficients
- `trends_data.csv` - Weighted employment trends
- `heterogeneity_results.csv` - Gender subgroup results

### Figures
- `figure1_trends.pdf` - Employment trends over time
- `figure2_eventstudy.pdf` - Event study plot
- `figure3_coefficients.pdf` - Coefficient comparison

### Report
- `replication_report_13.tex` - LaTeX source
- `replication_report_13.pdf` - Final report (20 pages)

---

## 6. Interpretation

DACA eligibility is associated with a 4.7 percentage point increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens who arrived before age 16 and have been in the US since 2007. This represents approximately a 7.5% relative increase from the baseline employment rate.

The effect is:
- Statistically significant at the 1% level
- Robust to alternative specifications
- Supported by parallel trends in the pre-period
- Not driven by spurious trends (placebo test insignificant)
- Present for both men and women

---

## 7. Limitations Noted

1. **Undocumented status approximation**: CITIZEN = 3 includes both documented and undocumented non-citizens
2. **Age-based identification**: Treatment/control groups differ by age (5 years), which may correlate with other factors
3. **External validity**: Results specific to Mexican-born Hispanic population
4. **Intensive margin only**: Full-time employment, not overall employment

---

*Log completed: 2026-01-25*
