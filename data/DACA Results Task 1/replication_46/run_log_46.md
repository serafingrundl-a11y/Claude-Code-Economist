# Replication Run Log - Task 46

## Project: DACA Impact on Full-Time Employment

### Date: 2026-01-25

---

## 1. Initial Setup and Data Understanding

### 1.1 Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability of full-time employment (35+ hours/week) in years 2013-2016?

### 1.2 Data Sources
- **Primary data**: ACS 2006-2016 from IPUMS USA (data.csv)
- **Data dictionary**: acs_data_dict.txt
- **Optional supplementary**: state_demo_policy.csv (not used)

### 1.3 Key Variables Identified
| Variable | Description | Use |
|----------|-------------|-----|
| YEAR | Census year | Time period indicator |
| HISPAN | Hispanic origin | = 1 for Mexican |
| BPL | Birthplace | = 200 for Mexico |
| CITIZEN | Citizenship status | = 3 for non-citizen |
| YRIMMIG | Year of immigration | Eligibility check |
| BIRTHYR | Birth year | Eligibility check |
| BIRTHQTR | Birth quarter | Eligibility check (age calculation) |
| AGE | Age | Sample restriction |
| UHRSWORK | Usual hours worked | Outcome: >=35 = full-time |
| PERWT | Person weight | Survey weighting |

---

## 2. Analytical Design Decisions

### 2.1 DACA Eligibility Criteria (per instructions)
1. Arrived in US before 16th birthday
2. Not yet 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (arrived by 2007)
4. Present in US on June 15, 2012
5. Not a citizen or legal resident

### 2.2 Operationalization of Eligibility
- **Arrived before 16**: (YRIMMIG - BIRTHYR) < 16
- **Born after June 15, 1981**: BIRTHYR > 1981 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
- **Arrived by 2007**: YRIMMIG <= 2007
- **Valid immigration year**: YRIMMIG > 0
- **Not a citizen**: CITIZEN == 3 (applied to entire sample)

### 2.3 Sample Restrictions
- Hispanic-Mexican ethnicity: HISPAN == 1
- Born in Mexico: BPL == 200
- Non-citizen: CITIZEN == 3
- Working-age population: AGE 16-64
- Exclude 2012 (implementation year)

### 2.4 Control Group Definition
Non-eligible Mexican-born Hispanic non-citizens who:
- Arrived at age 16+ (too old at arrival), OR
- Were born on or before June 15, 1981 (too old by June 2012), OR
- Arrived after 2007 (didn't meet continuous residence)

### 2.5 Outcome Variable
- **Full-time employment**: Binary indicator = 1 if UHRSWORK >= 35, = 0 otherwise

### 2.6 Estimation Strategy
Difference-in-Differences (DID):
- **Treatment group**: DACA-eligible individuals
- **Control group**: Non-eligible but similar Mexican-born Hispanic non-citizens
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 as implementation year)

Model specification:
```
fulltime_ist = β₀ + β₁·eligible_i + β₂·post_t + β₃·(eligible_i × post_t) + X_i'γ + δ_t + λ_s + ε_ist
```

Where β₃ is the DID estimator (causal effect of DACA eligibility on full-time employment)

### 2.7 Covariates
- Age, Age²
- Female indicator
- Married indicator
- High school or more indicator
- Year fixed effects
- State fixed effects

---

## 3. Commands and Code Execution Log

### 3.1 Data Loading and Cleaning
```python
# Read data in chunks to handle large file
chunks = []
for chunk in pd.read_csv(data_path, chunksize=500000, low_memory=False):
    # Pre-filter to reduce memory: Mexican-born Hispanics only
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk_filtered)
df = pd.concat(chunks, ignore_index=True)
```

**Results:**
- Total Mexican-born Hispanic observations: 991,261
- After age restriction (16-64): 851,090
- After restricting to non-citizens: 618,640
- After excluding 2012: 561,470

### 3.2 Sample Construction

**Eligibility coding:**
```python
# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligible if all criteria met
df['eligible'] = (
    (df['age_at_immig'] < 16) &  # Arrived before 16
    ((df['BIRTHYR'] > 1981) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))) &  # Born after June 15, 1981
    (df['YRIMMIG'] <= 2007) &  # Arrived by 2007
    (df['YRIMMIG'] > 0)  # Valid immigration year
).astype(int)
```

**Distribution:**
- DACA eligible: 83,611 (14.9%)
- Non-eligible: 477,859 (85.1%)

### 3.3 Regression Analysis

**Models estimated:**
1. Basic DID (no controls)
2. DID with demographic controls
3. DID with year fixed effects
4. DID with state and year fixed effects (unweighted)
5. Weighted DID with state and year fixed effects (PREFERRED)

**Key command for preferred specification:**
```python
model5 = smf.wls(
    'fulltime ~ eligible + eligible_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
    data=df,
    weights=df['PERWT']
).fit(cov_type='HC1')
```

---

## 4. Key Decisions and Justifications

### Decision 1: Exclude 2012 from analysis
**Justification**: DACA was implemented on June 15, 2012. Since ACS data does not include month of interview, 2012 observations could be from before or after DACA implementation, creating ambiguity. Excluding 2012 provides cleaner pre/post comparison.

### Decision 2: Age restriction 16-64
**Justification**: Focus on working-age population where employment is a relevant outcome. Exclude children (<16) who are not in labor force and elderly (65+) who may be retired.

### Decision 3: Control group based on eligibility criteria
**Justification**: Using Mexican-born Hispanic non-citizens who failed to meet eligibility criteria provides a credible counterfactual, as they share similar observable characteristics but were not eligible for DACA protections.

### Decision 4: Use YRIMMIG for arrival timing
**Justification**: YRIMMIG provides the year of immigration which is needed to determine whether someone arrived before their 16th birthday and whether they had continuous presence since 2007.

### Decision 5: Use person weights (PERWT)
**Justification**: ACS weights ensure estimates are representative of the U.S. population and account for sampling design.

### Decision 6: Robust (HC1) standard errors
**Justification**: Heteroskedasticity-robust standard errors are appropriate for binary outcomes estimated with OLS/WLS.

---

## 5. Analysis Output Summary

### 5.1 Main Result (Preferred Specification)

| Statistic | Value |
|-----------|-------|
| DID Coefficient (β₃) | 0.0321 |
| Standard Error | 0.0042 |
| 95% CI | [0.0238, 0.0403] |
| P-value | < 0.001 |
| Sample Size | 561,470 |
| R-squared | 0.228 |

**Interpretation:** DACA eligibility is associated with a **3.21 percentage point increase** in the probability of full-time employment. This represents a 7.5% relative increase from the pre-DACA baseline of 43.1% for eligible individuals.

### 5.2 Descriptive Statistics

| Variable | Non-Eligible | DACA-Eligible |
|----------|-------------|---------------|
| Age (mean) | 39.52 | 22.53 |
| Female (%) | 46.1 | 44.9 |
| Married (%) | 65.5 | 25.9 |
| High School+ (%) | 10.8 | 15.8 |
| Full-time (%) | 59.4 | 45.9 |
| N | 477,859 | 83,611 |

### 5.3 Event Study Results

| Year | Coefficient | SE | p-value |
|------|------------|-----|---------|
| 2006 | -0.0161 | 0.0097 | 0.098 |
| 2007 | -0.0142 | 0.0094 | 0.131 |
| 2008 | -0.0018 | 0.0095 | 0.848 |
| 2009 | 0.0056 | 0.0094 | 0.553 |
| 2010 | 0.0069 | 0.0091 | 0.449 |
| 2011 | --- (ref) | --- | --- |
| 2013 | 0.0129 | 0.0091 | 0.154 |
| 2014 | 0.0239 | 0.0091 | 0.009 |
| 2015 | 0.0401 | 0.0091 | 0.000 |
| 2016 | 0.0422 | 0.0093 | 0.000 |

**Interpretation:** Pre-treatment coefficients are small and insignificant, supporting parallel trends. Post-treatment effects grow over time, consistent with gradual DACA take-up.

### 5.4 Robustness Checks

| Specification | Coefficient | SE | N |
|---------------|------------|-----|------|
| Restricted control (arrived 16-25) | 0.0344 | 0.0044 | 350,440 |
| Employment (extensive margin) | 0.0424 | 0.0041 | 561,470 |
| Men only | 0.0275 | 0.0055 | 303,717 |
| Women only | 0.0277 | 0.0063 | 257,753 |

All robustness checks confirm positive, significant effects.

---

## 6. Files Generated

| File | Description |
|------|-------------|
| analysis_script.py | Main Python analysis code |
| analysis_results.txt | Summary of key results |
| table1_descriptives.tex | LaTeX table of descriptive statistics |
| table2_main_results.tex | LaTeX table of main DID results |
| table3_event_study.tex | LaTeX table of event study coefficients |
| table4_robustness.tex | LaTeX table of robustness checks |
| yearly_means.csv | Full-time employment rates by year |
| replication_report_46.tex | Full LaTeX report |
| replication_report_46.pdf | Final PDF report (17 pages) |
| run_log_46.md | This log file |

---

## 7. Software Environment

- **Python version**: 3.x
- **Key packages**: pandas, numpy, statsmodels, scipy
- **LaTeX**: pdfTeX (MiKTeX)

---

## 8. Completion Time

Analysis completed: 2026-01-25 11:16

---

## 9. Final Summary

This replication study estimated the causal impact of DACA eligibility on full-time employment among Mexican-born Hispanic non-citizens using difference-in-differences analysis with ACS data from 2006-2016. The preferred estimate indicates that DACA eligibility increased full-time employment by 3.2 percentage points (95% CI: 2.4-4.0 pp, p<0.001). Event study analysis supports the parallel trends assumption. Results are robust across specifications and subgroups.
