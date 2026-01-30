# DACA Replication Study - Run Log

## Session Start: 2026-01-25

### Overview
Independent replication study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Step 1: Data Dictionary Review

### Key Variables Identified:
- **YEAR**: Census year (2006-2016)
- **PERWT**: Person weight for survey weighting
- **AGE**: Age of respondent
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican, 100-107 detailed Mexican codes)
- **BPL/BPLD**: Birthplace (200=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **EMPSTAT**: Employment status (1=Employed)
- **UHRSWORK**: Usual hours worked per week (35+ = full-time)
- **STATEFIP**: State FIPS code

### DACA Eligibility Criteria (from instructions):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

### Operationalization Decisions:
- **Treatment Group**: Non-citizen, Mexican-born, Hispanic-Mexican individuals who:
  - Immigrated before age 16 (YRIMMIG - BIRTHYR < 16)
  - Were under 31 as of June 2012 (BIRTHYR >= 1981)
  - Immigrated by 2007 or earlier (YRIMMIG <= 2007)

- **Control Group**: Non-citizen, Mexican-born, Hispanic-Mexican individuals who do NOT meet all DACA criteria (e.g., arrived after age 16, or arrived after 2007)

- **Outcome**: Full-time employment (UHRSWORK >= 35)

- **Post-treatment Period**: 2013-2016 (DACA implemented June 2012; 2012 excluded due to ambiguity)

- **Pre-treatment Period**: 2006-2011 (using data from 2006 onwards as instructed)

---

## Step 2: Data Exploration

### Data File Information:
- **File**: data/data.csv
- **Total rows**: 33,851,425
- **Years covered**: 2006-2016 (ACS 1-year samples)

### Initial Filter:
- Filtered to Hispanic-Mexican (HISPAN == 1) and Mexican-born (BPL == 200): 991,261 observations

---

## Step 3: Sample Construction

### Final Analysis Sample:
After applying all restrictions:
- Working age (16-64): 771,888 observations
- Non-citizens only (CITIZEN == 3): 561,470 observations
- Excluding 2012 (mid-year implementation): 561,470 observations

### Treatment and Control Groups:
- **DACA-eligible (treatment)**: 85,466 observations
- **Non-eligible (control)**: 476,004 observations

---

## Step 4: Analysis Commands

### Python Environment:
```bash
python analysis.py
```

### Key Libraries Used:
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)

---

## Step 5: Main Results

### Preferred Model: Model 3 (DID with State and Year Fixed Effects + Controls)

**Specification:**
```
fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + male + married + educ_hs + C(STATEFIP) + C(YEAR)
```

**Main Finding:**
- **DID Coefficient**: 0.02176
- **Standard Error**: 0.00417 (robust/HC1)
- **95% CI**: [0.01358, 0.02994]
- **P-value**: < 0.0001

**Interpretation**: DACA eligibility is associated with a 2.2 percentage point increase in the probability of full-time employment, statistically significant at the 1% level.

---

## Step 6: Robustness Checks

### Alternative Control Group (arrived age 16-25):
- DID coefficient: 0.02344 (SE: 0.00438)

### Including 2012 as Post-Period:
- DID coefficient: 0.01506 (SE: 0.00392)

### Any Employment (not just full-time):
- DID coefficient: 0.03871 (SE: 0.00408)

---

## Step 7: Event Study Results

Coefficients relative to 2011 (last pre-treatment year):

| Year | Coefficient | SE | 95% CI |
|------|------------|-----|--------|
| 2006 | -0.0199 | 0.0095 | [-0.039, -0.001] |
| 2007 | -0.0068 | 0.0093 | [-0.025, 0.011] |
| 2008 | 0.0025 | 0.0094 | [-0.016, 0.021] |
| 2009 | 0.0120 | 0.0092 | [-0.006, 0.030] |
| 2010 | 0.0146 | 0.0090 | [-0.003, 0.032] |
| 2011 | 0 (ref) | - | - |
| 2013 | 0.0123 | 0.0090 | [-0.005, 0.030] |
| 2014 | 0.0163 | 0.0091 | [-0.002, 0.034] |
| 2015 | 0.0328 | 0.0090 | [0.015, 0.050] |
| 2016 | 0.0314 | 0.0092 | [0.013, 0.049] |

**Observation**: Pre-treatment coefficients (2007-2010) are close to zero and statistically insignificant, supporting parallel trends assumption. Post-treatment effects increase over time, becoming significant by 2015-2016.

---

## Step 8: Heterogeneity Analysis

### By Gender:
- Male: DID = 0.01478 (SE: 0.00562)
- Female: DID = 0.02016 (SE: 0.00601)

### By Education:
- Less than High School: DID = 0.01016 (SE: 0.00597)
- High School or More: DID = 0.02073 (SE: 0.00583)

---

## Key Decisions and Justifications

1. **Excluding 2012**: DACA was implemented on June 15, 2012. The ACS does not indicate when during the year a respondent was surveyed, so 2012 observations could be pre- or post-treatment. Excluding 2012 provides cleaner identification.

2. **Control Group Selection**: Used non-citizens who do not meet DACA criteria as the control group. This includes those who arrived after age 16 or arrived after 2007. This ensures the control group has similar undocumented status but was not eligible for DACA.

3. **Weighting**: Used PERWT (person weights) from IPUMS to make estimates representative of the target population.

4. **Robust Standard Errors**: Used HC1 (heteroskedasticity-consistent) standard errors to account for heteroskedasticity in the linear probability model.

5. **Age Restriction (16-64)**: Focused on working-age population to examine labor market outcomes meaningfully.

---

## Output Files Generated

1. `analysis.py` - Main analysis script
2. `descriptive_stats.csv` - Summary statistics by treatment group
3. `fulltime_rates_by_year.csv` - Full-time employment rates by year and eligibility
4. `event_study_coefficients.csv` - Event study coefficients
5. `replication_report_72.tex` - LaTeX report
6. `replication_report_72.pdf` - Compiled PDF report

---

## Session End: 2026-01-25
