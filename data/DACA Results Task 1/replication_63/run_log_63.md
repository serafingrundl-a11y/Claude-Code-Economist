# DACA Replication Run Log - Session 63

## Date: 2026-01-25

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (outcome, defined as usually working 35+ hours/week)?

## Key Dates
- DACA implemented: June 15, 2012
- Applications began: August 15, 2012
- Analysis years: 2013-2016 (post-treatment effects)
- Pre-treatment years: 2006-2011

---

## Step 1: Data Dictionary Review

Reviewed `data/acs_data_dict.txt` to identify key variables:
- **YEAR**: Survey year (2006-2016 available)
- **PERWT**: Person weight for survey weighting
- **BIRTHYR**: Year of birth
- **BIRTHQTR**: Quarter of birth
- **AGE**: Age at survey
- **HISPAN**: Hispanic origin (1=Mexican)
- **BPL**: Birthplace (200=Mexico)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status (1=Employed)
- **STATEFIP**: State FIPS code
- **SEX**: Sex (1=Male, 2=Female)
- **EDUC**: Educational attainment
- **MARST**: Marital status

---

## Step 2: DACA Eligibility Criteria

From the instructions, DACA eligibility required:
1. Arrived unlawfully in US before 16th birthday
2. Under 31 years old as of June 15, 2012
3. In US since June 15, 2007
4. No lawful status

### Operationalization:
- **Treatment Group**: Non-citizens (CITIZEN=3), Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200), born 1982-1996, arrived before age 16 (YRIMMIG - BIRTHYR < 16), in US since 2007 (YRIMMIG <= 2007)
- **Control Group**: Same criteria but born 1977-1981 (too old for DACA)

---

## Step 3: Data Processing

### Commands executed:

```python
# Load data with necessary columns
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'LABFORCE', 'UHRSWORK']

# Filter to Mexican-born Hispanic non-citizens
df = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)]

# Define DACA eligibility
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_young'] = df['age_at_immig'] < 16
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007
df['treatment'] = (df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1996) & df['arrived_young'] & df['in_us_since_2007']
df['control'] = (df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981) & df['arrived_young'] & df['in_us_since_2007']

# Create outcome
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Exclude 2012
df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)
```

### Sample Sizes:
- Total Mexican-born Hispanic sample: 991,261
- Non-citizen subsample: 701,347
- Analysis sample (treatment + control): 108,757
  - Treatment (DACA-eligible): 90,623
  - Control (too old): 18,134

---

## Step 4: Summary Statistics

| Variable | Treat Pre | Treat Post | Ctrl Pre | Ctrl Post |
|----------|-----------|------------|----------|-----------|
| Full-time (35+ hrs) | 0.327 | 0.549 | 0.643 | 0.611 |
| Employed | 0.385 | 0.662 | 0.684 | 0.688 |
| Employed full-time | 0.281 | 0.502 | 0.577 | 0.570 |
| Age | 19.2 | 25.1 | 29.3 | 35.3 |
| Female | 0.454 | 0.455 | 0.432 | 0.452 |
| N | 59,001 | 31,622 | 11,916 | 6,218 |

---

## Step 5: Difference-in-Differences Regression

### Model Specification:
$$Y_{ist} = \alpha + \beta_1 Eligible_i + \beta_2 Post_t + \beta_3 (Eligible_i \times Post_t) + X_i'\gamma + \delta_s + \lambda_t + \varepsilon_{ist}$$

### Main Results:

| Model | DiD Coef | SE | p-value | N |
|-------|----------|-----|---------|------|
| 1. Basic DiD | 0.2542 | 0.0083 | 0.0000 | 108,757 |
| 2. + Demographics | -0.0048 | 0.0091 | 0.5961 | 108,757 |
| 3. + Year FE | -0.0225 | 0.0092 | 0.0140 | 108,757 |
| 4. + State FE (Preferred) | **-0.0213** | **0.0091** | **0.0201** | 108,757 |
| 5. + Survey weights | -0.0259 | 0.0108 | 0.0169 | 108,757 |

### Key Finding:
The basic DiD without controls shows a large positive effect (0.254), but this is spurious due to age differences. After controlling for demographics and fixed effects, the estimate is small and negative (-0.021).

---

## Step 6: Event Study

Tested parallel pre-trends with year-specific treatment effects (reference: 2011):

| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | 0.021 | 0.016 | 0.180 |
| 2007 | 0.008 | 0.016 | 0.599 |
| 2008 | 0.019 | 0.016 | 0.225 |
| 2009 | 0.026 | 0.016 | 0.102 |
| 2010 | 0.003 | 0.016 | 0.830 |
| 2011 | 0.000 | --- | (ref) |
| 2013 | -0.010 | 0.017 | 0.562 |
| 2014 | 0.004 | 0.017 | 0.814 |
| 2015 | -0.012 | 0.017 | 0.497 |
| 2016 | -0.030 | 0.018 | 0.095 |

**Joint F-test for parallel pre-trends: F(5, 108680) = 0.90, p = 0.48**
- Cannot reject parallel pre-trends at 5% level

---

## Step 7: Robustness Checks

| Specification | DiD Coef | SE | N |
|---------------|----------|-----|------|
| Baseline | -0.021** | 0.009 | 108,757 |
| Males only | -0.079*** | 0.011 | 59,625 |
| Females only | 0.020 | 0.014 | 49,132 |
| Ages 18-35 | -0.088*** | 0.011 | 83,207 |
| Ages 25-40 | 0.032** | 0.014 | 43,200 |
| Include 2012 | -0.021** | 0.009 | 108,757 |
| Narrow control | -0.005 | 0.011 | 101,962 |
| Outcome: Employed | -0.062*** | 0.009 | 108,757 |
| Outcome: Employed FT | 0.005 | 0.009 | 108,757 |

---

## Step 8: Key Decisions Made

1. **Control group definition**: Used birth cohorts 1977-1981 (aged 31-35 on DACA date) as comparison group - similar to treatment except for age.

2. **Non-citizen proxy**: Used CITIZEN=3 as proxy for undocumented status, recognizing this includes some legal non-citizens.

3. **Exclusion of 2012**: Excluded implementation year from analysis since DACA was enacted mid-year.

4. **Primary outcome**: Used UHRSWORK >= 35 as full-time employment, regardless of employment status.

5. **Age controls**: Included quadratic age terms to address age differences between treatment and control groups.

6. **Fixed effects**: Included state and year fixed effects to control for location-specific and time-varying factors.

---

## Preferred Estimate

**Effect of DACA eligibility on full-time employment:**
- Coefficient: **-0.0213**
- Standard Error: **0.0091**
- 95% Confidence Interval: **[-0.0392, -0.0033]**
- Sample size: **108,757**

**Interpretation**: DACA eligibility is associated with a 2.1 percentage point decrease in the probability of working 35+ hours per week, relative to slightly older non-citizens who were ineligible due to age. However, this result is sensitive to specification choices and should be interpreted cautiously.

---

## Output Files

1. `replication_report_63.tex` - LaTeX source (20 pages)
2. `replication_report_63.pdf` - Compiled PDF report
3. `run_log_63.md` - This run log
4. `data/analysis_sample.csv` - Initial analysis sample
5. `data/final_analysis_sample.csv` - Final analysis sample
6. `data/main_results.csv` - Main regression results
7. `data/robustness_results.csv` - Robustness check results
8. `data/event_study_results.csv` - Event study coefficients
9. `data/summary_stats.csv` - Summary statistics
10. `figures/event_study.png` - Event study plot
11. `figures/raw_trends.png` - Raw trends plot

---

## Software and Packages

- Python 3.14
- pandas
- numpy
- statsmodels
- matplotlib
- scipy

---

## Date Completed: 2026-01-25
