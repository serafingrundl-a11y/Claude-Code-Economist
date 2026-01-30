# Run Log for DACA Replication Study (ID: 53)

## Overview
This document logs all commands executed and key decisions made during the independent replication of the DACA effect on full-time employment study.

**Date:** January 25, 2026
**Analysis Language:** Python 3 with pandas, numpy, statsmodels, scipy, matplotlib

---

## 1. Data Exploration

### 1.1 Initial File Inspection
```bash
# List files in project directory
ls -la "C:\Users\seraf\DACA Results Task 1\replication_53\data"
# Output: data.csv (6.27 GB), acs_data_dict.txt, state_demo_policy.csv
```

### 1.2 Data Dictionary Review
- Reviewed `acs_data_dict.txt` for variable definitions
- Key variables identified:
  - YEAR: Census/survey year (2006-2016)
  - HISPAN: Hispanic origin (1 = Mexican)
  - BPL: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - BIRTHYR: Year of birth
  - BIRTHQTR: Quarter of birth
  - YRIMMIG: Year of immigration
  - UHRSWORK: Usual hours worked per week
  - AGE, SEX, EDUCD, MARST, NCHILD: Demographics

### 1.3 Data Structure Check
```bash
head -5 data/data.csv
# Confirmed CSV format with 54 columns, header row present
```

---

## 2. Key Analytical Decisions

### 2.1 Sample Selection

**Decision:** Focus on Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200), non-citizen (CITIZEN=3) population.

**Rationale:**
- DACA-eligible population is predominantly Mexican-origin
- Non-citizens are the relevant population for DACA (citizens are already authorized to work)
- This creates a homogeneous comparison group

### 2.2 DACA Eligibility Criteria

**Decision:** Define DACA eligibility based on three observable criteria:

1. **Age requirement:** Under 31 years old on June 15, 2012
   - Calculated as: `age_2012 = 2012 - BIRTHYR`
   - Adjusted for birth quarter: those born in Q3/Q4 are one year younger on June 15

2. **Arrival age requirement:** Arrived before 16th birthday
   - Calculated as: `age_at_arrival = YRIMMIG - BIRTHYR < 16`

3. **Continuous residence:** In US since June 15, 2007
   - Proxied by: `YRIMMIG <= 2007`

**Rationale:**
- These are the key DACA eligibility criteria we can observe in ACS data
- Cannot observe education requirements or criminal history (assume met)
- Cannot verify continuous physical presence (proxy with year of immigration)

### 2.3 Treatment Period Definition

**Decision:**
- Pre-period: 2006-2011
- Post-period: 2013-2016
- 2012 excluded as transition year

**Rationale:**
- DACA was announced June 15, 2012 and applications began August 15, 2012
- ACS does not record month of survey, so 2012 observations mix pre- and post-treatment
- Excluding 2012 provides cleaner identification

### 2.4 Outcome Variable

**Decision:** Full-time employment = UHRSWORK >= 35 hours per week

**Rationale:**
- Standard BLS definition of full-time work
- Captures the extensive margin effect of work authorization
- More policy-relevant than any employment (which could include informal work)

### 2.5 Working-Age Sample Restriction

**Decision:** Restrict to ages 16-64

**Rationale:**
- Standard working-age population definition
- Excludes children and typical retirement ages
- Consistent with labor economics literature

### 2.6 Control Variables

**Decision:** Include the following controls:
- Age and age-squared (life-cycle effects)
- Sex (gender differences in labor force participation)
- Education categories (less than HS, HS, some college, BA+)
- Marital status
- Number of children
- Year fixed effects
- State fixed effects

**Rationale:**
- These are standard demographic controls in labor economics
- Year FE absorb common time trends
- State FE absorb regional differences in labor markets and immigration enforcement

---

## 3. Analysis Commands

### 3.1 Main Analysis Script
```python
# analysis.py - Key operations

# Load data
df = pd.read_csv('data/data.csv')
# Total observations: 33,851,424

# Filter to sample population
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)]  # 991,261 obs
df_mex_nc = df_mex[df_mex['CITIZEN'] == 3]  # 701,347 obs

# Create outcome variable
df_mex_nc['fulltime'] = (df_mex_nc['UHRSWORK'] >= 35).astype(int)

# Create DACA eligibility
df_mex_nc['age_2012'] = 2012 - df_mex_nc['BIRTHYR']
df_mex_nc.loc[df_mex_nc['BIRTHQTR'].isin([3, 4]), 'age_2012'] -= 1
df_mex_nc['under31_2012'] = (df_mex_nc['age_2012'] < 31)
df_mex_nc['age_at_arrival'] = df_mex_nc['YRIMMIG'] - df_mex_nc['BIRTHYR']
df_mex_nc['arrived_before_16'] = (df_mex_nc['age_at_arrival'] < 16)
df_mex_nc['in_us_since_2007'] = (df_mex_nc['YRIMMIG'] <= 2007)
df_mex_nc['daca_eligible'] = (under31_2012 & arrived_before_16 & in_us_since_2007)

# Define treatment period
df_mex_nc['post'] = (df_mex_nc['YEAR'] >= 2013).astype(int)

# Exclude 2012 and restrict to working age
df_analysis = df_mex_nc[(df_mex_nc['YEAR'] != 2012) &
                        (df_mex_nc['AGE'] >= 16) & (df_mex_nc['AGE'] <= 64)]
# Final sample: 561,470 observations
```

### 3.2 Regression Models
```python
# Model 1: Basic DiD
model1 = smf.ols('fulltime ~ daca_eligible + post + treat_post', data=df_working).fit(cov_type='HC1')

# Model 5: Full specification (preferred)
model5 = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + SEX + '
                 'educ_hs + educ_somecoll + educ_ba + married + has_children + '
                 'C(YEAR) + C(STATEFIP)', data=df_working).fit(cov_type='HC1')
```

### 3.3 Event Study
```python
# Create year-by-eligibility interactions
# Reference year: 2011
event_vars = ['elig_2006', 'elig_2007', 'elig_2008', 'elig_2009', 'elig_2010',
              'elig_2013', 'elig_2014', 'elig_2015', 'elig_2016']
model_event = smf.ols('fulltime ~ daca_eligible + ' + ' + '.join(event_vars) +
                      ' + controls + C(YEAR) + C(STATEFIP)', data=df_working).fit(cov_type='HC1')
```

---

## 4. Figure Generation

```python
# create_figures.py
# Figure 1: Event study plot
# Figure 2: Trends by eligibility status
# Figure 3: Age distributions
# Figure 4: Coefficient comparison across models
```

---

## 5. LaTeX Compilation

```bash
cd "C:\Users\seraf\DACA Results Task 1\replication_53"
pdflatex -interaction=nonstopmode replication_report_53.tex
pdflatex -interaction=nonstopmode replication_report_53.tex  # Second pass for refs
pdflatex -interaction=nonstopmode replication_report_53.tex  # Third pass
```

---

## 6. Results Summary

### 6.1 Sample Sizes
| Group | Pre (2006-11) | Post (2013-16) | Total |
|-------|---------------|----------------|-------|
| Ineligible | 298,978 | 178,881 | 477,859 |
| Eligible | 46,814 | 36,797 | 83,611 |
| Total | 345,792 | 215,678 | 561,470 |

### 6.2 Full-Time Employment Rates
| Group | Pre | Post | Difference |
|-------|-----|------|------------|
| Eligible | 43.1% | 49.6% | +6.5 pp |
| Ineligible | 60.4% | 57.9% | -2.5 pp |
| **DiD** | | | **+9.0 pp** |

### 6.3 Regression Results (Preferred Estimate)
- **Coefficient:** 0.0324
- **Standard Error:** 0.0035
- **95% CI:** [0.026, 0.039]
- **t-statistic:** 9.33
- **p-value:** <0.0001
- **N:** 561,470
- **R-squared:** 0.218

### 6.4 Event Study Pre-Trends Test
- F-statistic: 3.01
- p-value: 0.010
- Some evidence of small pre-trends in 2006-2007, but magnitudes small relative to post-DACA effects

---

## 7. Output Files

1. `analysis.py` - Main analysis script
2. `create_figures.py` - Figure generation script
3. `regression_results.csv` - Regression coefficient table
4. `event_study_results.csv` - Event study coefficients
5. `descriptive_stats.csv` - Summary statistics
6. `figure1_event_study.png/.pdf` - Event study figure
7. `figure2_trends.png/.pdf` - Employment trends figure
8. `figure3_age_distribution.png/.pdf` - Age distribution figure
9. `figure4_coefficients.png/.pdf` - Coefficient comparison figure
10. `replication_report_53.tex` - LaTeX source
11. `replication_report_53.pdf` - Final report (20 pages)

---

## 8. Interpretation

DACA eligibility increased the probability of full-time employment by approximately 3.2 percentage points among Mexican-born Hispanic non-citizens. This effect is:
- Statistically significant at conventional levels (p < 0.001)
- Robust across multiple specifications
- Consistent with gradual phase-in after DACA implementation
- Similar in magnitude for both men and women

The preferred estimate suggests that DACA's provision of work authorization enabled eligible individuals to access formal sector employment opportunities, increasing their likelihood of full-time work by about 7.4% relative to the pre-DACA baseline.

---

## 9. Robustness Considerations

1. **Pre-trends:** Small but statistically significant differences in 2006-2007 warrant some caution, though magnitudes are small.
2. **Eligibility measurement:** Cannot observe all DACA criteria; our estimate captures intent-to-treat.
3. **Selection:** Non-citizen sample may be affected by differential naturalization or return migration.
4. **Geographic heterogeneity:** Results may vary by state enforcement policies (not explored in detail).

---

*End of Run Log*
