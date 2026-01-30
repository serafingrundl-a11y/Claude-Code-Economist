# Run Log for DACA Replication Study (Replication 97)

## Project Overview
- **Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (working 35+ hours per week)?
- **Treatment Period**: 2013-2016 (DACA implemented June 15, 2012)
- **Data Source**: American Community Survey (ACS) from IPUMS USA, 2006-2016

## Key Decisions Log

### 1. Sample Restriction Decisions
- **Population**: Hispanic-Mexican individuals born in Mexico
  - HISPAN = 1 (Mexican)
  - BPL = 200 (Mexico)
- **Age restriction**: Working-age adults 16-64
- **Citizenship**: Non-citizens assumed undocumented (CITIZEN = 3)
  - Per instructions: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented"
- **Exclude 2012**: Cannot distinguish pre/post DACA implementation within 2012

### 2. DACA Eligibility Criteria (from instructions)
1. Arrived in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 and undocumented

### 3. Operationalization of DACA Eligibility
- **Age at arrival < 16**: Age at immigration calculated as AGE - (YEAR - YRIMMIG)
- **Under 31 as of June 2012**: Born in 1982 or later (BIRTHYR >= 1982)
- **Continuous presence since 2007**: YRIMMIG <= 2007
- **Undocumented**: CITIZEN = 3 (not a citizen)

### 4. Outcome Variable
- **Full-time employment**: UHRSWORK >= 35 (usual hours worked per week)
- Binary indicator: 1 if employed full-time, 0 otherwise

### 5. Identification Strategy
- **Difference-in-Differences (DiD)**: Compare outcomes before/after DACA between eligible and non-eligible groups
- **Treatment group**: DACA-eligible non-citizens
- **Control group**: Non-eligible Hispanic-Mexican immigrants from Mexico (e.g., arrived too old, arrived after 2007, or naturalized)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

### 6. Control Variables
- Age, age-squared
- Sex (male indicator)
- Education level (categorical)
- Marital status
- State fixed effects (51 states)
- Year fixed effects (10 years)

### 7. Regression Specifications
1. Basic DiD without controls
2. DiD with demographic controls (age, sex, marital status)
3. DiD with demographics + education controls
4. DiD with demographics, education, and year fixed effects
5. Full model with all controls, state and year fixed effects (PREFERRED)

### 8. Standard Errors
- Clustered at state level to account for within-state correlation over time

## Commands Executed

### Data Exploration
```bash
# List directory contents
ls -la data/
# Result: data.csv (6.3GB), acs_data_dict.txt, state_demo_policy.csv

# Examine column names
head -1 data/data.csv | tr ',' '\n'
# Result: 54 columns including YEAR, HISPAN, BPL, CITIZEN, YRIMMIG, UHRSWORK, etc.

# Count observations
wc -l data/data.csv
# Result: 33,851,425 rows
```

### Analysis Script Execution
```bash
python analysis_97.py
# Processing time: ~5 minutes
# Filtered to 991,261 Hispanic-Mexican Mexico-born individuals
# After removing 2012 and age restrictions: 771,888 observations
# DACA eligible: 81,508 (10.6%)
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_97.tex  # Run 1
pdflatex -interaction=nonstopmode replication_report_97.tex  # Run 2 (for TOC)
pdflatex -interaction=nonstopmode replication_report_97.tex  # Run 3 (cross-refs)
pdflatex -interaction=nonstopmode replication_report_97.tex  # Run 4 (final)
# Result: 22 pages
```

## Key Results

### Preferred Estimate (Model 5 with clustered SEs)
- **Effect size**: 0.0803 (8.03 percentage points)
- **Standard error**: 0.0035
- **95% CI**: [0.0735, 0.0871]
- **t-statistic**: 23.19
- **p-value**: < 0.0001
- **Sample size**: 771,888

### Simple DiD Table
|              | Pre-DACA | Post-DACA | Change |
|--------------|----------|-----------|--------|
| Not Eligible | 0.619    | 0.599     | -0.020 |
| Eligible     | 0.425    | 0.494     | +0.069 |
| **DiD**      |          |           | **0.089** |

### Event Study (Year-by-Year Effects, Reference: 2011)
| Year | Coefficient | SE    | Significant |
|------|-------------|-------|-------------|
| 2006 | -0.044      | 0.010 | Yes         |
| 2007 | -0.032      | 0.005 | Yes         |
| 2008 | -0.019      | 0.009 | Yes         |
| 2009 | -0.011      | 0.007 | No          |
| 2010 | -0.001      | 0.009 | No          |
| 2011 | 0.000       | (ref) | --          |
| 2013 | +0.028      | 0.007 | Yes         |
| 2014 | +0.054      | 0.012 | Yes         |
| 2015 | +0.081      | 0.007 | Yes         |
| 2016 | +0.094      | 0.007 | Yes         |

### Heterogeneity Results
| Subgroup       | Coefficient | SE    | N       |
|----------------|-------------|-------|---------|
| Male           | 0.081       | 0.004 | 408,657 |
| Female         | 0.070       | 0.006 | 363,231 |
| Less than HS   | 0.082       | 0.003 | 649,338 |
| HS or more     | 0.068       | 0.010 | 122,550 |
| Age 16-24      | 0.032       | 0.006 | 101,406 |
| Age 25-34      | 0.008       | 0.006 | 192,347 |

## Files Generated
- `analysis_97.py`: Main analysis script (Python)
- `results_97.json`: Numerical results in JSON format
- `replication_report_97.tex`: LaTeX source for report
- `replication_report_97.pdf`: Final PDF report (22 pages)
- `run_log_97.md`: This log file

## Interpretation

DACA eligibility is associated with an 8.03 percentage point increase in the probability of full-time employment among Hispanic-Mexican individuals born in Mexico. This represents approximately an 18% increase relative to the pre-DACA baseline employment rate of 42.5% among the eligible population.

The event study shows:
- Pre-trends: Some differential trends in early years (2006-2008), but the years immediately before DACA (2009-2011) show parallel trends
- Post-DACA effects: Immediate positive effect that grows over time from 2.8pp in 2013 to 9.4pp in 2016

The effects are robust across specifications and are larger for:
- Males (vs. females)
- Less educated individuals (vs. more educated)
- Younger individuals (most eligible for DACA)

## Session Notes
- Date: 2026-01-25
- Analysis performed using Python 3.14 with pandas, numpy, statsmodels, scipy
- LaTeX compilation with MiKTeX/pdfTeX
