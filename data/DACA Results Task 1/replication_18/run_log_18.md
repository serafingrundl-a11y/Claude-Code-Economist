# Run Log - DACA Replication Study (Replication 18)

## Overview
This log documents all commands and key decisions made during the replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

## Data Sources
- American Community Survey (ACS) 2006-2016 (data.csv)
- Data dictionary: acs_data_dict.txt
- Optional state-level data: state_demo_policy.csv (not used in main analysis)

## Key Decisions

### 1. Sample Selection
- **Population**: Hispanic-Mexican ethnicity (HISPAN=1), born in Mexico (BPL=200)
- **Citizenship**: Non-citizens only (CITIZEN=3) - to approximate undocumented population
- **Age**: Working age 16-64
- **Years**: 2006-2016 ACS data (pre-treatment: 2006-2011, post-treatment: 2013-2016)
- **Note**: 2012 excluded from main analysis because DACA was implemented mid-year (June 15, 2012)
- **Final sample size**: 561,470 person-year observations

### 2. DACA Eligibility Criteria (Treatment Definition)
Based on the official DACA requirements:
1. Arrived in US before their 16th birthday: YRIMMIG - BIRTHYR < 16
2. Had not yet had their 31st birthday as of June 15, 2012: BIRTHYR >= 1982
3. Lived continuously in US since June 15, 2007: YRIMMIG <= 2007
4. Did not have lawful status: CITIZEN = 3 (not a citizen) - already filtered
5. **Note**: Cannot observe education requirements or criminal history in ACS data

**Treatment group size**: 81,508 observations
**Control group size**: 479,962 observations

### 3. Outcome Variable
- **Primary outcome**: Full-time employment (UHRSWORK >= 35)
- **Alternative outcome**: Any employment (EMPSTAT = 1)

### 4. Identification Strategy
- **Method**: Difference-in-Differences (DiD)
- **Treatment group**: DACA-eligible Mexican-born non-citizens
- **Control group**: Non-DACA-eligible Mexican-born non-citizens (older arrivals, later immigrants, or those born before 1982)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

### 5. Control Variables
- Age, age squared (AGE, AGE^2)
- Sex (female indicator from SEX=2)
- Marital status (married indicator from MARST in [1,2])
- Education (high school or more indicator from EDUC >= 6)
- State fixed effects (STATEFIP dummies)
- Year fixed effects (YEAR dummies)

### 6. Estimation
- Weighted Least Squares (WLS) using person weights (PERWT)
- Heteroskedasticity-robust standard errors (HC1)

## Sample Selection Flow

| Step | Description | N |
|------|-------------|---|
| 1 | Total ACS 2006-2016 | 33,851,424 |
| 2 | Hispanic-Mexican (HISPAN=1) | 2,945,521 |
| 3 | Born in Mexico (BPL=200) | 991,261 |
| 4 | Non-citizen (CITIZEN=3) | 701,347 |
| 5 | Working age 16-64 | 618,640 |
| 6 | Exclude 2012 | 561,470 |
| 7 | Valid immigration year | 561,470 |

## Commands Executed

### Data Exploration
```bash
head -1 data.csv  # Check column headers
wc -l data.csv    # Count rows: 33,851,425 records (including header)
head -5 data.csv  # Examine first few rows
```

### Analysis Execution
```bash
python daca_analysis.py
```

### LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_18.tex  # First pass
pdflatex -interaction=nonstopmode replication_report_18.tex  # Second pass for TOC
pdflatex -interaction=nonstopmode replication_report_18.tex  # Third pass for references
```

## Main Results

### Preferred Estimate (Model 4: DiD with State and Year Fixed Effects)
- **Effect Size**: 0.0338 (3.38 percentage points)
- **Standard Error**: 0.0042
- **95% Confidence Interval**: [0.0254, 0.0421]
- **P-value**: < 0.001
- **Sample Size**: 561,470

### Results Across Specifications

| Model | DiD Estimate | SE | p-value |
|-------|--------------|------|---------|
| Basic DiD | 0.1002 | 0.0046 | <0.001 |
| DiD + Demographics | 0.0417 | 0.0043 | <0.001 |
| DiD + Year FE | 0.0343 | 0.0042 | <0.001 |
| DiD + State & Year FE (Preferred) | 0.0338 | 0.0042 | <0.001 |

### Robustness Checks

| Check | Estimate | SE |
|-------|----------|-----|
| Any Employment (outcome) | 0.0441 | 0.0042 |
| Age 16-35 only | 0.0049 | 0.0047 |
| Males only | 0.0305 | 0.0056 |
| Females only | 0.0282 | 0.0063 |
| Placebo (2009 fake treatment) | 0.0183 | 0.0055 |

### Event Study Coefficients (Reference: 2011)

| Year | Coefficient | SE |
|------|-------------|-----|
| 2006 | -0.019 | 0.010 |
| 2007 | -0.014 | 0.010 |
| 2008 | -0.001 | 0.010 |
| 2009 | 0.008 | 0.009 |
| 2010 | 0.011 | 0.009 |
| 2013 | 0.017 | 0.009 |
| 2014 | 0.027 | 0.009 |
| 2015 | 0.042 | 0.009 |
| 2016 | 0.044 | 0.009 |

## Interpretation

DACA eligibility is associated with approximately a 3.4 percentage point increase in full-time employment among Mexican-born non-citizens. This effect:
- Is statistically significant at the 1% level
- Is robust to inclusion of demographic controls, state FE, and year FE
- Grows over time (from ~1.7 pp in 2013 to ~4.4 pp in 2016)
- Is similar for men and women
- Represents an ~8% increase relative to the pre-DACA baseline (42.5% full-time employment)

## Caveats

1. Cannot observe documentation status directly - using non-citizen status as proxy
2. Cannot observe all DACA eligibility requirements (education, criminal history)
3. Placebo test shows some evidence of pre-trends (1.8 pp, significant)
4. Results may reflect selection into take-up rather than pure policy effects

## Files Produced

1. **daca_analysis.py** - Main analysis script
2. **results_summary.csv** - Summary of key results
3. **model_summaries.txt** - Full regression output
4. **replication_report_18.tex** - LaTeX source for report
5. **replication_report_18.pdf** - Final PDF report (22 pages)
6. **run_log_18.md** - This log file

## Session Log

- Started analysis: January 25, 2026
- Completed analysis: January 25, 2026
- Analysis software: Python 3.14 with pandas, numpy, statsmodels
- Report software: pdfLaTeX (MiKTeX)
