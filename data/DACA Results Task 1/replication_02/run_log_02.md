# Replication Run Log - Replication 02

## Project Overview
**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**DACA Implementation Date:** June 15, 2012
**Study Period:** 2013-2016 (post-treatment outcomes)

---

## Data Files
- **data.csv**: ACS data from 2006-2016 (33,851,425 total observations)
- **acs_data_dict.txt**: IPUMS variable codebook
- **state_demo_policy.csv**: Optional state-level supplementary data (not used)

---

## Key Variables from ACS Data Dictionary

### Identification Variables
- `YEAR`: Survey year (2006-2016)
- `PERWT`: Person weight for population estimates

### DACA Eligibility Criteria Variables
- `HISPAN` / `HISPAND`: Hispanic origin (1=Mexican for general; 100-107 for Mexican detailed)
- `BPL` / `BPLD`: Birthplace (200=Mexico for general; 20000=Mexico for detailed)
- `CITIZEN`: Citizenship status (3=Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Year of birth
- `BIRTHQTR`: Quarter of birth (1-4)
- `AGE`: Age at time of survey

### Outcome Variable
- `UHRSWORK`: Usual hours worked per week (>=35 indicates full-time employment)
- `EMPSTAT`: Employment status (1=Employed)

### Control Variables
- `SEX`: Gender (1=Male, 2=Female)
- `EDUC` / `EDUCD`: Educational attainment
- `MARST`: Marital status
- `STATEFIP`: State FIPS code
- `AGE`: Age

---

## DACA Eligibility Criteria (from instructions)

To be DACA eligible, a person must:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

### Operationalization in ACS Data:
- **Hispanic-Mexican, born in Mexico:** `HISPAN == 1` AND `BPL == 200`
- **Non-citizen:** `CITIZEN == 3` (assume undocumented)
- **Age at arrival < 16:** `YRIMMIG - BIRTHYR < 16`
- **Born after June 15, 1981:** Use `BIRTHYR` and `BIRTHQTR` (Q1-Q2 1981 ineligible; Q3-Q4 1981+ eligible)
- **In US since at least 2007:** `YRIMMIG <= 2007`
- **Working-age sample:** Ages 18-45

---

## Commands and Analysis Steps

### Step 1: Data Exploration
```
Command: Examined data.csv structure via head and wc -l
Result: 33,851,425 total rows, years 2006-2016
```

### Step 2: Data Loading and Filtering
```python
# Load with chunked reading to manage memory (33M+ rows)
# Filter to Hispanic-Mexican (HISPAN==1) born in Mexico (BPL==200)
# Result: 991,261 observations
```

### Step 3: Sample Restrictions Applied
```
1. Non-citizen (CITIZEN==3): 701,347 observations
2. Exclude 2012: 636,722 observations
3. Age 18-45: 413,906 observations
4. Valid immigration year: 413,906 observations (final analytic sample)
```

### Step 4: DACA Eligibility Classification
```
DACA Eligible: 71,347 (17.2%)
Not Eligible: 342,559 (82.8%)
```

### Step 5: Outcome Variable Construction
```
Employment rate (EMPSTAT==1): 66.2%
Full-time employment (UHRSWORK>=35 & employed): 54.3%
```

### Step 6: DiD Variable Construction
```
Pre-period (2006-2011): 265,058 observations
Post-period (2013-2016): 148,848 observations
```

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Exclude 2012 | ACS lacks month of interview; cannot determine if observation is pre/post DACA announcement |
| Use CITIZEN==3 as undocumented proxy | Instructions state: "Assume that anyone who is not a citizen and who has not received immigration papers is undocumented" |
| Full-time = UHRSWORK >= 35 | Per research question definition |
| Age restriction: 18-45 | Focus on prime working-age adults; upper bound ensures control group has sufficient variation |
| DACA eligibility criteria | (1) arrived before age 16, (2) born after June 15, 1981, (3) in US since 2007 |
| Weighted least squares | Use PERWT for population-representative estimates |
| Cluster standard errors by state | Account for within-state correlation in outcomes |

---

## Main Results Summary

### Preferred Specification (Model 2: DiD with Demographic Controls)
- **DiD Coefficient:** 0.0258
- **Standard Error:** 0.0037 (clustered by state)
- **95% CI:** [0.0186, 0.0330]
- **P-value:** < 0.0001
- **Sample Size:** 413,906

### Interpretation
DACA eligibility is associated with a **2.58 percentage point increase** in the probability of full-time employment. This effect is statistically significant at the 1% level.

### Manual DiD Calculation (Validation)
```
Treatment group (DACA eligible):
    Pre:  0.4651
    Post: 0.5251
    Change: +0.0599

Control group (Not eligible):
    Pre:  0.5916
    Post: 0.5853
    Change: -0.0064

DiD Estimate: 0.0599 - (-0.0064) = 0.0663 (matches basic model)
```

### Model Comparison
| Model | DiD Coefficient | Std Error | P-value |
|-------|-----------------|-----------|---------|
| Basic DiD | 0.0663 | 0.0031 | <0.001 |
| With Controls | 0.0258 | 0.0037 | <0.001 |
| Year FE | 0.0154 | 0.0034 | <0.001 |
| State + Year FE | 0.0153 | 0.0047 | 0.001 |

### Robustness Checks
| Specification | DiD Coefficient | Std Error |
|--------------|-----------------|-----------|
| Main specification | 0.0258 | 0.0037 |
| Alternative control (arrival age 16-20) | 0.0254 | 0.0051 |
| Any employment outcome | 0.0407 | 0.0049 |
| Narrower age band (18-35) | 0.0224 | 0.0050 |
| Males only | 0.0096 | 0.0054 |
| Females only | 0.0395 | 0.0073 |

### Event Study Results (Relative to 2011)
| Year | Coefficient | Std Error |
|------|-------------|-----------|
| 2006 | 0.0077 | 0.0107 |
| 2007 | 0.0094 | 0.0067 |
| 2008 | 0.0194 | 0.0125 |
| 2009 | 0.0235 | 0.0131 |
| 2010 | 0.0208 | 0.0121 |
| 2013 | 0.0152 | 0.0098 |
| 2014 | 0.0245 | 0.0129 |
| 2015 | 0.0368 | 0.0112 |
| 2016 | 0.0394 | 0.0108 |

---

## Analysis Code Files
- `analysis.py`: Main Python analysis script
- `replication_report_02.tex`: LaTeX report
- `replication_report_02.pdf`: Compiled report

## Output Files Generated
- `descriptive_stats.csv`: Summary statistics by group
- `main_results.csv`: Main regression results
- `robustness_results.csv`: Robustness check results
- `event_study_results.csv`: Event study coefficients
- `sample_sizes.csv`: Sample sizes by group
- `preferred_model_summary.txt`: Full regression output
- `table_summary.tex`: LaTeX summary statistics table
- `table_main.tex`: LaTeX main results table
- `table_robust.tex`: LaTeX robustness table

---

## Software Environment
- Python 3.x
- pandas, numpy, statsmodels, scipy
- LaTeX (pdflatex)

