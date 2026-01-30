# Run Log for DACA Replication Study (ID: 56)

## Date: January 26, 2026

### Overview
This log documents all commands, key decisions, and analytical choices made during the independent replication of the DACA employment effect study.

---

## 1. Initial Setup and Data Exploration

### Files Available
- `data/data.csv` - Main ACS data file (~6.3 GB, 33,851,424 observations)
- `data/acs_data_dict.txt` - IPUMS data dictionary
- `data/state_demo_policy.csv` - Optional state-level data (not used)
- `replication_instructions.docx` - Study instructions

### Key Variables Identified from Data Dictionary
| Variable | Description | Key Values |
|----------|-------------|------------|
| YEAR | Survey year | 2006-2016 |
| BIRTHYR | Birth year | Numeric |
| BIRTHQTR | Birth quarter | 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec |
| HISPAN | Hispanic origin | 1=Mexican |
| BPL | Birthplace | 200=Mexico |
| CITIZEN | Citizenship status | 3=Not a citizen |
| YRIMMIG | Year of immigration | Numeric |
| UHRSWORK | Usual hours worked per week | 0-99 |
| EMPSTAT | Employment status | 1=Employed |
| PERWT | Person weight | Survey weight |
| SEX | Sex | 1=Male, 2=Female |
| MARST | Marital status | 1=Married spouse present |
| EDUC | Education | 6+=High school or more |
| STATEFIP | State FIPS code | State identifier |

---

## 2. Research Design Decisions

### Treatment and Control Groups
Based on instructions:
- **Treatment Group**: Ages 26-30 as of June 15, 2012
- **Control Group**: Ages 31-35 as of June 15, 2012

### Age Calculation Method
Age as of June 15, 2012 calculated as:
```
age_june_2012 = 2012 - BIRTHYR
if BIRTHQTR in [3, 4]:  # Born after June 15
    age_june_2012 -= 1
```

### DACA Eligibility Criteria Applied
1. Hispanic-Mexican ethnicity: `HISPAN == 1`
2. Born in Mexico: `BPL == 200`
3. Not a citizen: `CITIZEN == 3`
4. Arrived before age 16: `(YRIMMIG - BIRTHYR) < 16`
5. Arrived by 2007 (continuous residence): `YRIMMIG <= 2007`
6. Present in US by 2012: `YRIMMIG <= 2012`
7. Valid immigration year: `YRIMMIG > 0`

### Outcome Variable
- Full-time employment: `UHRSWORK >= 35` (binary indicator)

### Time Periods
- Pre-treatment: 2006-2011
- Post-treatment: 2013-2016
- Excluded: 2012 (DACA implemented June 15, 2012, mid-year)

---

## 3. Data Processing

### Sample Selection Flow
1. Start: 33,851,424 observations (ACS 2006-2016)
2. Filter HISPAN == 1 (Hispanic-Mexican): Reduced
3. Filter BPL == 200 (Born in Mexico): Reduced
4. Filter CITIZEN == 3 (Non-citizen): 701,347 observations
5. Filter to ages 26-35 as of June 15, 2012: 181,229 observations
6. Apply DACA eligibility criteria: 47,418 observations
7. Exclude 2012: **43,238 observations (final sample)**

### Final Sample Breakdown
- Treatment group (ages 26-30): 25,470 observations
- Control group (ages 31-35): 17,768 observations
- Pre-treatment period (2006-2011): 28,377 observations
- Post-treatment period (2013-2016): 14,861 observations

---

## 4. Analysis Commands

### Python Script: analysis.py
Main analysis performed using Python with the following packages:
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)

### Data Loading
Due to large file size (6.3 GB), data loaded in chunks of 500,000 rows with filtering applied during loading to reduce memory usage.

### Regression Models
All models estimated using weighted least squares (WLS) with:
- Weights: PERWT (ACS person weights)
- Standard errors: Robust (HC1)

#### Model Specifications:
1. **Model 1**: Basic DiD (treated + post + treated*post)
2. **Model 2**: + Year fixed effects
3. **Model 3**: + Individual covariates (female, married, educ_hs_plus)
4. **Model 4 (Preferred)**: + State fixed effects

---

## 5. Results Summary

### Descriptive Statistics (Pre-period, weighted)
| Characteristic | Control (31-35) | Treatment (26-30) |
|----------------|-----------------|-------------------|
| Female (%) | 41.4 | 43.4 |
| Married (%) | 46.9 | 32.9 |
| HS or more (%) | 52.9 | 61.3 |
| Mean Age | 29.8 | 24.8 |
| Full-time Rate (%) | 67.3 | 63.1 |

### Full-Time Employment Rates by Group and Period
| Group | Pre (2006-2011) | Post (2013-2016) | Change |
|-------|-----------------|------------------|--------|
| Control (31-35) | 67.31% | 64.33% | -2.98 pp |
| Treatment (26-30) | 63.05% | 65.97% | +2.92 pp |
| **DiD Estimate** | | | **+5.90 pp** |

### Main Regression Results
| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| Model 1 (Basic) | 0.0590 | 0.0117 | [0.036, 0.082] | <0.001 |
| Model 2 (Year FE) | 0.0574 | 0.0117 | [0.034, 0.080] | <0.001 |
| Model 3 (+Covariates) | 0.0459 | 0.0107 | [0.025, 0.067] | <0.001 |
| **Model 4 (Preferred)** | **0.0452** | **0.0107** | **[0.024, 0.066]** | **<0.001** |

### Event Study Results (Reference: 2011)
| Year | Coefficient | SE | Significant |
|------|-------------|-----|-------------|
| 2006 | -0.008 | 0.025 | No |
| 2007 | -0.044 | 0.024 | No |
| 2008 | -0.002 | 0.025 | No |
| 2009 | -0.014 | 0.026 | No |
| 2010 | -0.020 | 0.025 | No |
| 2011 | 0.000 | --- | Reference |
| 2013 | 0.038 | 0.027 | No |
| 2014 | 0.043 | 0.027 | No |
| 2015 | 0.023 | 0.027 | No |
| 2016 | 0.068 | 0.027 | Yes |

### Heterogeneity Analysis
| Subgroup | Coefficient | SE | 95% CI |
|----------|-------------|-----|--------|
| Male | 0.0446 | 0.0125 | [0.020, 0.069] |
| Female | 0.0454 | 0.0185 | [0.009, 0.082] |
| Less than HS | 0.0336 | 0.0180 | [-0.002, 0.069] |
| HS or more | 0.0768 | 0.0155 | [0.047, 0.107] |

---

## 6. Key Decisions and Justifications

### Decision 1: Age Calculation
**Choice**: Used birth quarter to determine if birthday occurred before/after June 15
**Justification**: Q1 and Q2 births assumed to have had birthday by June 15; Q3 and Q4 assumed not yet. This provides more accurate age assignment than using only birth year.

### Decision 2: Excluding 2012
**Choice**: Exclude all 2012 observations
**Justification**: DACA was implemented on June 15, 2012. Since ACS doesn't report collection month, we cannot distinguish pre- and post-implementation observations within 2012.

### Decision 3: DACA Eligibility Criteria
**Choice**: Applied arrived before 16 AND arrived by 2007 criteria to both groups
**Justification**: The control group represents individuals who would have been eligible except for age. Applying the same non-age criteria to both groups ensures comparability.

### Decision 4: Preferred Model
**Choice**: Model 4 with Year FE, State FE, and covariates
**Justification**: Year FE control for time trends; State FE control for geographic heterogeneity; Covariates (female, married, education) improve comparability between groups.

### Decision 5: Full-time Definition
**Choice**: UHRSWORK >= 35 hours per week
**Justification**: Standard BLS definition of full-time work.

### Decision 6: Robust Standard Errors
**Choice**: HC1 (heteroskedasticity-consistent) standard errors
**Justification**: Accounts for potential heteroskedasticity in the error terms without requiring specific distributional assumptions.

---

## 7. Output Files Generated

1. `analysis.py` - Main analysis script
2. `results_summary.csv` - Summary of regression results
3. `yearly_means.csv` - Full-time rates by year and treatment status
4. `event_study_coefs.csv` - Event study coefficients
5. `replication_report_56.tex` - LaTeX source for report
6. `replication_report_56.pdf` - Final PDF report (21 pages)
7. `run_log_56.md` - This run log

---

## 8. Preferred Estimate

**Effect Size**: 0.0452 (4.52 percentage points)
**Standard Error**: 0.0107
**95% Confidence Interval**: [0.0242, 0.0661]
**Sample Size**: 43,238
**p-value**: <0.001

**Interpretation**: DACA eligibility is associated with a 4.52 percentage point increase in the probability of full-time employment among Hispanic-Mexican individuals born in Mexico. This effect is statistically significant at the 1% level.

---

## 9. Software and Versions

- Python 3.14 (Python Software Foundation)
- pandas (data manipulation)
- numpy (numerical computing)
- statsmodels (statistical modeling)
- pdfLaTeX (MiKTeX 25.12)

---

## 10. Completion

Analysis completed: January 26, 2026
All required deliverables produced:
- [x] replication_report_56.tex
- [x] replication_report_56.pdf
- [x] run_log_56.md
