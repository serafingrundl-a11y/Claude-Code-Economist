# Replication Run Log - ID 99

## Project: DACA Impact on Full-Time Employment

### Date: 2026-01-25

---

## 1. Initial Setup and Data Understanding

### 1.1 Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (working 35+ hours per week)?

### 1.2 Data Source
- American Community Survey (ACS) data from IPUMS USA
- Years: 2006-2016 (1-year ACS files)
- Supplemental state-level data available (optional)

### 1.3 Key Variables Identified from Data Dictionary
- **YEAR**: Survey year (2006-2016)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican, 100-107=Mexican detailed)
- **BPL/BPLD**: Birthplace (200/20000 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth
- **UHRSWORK**: Usual hours worked per week (35+ = full-time)
- **AGE**: Age at time of survey
- **PERWT**: Person weight for population estimates

### 1.4 DACA Eligibility Criteria (from instructions)
1. Arrived in US before 16th birthday
2. Had not turned 31 by June 15, 2012 (born on or after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012
5. No lawful status (not a citizen, no legal residency)

---

## 2. Analytical Strategy

### 2.1 Identification Strategy
Difference-in-Differences (DiD) approach:
- **Treatment group**: Hispanic-Mexican, Mexico-born, non-citizens who meet DACA age and immigration timing criteria
- **Control group**: Hispanic-Mexican, Mexico-born, non-citizens who do NOT meet DACA eligibility (too old, arrived too recently, etc.)
- **Pre-period**: 2006-2011 (before DACA)
- **Post-period**: 2013-2016 (after DACA implementation)
- **Note**: 2012 excluded due to mid-year DACA implementation

### 2.2 Treatment Definition
An individual is DACA-eligible if:
1. HISPAN == 1 (Mexican) AND BPL == 200 (Mexico)
2. CITIZEN == 3 (not a citizen)
3. Age at arrival < 16 (calculated from YRIMMIG and BIRTHYR)
4. BIRTHYR >= 1981 (and born before June 15, 2012 to be at least as old as minimum DACA applicants)
5. YRIMMIG <= 2007 (arrived by June 15, 2007 - using year as proxy)

### 2.3 Outcome Variable
- Full-time employment: UHRSWORK >= 35

---

## 3. Commands and Code Execution Log

### 3.1 Data Loading and Initial Exploration
```
Command: python analysis.py
```

**Results:**
- Total observations loaded: 33,851,424
- Years in data: 2006-2016

### 3.2 Sample Restriction Steps
| Step | Filter | Remaining Obs |
|------|--------|---------------|
| 1 | Initial sample | 33,851,424 |
| 2 | Hispanic-Mexican (HISPAN==1) | 2,945,521 |
| 3 | Mexico birthplace (BPL==200) | 991,261 |
| 4 | Non-citizen (CITIZEN==3) | 701,347 |
| 5 | Working age 16-64 | 618,640 |
| 6 | Exclude 2012 | 561,470 |

### 3.3 Treatment and Control Group Construction
- **DACA eligible:** 84,188 observations
- **Non-eligible:** 477,282 observations

---

## 4. Main Results

### 4.1 Preferred Estimate (DiD with controls and robust SEs)
| Statistic | Value |
|-----------|-------|
| Effect size | 0.0349 |
| Robust SE | 0.0035 |
| t-statistic | 10.02 |
| p-value | <0.0001 |
| 95% CI | [0.0280, 0.0417] |
| R-squared | 0.2094 |
| N | 561,470 |

**Interpretation:** DACA eligibility is associated with a 3.49 percentage point increase in the probability of full-time employment after program implementation.

### 4.2 Model Specifications Tested
| Model | DiD Estimate | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD (no controls) | 0.0862 | 0.0037 | <0.001 |
| With demographic controls | 0.0349 | 0.0034 | <0.001 |
| With year + state FEs | 0.0291 | 0.0033 | <0.001 |
| Weighted (PERWT) | 0.0339 | 0.0033 | <0.001 |
| **Preferred (robust SE)** | **0.0349** | **0.0035** | **<0.001** |

---

## 5. Robustness Checks

### 5.1 Age Restriction (18-35)
- Effect: 0.0237 (SE: 0.0042), p<0.001

### 5.2 Placebo Test (fake treatment 2009)
- Effect: 0.0141 (SE: 0.0045), p=0.002
- Note: Some pre-trends detected, suggesting cautious interpretation

### 5.3 Event Study Coefficients (relative to 2011)
| Year | Coefficient | SE | Significance |
|------|-------------|-----|--------------|
| 2006 | -0.0171 | 0.0078 | ** |
| 2007 | -0.0139 | 0.0076 | * |
| 2008 | -0.0012 | 0.0077 | |
| 2009 | 0.0023 | 0.0076 | |
| 2010 | 0.0075 | 0.0074 | |
| 2013 | 0.0082 | 0.0073 | |
| 2014 | 0.0198 | 0.0073 | *** |
| 2015 | 0.0401 | 0.0073 | *** |
| 2016 | 0.0388 | 0.0074 | *** |

---

## 6. Heterogeneity Analysis

### 6.1 By Sex
- Male: 0.0332 (SE: 0.0046), p<0.001
- Female: 0.0301 (SE: 0.0051), p<0.001

### 6.2 By Education
- Less than HS: 0.0233 (SE: 0.0050), p<0.001
- HS or more: 0.0310 (SE: 0.0048), p<0.001

---

## 7. Key Decisions and Justifications

1. **Excluded 2012:** DACA was implemented mid-year (June 15, 2012), creating ambiguity about treatment status for observations in that year.

2. **Age restriction 16-64:** Standard working-age population definition for employment analysis.

3. **Non-citizen filter:** Per instructions, assumed non-citizens without papers are undocumented. This is a limitation as some may have other visa statuses.

4. **Continuous presence proxy:** Used YRIMMIG <= 2007 as proxy for continuous presence since June 15, 2007.

5. **Birth year cutoff:** Used BIRTHYR >= 1981 to identify those under 31 on June 15, 2012.

6. **Full-time definition:** UHRSWORK >= 35 per BLS definition and research question specification.

---

## 8. Output Files Generated

- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `results_summary.csv` - Summary of regression results
- `model_output.txt` - Full regression output
- `key_stats.txt` - Key statistics for LaTeX
- `figure1_trends.png` - Parallel trends plot
- `figure2_eventstudy.png` - Event study coefficients
- `figure3_hours.png` - Hours distribution
- `figure4_composition.png` - Sample composition
- `figure5_heterogeneity.png` - Heterogeneity analysis
- `replication_report_99.tex` - LaTeX report
- `replication_report_99.pdf` - Final PDF report

---

## 9. Session Log

- Analysis started: 2026-01-25
- Data loaded successfully from data/data.csv
- All sample restrictions applied successfully
- DiD analysis completed with multiple specifications
- Robustness checks completed
- Figures generated
- LaTeX report compiled
