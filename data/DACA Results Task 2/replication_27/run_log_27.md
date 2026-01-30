# Run Log - DACA Replication Study (Replication 27)

## Overview
This document logs all key commands and decisions made during the replication of the DACA impact study on full-time employment.

---

## 1. Initial Setup and Data Exploration

### Date: 2026-01-26

### Data Source
- ACS data from IPUMS (2006-2016, 1-year samples)
- File: `data/data.csv` (33,851,424 observations)
- Data dictionary: `data/acs_data_dict.txt`

### Initial Data Examination
```
Total observations: 33,851,424
Years available: 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016
```

---

## 2. Sample Construction Decisions

### Decision 2.1: Hispanic-Mexican Ethnicity
- **Variable**: HISPAN
- **Code value**: 1 (Mexican)
- **Rationale**: Per instructions, limit to ethnically Hispanic-Mexican individuals
- **Result**: 2,945,521 observations

### Decision 2.2: Mexican Birthplace
- **Variable**: BPL
- **Code value**: 200 (Mexico)
- **Rationale**: Per instructions, limit to Mexican-born individuals
- **Result**: 991,261 observations

### Decision 2.3: Non-Citizen Status (Proxy for Undocumented)
- **Variable**: CITIZEN
- **Code value**: 3 (Not a citizen)
- **Rationale**: Instructions state to assume non-citizens without papers are undocumented. CITIZEN=3 indicates not a citizen, which serves as proxy for undocumented status since we cannot distinguish documented vs undocumented non-citizens in the data.
- **Result**: 701,347 observations

### Decision 2.4: Arrived Before Age 16
- **Calculation**: YRIMMIG - BIRTHYR < 16
- **Variables used**: YRIMMIG (year of immigration), BIRTHYR (birth year)
- **Rationale**: DACA eligibility requirement
- **Note**: Also required YRIMMIG > 0 to exclude missing values
- **Result**: 205,327 observations

### Decision 2.5: Arrived by June 15, 2007
- **Variable**: YRIMMIG
- **Condition**: YRIMMIG <= 2007
- **Rationale**: DACA required continuous US residence since June 15, 2007. Using year-level data, anyone with YRIMMIG <= 2007 would have been in the US by then.
- **Result**: 195,023 observations

### Decision 2.6: Treatment and Control Group Definitions
- **Treatment group**: Individuals aged 26-30 on June 15, 2012
  - Birth years: 1982-1986
  - Rationale: DACA required being under 31 on June 15, 2012
- **Control group**: Individuals aged 31-35 on June 15, 2012
  - Birth years: 1977-1981
  - Rationale: Similar to treatment but ineligible due to age cutoff
- **Result after group restriction**: 49,019 observations

### Decision 2.7: Exclude 2012 Survey Year
- **Rationale**: DACA was implemented mid-year (June 15, 2012). ACS does not identify survey month, so 2012 observations may be from before or after implementation.
- **Result**: 44,725 observations

---

## 3. Variable Construction

### Outcome Variable: Full-Time Employment
- **Variable used**: UHRSWORK (usual hours worked per week)
- **Definition**: fulltime = 1 if UHRSWORK >= 35, else 0
- **Rationale**: Instructions define full-time as "usually working 35 hours per week or more"

### Treatment Indicator
- **Variable**: treat
- **Definition**: 1 if birth year in 1982-1986 (ages 26-30 in 2012), else 0

### Post-Period Indicator
- **Variable**: post
- **Definition**: 1 if YEAR >= 2013, else 0
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016

### Covariates Created
- **female**: SEX == 2
- **married**: MARST <= 2 (married spouse present or absent)
- **educ_hs**: EDUC >= 6 (high school or higher)

---

## 4. Analysis Commands and Results

### 4.1 Simple Difference-in-Differences Calculation
```
Treatment pre-period mean:  0.6111
Treatment post-period mean: 0.6339
Treatment change:           0.0228

Control pre-period mean:    0.6431
Control post-period mean:   0.6108
Control change:            -0.0323

DiD estimate:               0.0551
```

### 4.2 Unweighted OLS Regression
```python
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
```
- DiD coefficient: 0.0551
- Robust SE: 0.0098
- 95% CI: [0.0359, 0.0744]
- p-value: < 0.0001

### 4.3 Weighted Regression (Preferred Specification)
```python
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
```
- DiD coefficient: 0.0620
- Robust SE: 0.0116
- 95% CI: [0.0394, 0.0847]
- p-value: < 0.0001

### 4.4 With Covariates and Fixed Effects
```python
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + C(STATEFIP) + C(YEAR)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
```
- DiD coefficient: 0.0484
- Robust SE: 0.0105
- 95% CI: [0.0278, 0.0691]
- p-value: < 0.0001

### 4.5 Event Study
Reference year: 2011
```
Year    Coefficient    SE        p-value
2006    -0.0053       0.0243    0.8270
2007    -0.0133       0.0241    0.5799
2008     0.0186       0.0247    0.4520
2009     0.0169       0.0252    0.5028
2010     0.0189       0.0250    0.4504
2013     0.0595       0.0263    0.0234
2014     0.0696       0.0266    0.0089
2015     0.0427       0.0266    0.1083
2016     0.0953       0.0267    0.0004
```

### 4.6 Subgroup Analysis by Sex
- **Males**: DiD = 0.0621 (SE: 0.0124), N = 25,058
- **Females**: DiD = 0.0313 (SE: 0.0182), N = 19,667

### 4.7 Robustness: Narrower Age Bands (27-29 vs 32-34)
- DiD coefficient: 0.0682
- Robust SE: 0.0149
- N = 26,792

### 4.8 Placebo Test (2008-2009 as fake post-period)
- Placebo DiD: 0.0120
- SE: 0.0135
- p-value: 0.3749 (not significant, as expected)

---

## 5. Key Decisions Summary

| Decision | Choice Made | Rationale |
|----------|-------------|-----------|
| Sample population | Hispanic-Mexican, Mexican-born | Per instructions |
| Undocumented proxy | Non-citizens (CITIZEN=3) | Cannot distinguish documented vs undocumented |
| Arrival before age 16 | YRIMMIG - BIRTHYR < 16 | DACA eligibility criterion |
| Continuous residence | YRIMMIG <= 2007 | DACA requirement for residence since 6/15/2007 |
| Treatment ages | 26-30 in 2012 (birth 1982-1986) | Per instructions |
| Control ages | 31-35 in 2012 (birth 1977-1981) | Per instructions |
| Exclude 2012 | Yes | Mid-year DACA implementation |
| Full-time definition | UHRSWORK >= 35 | Per instructions (35+ hours/week) |
| Weighting | PERWT | Survey weights for population representativeness |
| Standard errors | Heteroskedasticity-robust (HC1) | Account for heteroskedasticity in LPM |

---

## 6. Final Sample Characteristics

| Statistic | Treatment | Control |
|-----------|-----------|---------|
| N (pre-period) | 17,410 | 11,916 |
| N (post-period) | 9,181 | 6,218 |
| Mean age | 26.3 | 31.4 |
| Female (%) | 44.0 | 43.9 |
| Married (%) | 41.9 | 54.7 |
| High school+ (%) | 62.2 | 54.3 |
| Avg hours worked | 29.8 | 30.2 |

---

## 7. Preferred Estimate

**Effect of DACA eligibility on full-time employment:**
- Estimate: 6.20 percentage points
- Standard Error: 1.16 percentage points
- 95% CI: [3.94, 8.47] percentage points
- p-value: < 0.0001

---

## 8. Files Produced

1. `analysis.py` - Main analysis script
2. `results.json` - Key results in JSON format
3. `run_log_27.md` - This log file
4. `replication_report_27.tex` - LaTeX report
5. `replication_report_27.pdf` - Final PDF report

---

## 9. Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy
- LaTeX (pdflatex)

---

End of log.
