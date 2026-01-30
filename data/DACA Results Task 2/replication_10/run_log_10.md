# Run Log - DACA Replication Study (Replication 10)

## Overview
This log documents all commands, key decisions, and analytical choices made during this independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals.

## Date: 2026-01-26

---

## 1. Data Exploration

### 1.1 Data Files
- **Main data file**: `data/data.csv` (33,851,425 rows including header)
- **Data dictionary**: `data/acs_data_dict.txt`
- **Optional state-level data**: `data/state_demo_policy.csv`

### 1.2 Key Variables Identified
Based on the data dictionary and research question:

**Outcome Variable:**
- `UHRSWORK`: Usual hours worked per week (full-time = 35+ hours)

**Sample Selection Variables:**
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `YEAR`: Survey year

**Control Variables (potential):**
- `SEX`: Gender
- `EDUC`/`EDUCD`: Educational attainment
- `MARST`: Marital status
- `STATEFIP`: State
- `AGE`: Age

---

## 2. Research Design Decisions

### 2.1 Treatment and Control Groups
Based on the instructions:
- **Treatment group**: Ages 26-30 as of June 15, 2012 (born 1982-1986, with precise cutoff accounting for DACA date)
- **Control group**: Ages 31-35 as of June 15, 2012 (born 1977-1981)

### 2.2 Time Periods
- **Pre-treatment**: 2006-2011 (excluding 2012 due to implementation timing ambiguity)
- **Post-treatment**: 2013-2016

### 2.3 DACA Eligibility Criteria
To be DACA-eligible, individuals must:
1. Be Hispanic-Mexican (`HISPAN == 1`)
2. Be born in Mexico (`BPL == 200`)
3. Not be a U.S. citizen (`CITIZEN == 3`)
4. Have arrived in the U.S. before age 16
5. Have been in the U.S. since June 15, 2007 (immigration year <= 2007)
6. Age restrictions as specified above

### 2.4 Outcome Definition
- Full-time employment = `UHRSWORK >= 35` (usual hours worked 35+ per week)

### 2.5 Key Decisions and Justifications

1. **Exclusion of 2012**: The year 2012 is excluded because DACA was implemented on June 15, 2012, and the ACS does not specify the month of data collection. Including 2012 would create ambiguity in treatment status.

2. **Age calculation using BIRTHQTR**: To precisely determine age as of June 15, 2012, we use birth quarter information. Those born in Q1-Q2 (Jan-Jun) would have had their birthday by June 15, while those born in Q3-Q4 (Jul-Dec) would not.

3. **Non-citizen proxy for undocumented**: Since ACS cannot distinguish documented from undocumented non-citizens, we assume all non-citizens who have not received immigration papers are undocumented. This may include some documented non-citizens (false positives).

4. **Arrived before age 16**: Computed as YRIMMIG - BIRTHYR < 16. This is an approximation since we don't have month-level immigration data.

5. **Continuous residence since 2007**: Implemented as YRIMMIG <= 2007.

---

## 3. Sample Construction

### 3.1 Sample Filtering Steps

```
Initial observations:                      33,851,424
After Hispanic-Mexican filter (HISPAN=1):   2,945,521
After Mexico birthplace (BPL=200):            991,261
After non-citizen filter (CITIZEN=3):         701,347
After excluding 2012:                         636,722
After age group filter (26-35 in June 2012):  164,874
After arrived before age 16:                   43,238
After continuous residence since 2007:         43,238

FINAL ANALYSIS SAMPLE:                         43,238
```

### 3.2 Sample by Group and Period

| Group              | Pre (2006-2011) | Post (2013-2016) | Total  |
|--------------------|-----------------|------------------|--------|
| Control (31-35)    | 11,683          | 6,085            | 17,768 |
| Treatment (26-30)  | 16,694          | 8,776            | 25,470 |
| Total              | 28,377          | 14,861           | 43,238 |

### 3.3 Weighted Sample Sizes

| Group              | Pre (2006-2011) | Post (2013-2016) |
|--------------------|-----------------|------------------|
| Control (31-35)    | 1,631,151       | 845,134          |
| Treatment (26-30)  | 2,280,009       | 1,244,124        |

---

## 4. Main Analysis Commands

### 4.1 Python Script Execution
```bash
python analysis_script.py
```

### 4.2 Regression Models Estimated

1. **Model 1**: Basic DiD (unweighted OLS)
   - Formula: `fulltime ~ treated + post + treated_post`

2. **Model 2**: Basic DiD (weighted)
   - Formula: `fulltime ~ treated + post + treated_post`
   - Weights: PERWT (person weights)

3. **Model 3**: With demographic controls (weighted)
   - Formula: `fulltime ~ treated + post + treated_post + female + married + C(educ_cat)`

4. **Model 4**: With year fixed effects (weighted)
   - Formula: `fulltime ~ treated + C(YEAR) + treated_post + female + married + C(educ_cat)`

5. **Model 5**: With year and state fixed effects (weighted)
   - Formula: `fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + C(educ_cat)`

6. **Preferred Model**: Model 5 with state-clustered standard errors

---

## 5. Main Results

### 5.1 Full-Time Employment Rates by Group and Period

| Group              | Pre (2006-2011) | Post (2013-2016) | Change   |
|--------------------|-----------------|------------------|----------|
| Control (31-35)    | 67.31%          | 64.33%           | -2.99 pp |
| Treatment (26-30)  | 63.05%          | 65.97%           | +2.92 pp |

**Simple DiD Estimate: 5.90 percentage points**

### 5.2 Regression Results Summary

| Model | Specification              | DiD Coef. | SE      | 95% CI              |
|-------|----------------------------|-----------|---------|---------------------|
| 1     | Basic (unweighted)         | 0.0516    | 0.0100  | [0.0321, 0.0711]    |
| 2     | Basic (weighted)           | 0.0590    | 0.0098  | [0.0398, 0.0782]    |
| 3     | + Demographics             | 0.0473    | 0.0090  | [0.0296, 0.0650]    |
| 4     | + Year FE                  | 0.0456    | 0.0090  | [0.0280, 0.0632]    |
| 5     | + State FE                 | 0.0448    | 0.0090  | [0.0272, 0.0624]    |
| 5R    | Robust SE                  | 0.0448    | 0.0106  | [0.0239, 0.0657]    |
| 5C    | Clustered SE (state)       | 0.0448    | 0.0099  | [0.0254, 0.0642]    |

### 5.3 Preferred Estimate

**Effect Size: 0.0448 (4.48 percentage points)**
**Standard Error: 0.0099 (state-clustered)**
**95% CI: [0.0254, 0.0642]**
**Sample Size: 43,238**
**p-value: < 0.0001**

---

## 6. Robustness Checks

### 6.1 Event Study Coefficients (relative to 2011)

| Year | Coefficient | SE     |
|------|-------------|--------|
| 2006 | 0.0062      | 0.0227 |
| 2007 | -0.0305     | 0.0222 |
| 2008 | 0.0087      | 0.0227 |
| 2009 | -0.0065     | 0.0234 |
| 2010 | -0.0151     | 0.0232 |
| 2013 | 0.0335      | 0.0241 |
| 2014 | 0.0348      | 0.0245 |
| 2015 | 0.0213      | 0.0248 |
| 2016 | 0.0673      | 0.0246 |

**Interpretation**: Pre-treatment coefficients are small and not statistically significant, supporting the parallel trends assumption. Post-treatment coefficients are generally positive and increasing.

### 6.2 Placebo Test (Pre-Period Only)

Using 2006-2008 as "pre" and 2009-2011 as "post":
- Placebo DiD Coefficient: -0.0011
- SE: 0.0125
- p-value: 0.9323

**Interpretation**: The placebo test shows no significant differential trend in the pre-period, supporting the validity of the research design.

### 6.3 Heterogeneity Analysis

**By Gender:**
| Gender | DiD Coef. | SE     | N      |
|--------|-----------|--------|--------|
| Male   | 0.0352    | 0.0124 | 24,243 |
| Female | 0.0488    | 0.0181 | 18,995 |

**By Education:**
| Education     | DiD Coef. | SE     | N      |
|---------------|-----------|--------|--------|
| Less than HS  | 0.0204    | 0.0159 | 18,057 |
| HS or more    | 0.0685    | 0.0145 | 25,181 |

---

## 7. Covariate Balance (Pre-Period)

| Variable    | Treatment | Control |
|-------------|-----------|---------|
| Female      | 0.434     | 0.414   |
| Married     | 0.377     | 0.518   |
| Age         | 24.77     | 29.79   |

**Education Distribution:**

| Education     | Treatment | Control |
|---------------|-----------|---------|
| Less than HS  | 38.7%     | 47.1%   |
| HS            | 44.3%     | 40.0%   |
| Some college  | 16.8%     | 12.3%   |
| College+      | 0.2%      | 0.6%    |

---

## 8. Output Files Generated

1. `analysis_script.py` - Main Python analysis script
2. `analysis_results.csv` - Key regression results
3. `fulltime_rates.csv` - Full-time employment rates by group/period
4. `event_study_results.csv` - Event study coefficients
5. `descriptive_stats.csv` - Descriptive statistics
6. `replication_report_10.tex` - LaTeX report
7. `replication_report_10.pdf` - Final PDF report

---

## 9. Software Environment

- Python 3.x
- pandas
- numpy
- statsmodels
- scipy

---

## 10. Notes and Limitations

1. **Cannot verify all DACA eligibility criteria**: ACS data does not allow us to verify physical presence on June 15, 2012, or lack of lawful immigration status.

2. **Non-citizen includes documented immigrants**: Our sample likely includes some documented non-citizens who would not be eligible for DACA.

3. **Age approximation**: Without month of birth, we use quarter to approximate age as of June 15, 2012.

4. **Immigration year approximation**: YRIMMIG is coded in years, so we cannot verify arrival before 16th birthday precisely for all respondents.

5. **Cross-sectional data**: ACS is a repeated cross-section, not panel data, so we observe different individuals before and after DACA.

