# DACA Replication Study - Run Log

## Study Information
- **Study ID:** 53
- **Date:** January 26, 2026
- **Research Question:** Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals

---

## 1. Data Loading and Exploration

### 1.1 Initial Data Assessment
- **Data Source:** IPUMS ACS data (2006-2016)
- **File:** `data/data.csv`
- **Initial observations:** 33,851,424
- **Variables:** 54

### 1.2 Key Variables Identified
From `data/acs_data_dict.txt`:
- `YEAR`: Survey year
- `PERWT`: Person weight for population estimates
- `HISPAN`: Hispanic origin (1 = Mexican)
- `BPL`: Birthplace (200 = Mexico)
- `CITIZEN`: Citizenship status (3 = Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter (1-4)
- `UHRSWORK`: Usual hours worked per week
- `SEX`, `AGE`, `MARST`, `EDUC`: Demographic covariates

---

## 2. Sample Construction

### 2.1 DACA Eligibility Criteria Applied
Following the instructions and DACA program requirements:

1. **Hispanic-Mexican ethnicity:** `HISPAN == 1`
   - 2,945,521 observations remaining

2. **Born in Mexico:** `BPL == 200`
   - 991,261 observations remaining

3. **Not a citizen:** `CITIZEN == 3`
   - 701,347 observations remaining

4. **Valid immigration year:** `YRIMMIG > 0`
   - 701,347 observations remaining

5. **Arrived before age 16:** `YRIMMIG - BIRTHYR < 16`
   - 205,327 observations remaining

6. **Continuous residence since June 2007:** `YRIMMIG <= 2007`
   - 195,023 observations remaining

### 2.2 Treatment and Control Group Definition

**Age at June 15, 2012 calculation:**
```python
age_june2012 = 2012 - BIRTHYR
# Adjust for those born after June (Q3, Q4)
if BIRTHQTR in [3, 4]:
    age_june2012 -= 1
```

**Group assignments:**
- Treatment: Ages 26-30 on June 15, 2012 (DACA-eligible)
- Control: Ages 31-35 on June 15, 2012 (too old for DACA)

### 2.3 Time Period Definition
- **Pre-DACA:** 2006-2011
- **Post-DACA:** 2013-2016
- **Excluded:** 2012 (ambiguous implementation year)

### 2.4 Final Analysis Sample
- **Total observations:** 43,238
- **Treatment group:** 25,470 (58.9%)
- **Control group:** 17,768 (41.1%)

---

## 3. Variable Construction

### 3.1 Outcome Variable
```python
fulltime = 1 if UHRSWORK >= 35 else 0
```
Definition: Full-time employment = usually working 35+ hours per week

### 3.2 Key Analytical Variables
- `treatment`: 1 for ages 26-30 at June 2012, 0 for ages 31-35
- `post`: 1 for years 2013-2016, 0 for years 2006-2011
- `treat_post`: Interaction term (treatment * post) - coefficient of interest

### 3.3 Covariates
- `age_centered`: Age minus sample mean
- `male`: SEX == 1
- `married`: MARST == 1
- `EDUC`: Education category (categorical)

---

## 4. Analysis Decisions

### 4.1 Identification Strategy
**Difference-in-Differences (DiD)** design exploiting the age-based eligibility cutoff:
- Individuals just below age 31 in June 2012 were eligible
- Individuals at or above age 31 were ineligible
- Compare changes in outcomes between these groups before vs. after DACA

### 4.2 Model Specifications
Five models estimated:

1. **Model 1:** Basic DiD (OLS, unweighted)
2. **Model 2:** Basic DiD (WLS, person weights)
3. **Model 3:** DiD with covariates (WLS) - includes age, sex, marital status, education
4. **Model 4:** DiD with year fixed effects (WLS)
5. **Model 5:** DiD with robust standard errors (HC1) - **PREFERRED SPECIFICATION**

### 4.3 Weighting Decision
Used person weights (PERWT) to obtain population-representative estimates, as recommended for ACS data.

### 4.4 Standard Errors
Heteroskedasticity-robust (HC1) standard errors used in preferred specification to account for potential heteroskedasticity.

---

## 5. Results Summary

### 5.1 Weighted Cell Means
| Group | Period | Full-Time Rate |
|-------|--------|---------------|
| Control (ages 31-35) | Pre (2006-2011) | 67.3% |
| Control (ages 31-35) | Post (2013-2016) | 64.3% |
| Treatment (ages 26-30) | Pre (2006-2011) | 63.1% |
| Treatment (ages 26-30) | Post (2013-2016) | 66.0% |

### 5.2 Manual DiD Calculation
- Treatment change: 66.0% - 63.1% = +2.9 pp
- Control change: 64.3% - 67.3% = -3.0 pp
- **DiD estimate:** +2.9 - (-3.0) = **+5.9 pp**

### 5.3 Regression Estimates

| Model | DiD Estimate | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| 1. OLS | 0.052 | 0.010 | [0.032, 0.071] | <0.001 |
| 2. WLS | 0.059 | 0.010 | [0.040, 0.078] | <0.001 |
| 3. WLS + Covariates | 0.045 | 0.009 | [0.027, 0.063] | <0.001 |
| 4. WLS + Year FE | 0.044 | 0.009 | [0.027, 0.062] | <0.001 |
| 5. WLS + Robust SE | **0.045** | **0.011** | **[0.024, 0.066]** | **<0.001** |

### 5.4 Preferred Estimate (Model 5)
- **Effect:** 4.49 percentage points
- **Robust SE:** 0.0107
- **95% CI:** [2.39, 6.59] pp
- **p-value:** <0.001

---

## 6. Parallel Trends Analysis

### 6.1 Yearly Full-Time Employment Rates
| Year | Control | Treatment | Difference |
|------|---------|-----------|------------|
| 2006 | 69.1% | 65.6% | -3.5 |
| 2007 | 73.3% | 66.3% | -7.0 |
| 2008 | 69.7% | 66.9% | -2.8 |
| 2009 | 65.3% | 61.2% | -4.1 |
| 2010 | 63.5% | 59.0% | -4.5 |
| 2011 | 61.7% | 59.1% | -2.6 |
| 2013 | 63.7% | 64.8% | +1.1 |
| 2014 | 61.8% | 63.5% | +1.7 |
| 2015 | 66.5% | 66.1% | -0.4 |
| 2016 | 65.7% | 69.9% | +4.2 |

### 6.2 Assessment
- Pre-trends show both groups declining during Great Recession
- Gap between groups relatively stable pre-DACA
- Post-DACA: Treatment group exceeds control in 3 of 4 years (reversal)
- Visual inspection supports parallel trends assumption

---

## 7. Key Decisions and Justifications

### 7.1 Age Range Selection
- Treatment: 26-30 (eligible, but close to cutoff)
- Control: 31-35 (just missed cutoff)
- Rationale: Groups similar except for DACA eligibility; avoid very young individuals with different labor market attachment

### 7.2 Exclusion of 2012
- DACA implemented June 15, 2012
- ACS doesn't distinguish survey month
- Including 2012 would contaminate pre/post comparison

### 7.3 Definition of Full-Time Employment
- Used UHRSWORK >= 35 hours/week
- Standard definition aligning with BLS/Census conventions

### 7.4 Non-Citizen Assumption
- Cannot distinguish documented vs. undocumented non-citizens
- Assumed all Mexican-born non-citizens without naturalization are undocumented
- This is a limitation but necessary given data constraints

---

## 8. Output Files Generated

1. **analysis.py** - Main analysis script
2. **replication_report_53.tex** - LaTeX source for report
3. **replication_report_53.pdf** - Final PDF report (20 pages)
4. **run_log_53.md** - This log file
5. **model_results.txt** - Full regression output
6. **descriptive_stats.csv** - Summary statistics by group/period
7. **yearly_means.csv** - Year-by-year employment rates
8. **results_summary.json** - Key results in JSON format
9. **figure1_parallel_trends.png** - Parallel trends visualization
10. **figure2_did_bars.png** - DiD bar chart visualization

---

## 9. Interpretation

The preferred estimate indicates that DACA eligibility **increased the probability of full-time employment by approximately 4.5 percentage points** (95% CI: 2.4-6.6 pp). This represents a **7% increase** relative to the pre-DACA treatment group baseline of 63.1%.

The effect is:
- **Statistically significant** (p < 0.001)
- **Robust** across specifications
- **Economically meaningful**

The positive effect is consistent with DACA providing:
- Legal work authorization
- Reduced deportation fear
- Access to documentation (SSN, driver's licenses)
- Facilitated formal sector employment

---

## 10. Limitations

1. Cannot directly observe DACA receipt (intent-to-treat estimate)
2. Repeated cross-section, not panel data
3. Cannot distinguish documented vs. undocumented non-citizens
4. Pre-trends not perfectly parallel
5. 5-year age gap between treatment and control groups

---

## Session Log

```
[2026-01-26] Session started
[2026-01-26] Read replication_instructions.docx
[2026-01-26] Examined data structure (33.8M rows, 54 columns)
[2026-01-26] Read acs_data_dict.txt for variable definitions
[2026-01-26] Created analysis.py script
[2026-01-26] Ran analysis - generated results
[2026-01-26] Created replication_report_53.tex (LaTeX report)
[2026-01-26] Compiled PDF (20 pages)
[2026-01-26] Created run_log_53.md
[2026-01-26] Session completed
```

---

*End of Run Log*
