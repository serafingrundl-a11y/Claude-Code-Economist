# Run Log - DACA Replication Study #86

## Overview
This log documents the independent replication of a study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability that the eligible person is employed full-time (35+ hours/week)?

## Identification Strategy
- **Treatment Group**: Ages 26-30 as of June 15, 2012 (born 1982-1986)
- **Control Group**: Ages 31-35 as of June 15, 2012 (born 1977-1981)
- **Method**: Difference-in-differences comparing pre-DACA (2006-2011) to post-DACA (2013-2016)
- **Note**: 2012 excluded as DACA implemented mid-year (June 15, 2012)

## Data
- ACS data from IPUMS: 2006-2016 (1-year files)
- Sample: Hispanic-Mexican ethnicity, born in Mexico, non-citizens without papers

---

## Session Log

### Step 1: Data Exploration
**Date**: 2026-01-26

**Files identified**:
- `data/data.csv` - Main ACS data (6.2 GB, 33,851,424 observations)
- `data/acs_data_dict.txt` - Data dictionary
- `data/state_demo_policy.csv` - State-level supplementary data (optional, not used)

**Key variables identified**:
- YEAR: Survey year (2006-2016)
- BIRTHYR: Birth year
- BIRTHQTR: Birth quarter (1-4)
- HISPAN/HISPAND: Hispanic origin (HISPAN=1 for Mexican)
- BPL/BPLD: Birthplace (BPL=200 for Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight

### Step 2: Sample Definition Decisions
**DACA Eligibility Criteria Applied**:
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3) - proxy for undocumented status
4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
5. Arrived by 2007 (YRIMMIG <= 2007, for continuous US presence since June 2007)
6. For age at June 2012: Using BIRTHYR to determine age cohorts

**Treatment/Control Definition**:
- Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
- Control: Born 1977-1981 (ages 31-35 on June 15, 2012)
- Pre-period: 2006-2011
- Post-period: 2013-2016

**Outcome Variable**:
- Full-time employment: UHRSWORK >= 35

### Step 3: Sample Construction Results
```
Total observations loaded: 33,851,424
After Hispanic-Mexican filter: 2,945,521
After Mexico birthplace filter: 991,261
After non-citizen filter: 701,347
After arrival before age 16 filter: 205,327
After arrival by 2007 filter: 195,023
After age group filter (born 1977-1986): 49,019
After excluding 2012: 44,725

Final analytic sample: 44,725
  Treatment group (ages 26-30 in 2012): 26,591
  Control group (ages 31-35 in 2012): 18,134
  Pre-period (2006-2011): 29,326
  Post-period (2013-2016): 15,399
```

### Step 4: Analysis Implementation
**Python script created**: `analysis_86.py`

**Models estimated**:
1. Simple DiD (no controls)
2. DiD with demographic controls (female, married, education, children, age)
3. DiD with year fixed effects
4. DiD with year and state fixed effects (preferred specification)

**Control variables used**:
- Female (SEX == 2)
- Married (MARST in {1, 2})
- High school or more (EDUC >= 6)
- Has children (NCHILD > 0)
- Age (YEAR - BIRTHYR)

### Step 5: Main Results

**Difference-in-Differences Estimates**:
| Model | Coefficient | Std Error | 95% CI | p-value |
|-------|-------------|-----------|--------|---------|
| Simple DiD | 0.0620 | 0.0116 | [0.039, 0.085] | <0.001 |
| With Controls | 0.0477 | 0.0106 | [0.027, 0.068] | <0.001 |
| Year FE | 0.0464 | 0.0106 | [0.026, 0.067] | <0.001 |
| Year + State FE | 0.0458 | 0.0105 | [0.025, 0.066] | <0.001 |

**Preferred Estimate (Model 4)**:
- Effect Size: 0.0458 (4.58 percentage points)
- Standard Error: 0.0105
- 95% CI: [0.0251, 0.0664]
- p-value: <0.001
- Sample Size: 44,725

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 4.6 percentage points.

### Step 6: Robustness Checks

**1. Any Employment (EMPSTAT==1)**:
- DiD coefficient: 0.0453 (SE: 0.0101)
- Conclusion: Similar effect on any employment

**2. By Gender**:
- Men: DiD = 0.0472 (SE: 0.0123), N = 25,058
- Women: DiD = 0.0437 (SE: 0.0178), N = 19,667
- Conclusion: Similar effects for both genders

**3. Placebo Test (Pre-Period 2006-2008 vs 2009-2011)**:
- Placebo DiD: 0.0054 (SE: 0.0125), p = 0.664
- Conclusion: No evidence of differential pre-trends

### Step 7: Yearly Trends in Full-Time Employment

| Year | Treatment | Control | Difference |
|------|-----------|---------|------------|
| 2006 | 0.638 | 0.693 | -0.055 |
| 2007 | 0.660 | 0.723 | -0.063 |
| 2008 | 0.660 | 0.692 | -0.031 |
| 2009 | 0.612 | 0.645 | -0.033 |
| 2010 | 0.599 | 0.629 | -0.031 |
| 2011 | 0.580 | 0.630 | -0.050 |
| 2013 | 0.642 | 0.632 | +0.010 |
| 2014 | 0.637 | 0.617 | +0.020 |
| 2015 | 0.659 | 0.666 | -0.007 |
| 2016 | 0.699 | 0.654 | +0.046 |

### Step 8: Report Generation
- LaTeX report created: `replication_report_86.tex`
- PDF compiled: `replication_report_86.pdf` (18 pages)

---

## Key Decisions and Justifications

1. **Non-citizen as proxy for undocumented**: Following instructions, used CITIZEN == 3 as proxy since ACS does not directly identify documentation status.

2. **Arrived by 2007**: Used YRIMMIG <= 2007 to approximate the requirement of continuous presence since June 15, 2007.

3. **Excluding 2012**: DACA was implemented on June 15, 2012, so the year is excluded since we cannot distinguish pre/post observations.

4. **Age groups 26-30 vs 31-35**: These ages on June 15, 2012 correspond to birth years 1982-1986 (treatment) and 1977-1981 (control).

5. **Full-time threshold**: Used UHRSWORK >= 35 as specified in the research question.

6. **Person weights**: Used PERWT for all weighted analyses to account for sampling design.

7. **Robust standard errors**: Used HC1 (heteroskedasticity-consistent) standard errors.

8. **Preferred specification**: Model 4 with year and state fixed effects chosen as most conservative specification.

---

## Output Files Generated

| File | Description |
|------|-------------|
| `analysis_86.py` | Main analysis script |
| `results_summary_86.csv` | Summary of DiD coefficients across models |
| `descriptive_stats_86.csv` | Descriptive statistics by group and period |
| `yearly_trends_86.csv` | Year-by-year employment rates |
| `replication_report_86.tex` | LaTeX source for report |
| `replication_report_86.pdf` | Final 18-page report |
| `run_log_86.md` | This run log |

---

## Final Summary

**Research Question**: Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born non-citizens.

**Method**: Difference-in-differences comparing ages 26-30 (eligible) to 31-35 (ineligible due to age cutoff) before vs. after DACA implementation.

**Main Finding**: DACA eligibility increased full-time employment by 4.58 percentage points (95% CI: 2.51 to 6.64 pp, p < 0.001).

**Robustness**: Effect is robust to controls, fixed effects, and consistent across genders. Placebo test supports parallel trends assumption.

---

*Log completed: 2026-01-26*
