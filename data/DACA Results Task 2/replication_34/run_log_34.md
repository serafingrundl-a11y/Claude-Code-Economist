# Run Log - Replication 34: DACA Impact on Full-Time Employment

## Project Overview
Independent replication of a difference-in-differences analysis examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican individuals born in Mexico.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA on the probability of being employed full-time (35+ hours/week)?

- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012) - born 1982-1986
- **Control Group**: Ages 31-35 at DACA implementation (otherwise DACA-eligible if not for age) - born 1977-1981
- **Post-treatment Period**: 2013-2016
- **Pre-treatment Period**: 2006-2011
- **Excluded**: 2012 (mid-year implementation, cannot distinguish pre/post)

---

## Session Log

### Step 1: Data Exploration
**Date**: 2026-01-26

**Files found**:
- `data/data.csv` - Main ACS data file (33,851,425 rows)
- `data/acs_data_dict.txt` - Variable codebook
- `data/state_demo_policy.csv` - Optional state-level data (not used)
- `data/State Level Data Documentation.docx` - State data documentation (not used)

**Key Variables Identified**:
| Variable | Description | Values Used |
|----------|-------------|-------------|
| YEAR | Survey year | 2006-2016 (excl. 2012) |
| HISPAN | Hispanic origin | 1 = Mexican |
| BPL | Birthplace | 200 = Mexico |
| CITIZEN | Citizenship status | 3 = Not a citizen |
| YRIMMIG | Year of immigration | Numeric |
| BIRTHYR | Birth year | Numeric |
| UHRSWORK | Usual hours worked/week | 0-99 |
| EMPSTAT | Employment status | 1 = Employed |
| LABFORCE | Labor force status | 2 = In labor force |
| PERWT | Person weight | Numeric |
| STATEFIP | State FIPS code | For clustering |
| SEX | Sex | 1 = Male, 2 = Female |
| MARST | Marital status | 1-2 = Married |
| EDUC | Education | >= 6 = HS or more |

---

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria Applied**:
1. Hispanic-Mexican ethnicity (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Not a citizen (CITIZEN = 3) - proxy for undocumented status
4. Arrived in US before age 16: (YRIMMIG - BIRTHYR) < 16
5. In US since 2007: YRIMMIG <= 2007

**Sample Filtering Results**:
```
Total ACS observations loaded: 33,851,425
After Hispanic-Mexican filter: [filtered]
After born in Mexico filter: [filtered]
After non-citizen filter: 701,347
After arrived before age 16 filter: 205,327
After in US since 2007 filter: 195,023
Treatment + Control only: 49,019
After excluding 2012: 44,725
```

**Final Analysis Sample**: 44,725 observations

---

### Step 3: Treatment/Control Group Definitions

| Group | Birth Years | Age at DACA (June 2012) | N |
|-------|-------------|-------------------------|---|
| Treatment | 1982-1986 | 26-30 | 26,591 |
| Control | 1977-1981 | 31-35 | 18,134 |

---

### Step 4: Outcome Variable Definition

**Full-time Employment**: UHRSWORK >= 35 (binary indicator)
- 1 if usually works 35+ hours/week
- 0 otherwise

---

### Step 5: Analysis Implementation

**Statistical Approach**: Difference-in-Differences
- Weighted least squares using PERWT
- Standard errors clustered at state level (STATEFIP)
- Year fixed effects included
- Demographic controls: female, married, HS education

**Models Estimated**:
1. Basic DiD (no controls)
2. DiD + Year FE
3. DiD + Year FE + Demographics (PREFERRED)
4. DiD + Year FE + State FE + Demographics

**Event Study**: Year-by-year treatment effects with 2011 as reference year

---

## Key Results

### Main Finding (Preferred Model 3)
| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0491 |
| Standard Error | 0.0112 |
| 95% CI | [0.0272, 0.0711] |
| p-value | < 0.001 |
| Sample Size | 44,725 |
| R-squared | 0.155 |

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately **4.9 percentage points** (95% CI: 2.7 to 7.1 pp).

Relative to the pre-treatment mean of 61.1%, this represents an **8% increase**.

### Full-Time Employment Rates
| Period | Treatment | Control | Difference |
|--------|-----------|---------|------------|
| Pre (2006-2011) | 62.53% | 67.05% | -4.52 pp |
| Post (2013-2016) | 65.80% | 64.12% | +1.68 pp |
| Change | +3.27 pp | -2.93 pp | **DiD = +6.20 pp** |

### Robustness Checks
| Outcome | DiD Coef | SE | p-value |
|---------|----------|-----|---------|
| Full-time (35+ hrs) | 0.049 | 0.011 | <0.001 |
| Employment (any) | 0.047 | 0.007 | <0.001 |
| Labor Force Participation | 0.032 | 0.006 | <0.001 |

### Heterogeneity
| Subgroup | DiD | SE | N |
|----------|-----|-----|---|
| Male | 0.061 | 0.013 | 25,058 |
| Female | 0.030 | 0.017 | 19,667 |
| Less than HS | 0.045 | 0.014 | 18,328 |
| HS or more | 0.073 | 0.017 | 26,397 |

### Parallel Trends (Event Study)
Pre-period coefficients (all insignificant, supporting parallel trends):
- 2006: -0.005 (SE 0.020)
- 2007: -0.013 (SE 0.018)
- 2008: +0.019 (SE 0.019)
- 2009: +0.017 (SE 0.019)
- 2010: +0.019 (SE 0.021)

Post-period coefficients (all significant):
- 2013: +0.060** (SE 0.024)
- 2014: +0.070*** (SE 0.020)
- 2015: +0.043** (SE 0.021)
- 2016: +0.095*** (SE 0.017)

---

## Commands Executed

```bash
# Data exploration
head -5 "data/data.csv"
wc -l "data/data.csv"
# Result: 33,851,425 rows

# Run analysis
python analysis_34.py

# Compile LaTeX report
pdflatex -interaction=nonstopmode replication_report_34.tex
pdflatex -interaction=nonstopmode replication_report_34.tex
pdflatex -interaction=nonstopmode replication_report_34.tex
```

---

## Key Analytical Decisions

1. **Excluding 2012**: The ACS doesn't identify the month of data collection, and DACA was implemented mid-year (June 2012). Including 2012 would mix pre- and post-treatment observations ambiguously.

2. **Age at DACA Implementation**: Used birth year to assign cohorts. Treatment born 1982-1986 (ages 26-30 on June 15, 2012), Control born 1977-1981 (ages 31-35).

3. **Undocumented Proxy**: Following instructions, assumed non-citizens (CITIZEN=3) without naturalization are undocumented. This is an imperfect proxy as some may be documented non-citizens.

4. **Immigration Timing for Eligibility**: Required YRIMMIG <= 2007 to approximate "continuous presence since June 2007" and (YRIMMIG - BIRTHYR) < 16 for "arrived before age 16."

5. **Weighting**: Used PERWT for all analyses to produce population-representative estimates.

6. **Standard Errors**: Clustered at state level (STATEFIP) to account for within-state correlation and any state-level policy variation.

7. **Preferred Specification**: Model 3 with year FE and demographic controls (without state FE) chosen to balance bias reduction and statistical power.

---

## Output Files

| File | Description |
|------|-------------|
| `analysis_34.py` | Main Python analysis script |
| `results_34.py` | Auto-generated results dictionary |
| `event_study_34.csv` | Event study coefficients |
| `yearly_means_34.csv` | Yearly full-time rates by group |
| `replication_report_34.tex` | LaTeX source for report |
| `replication_report_34.pdf` | Compiled 19-page report |
| `run_log_34.md` | This log file |

---

## Software Environment

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (WLS regression with clustered SEs)
- pdfLaTeX (MiKTeX) for document compilation

---

## Summary

This replication finds strong evidence that DACA eligibility increased full-time employment among the target population by approximately 4.9 percentage points. The effect is:
- Statistically significant (p < 0.001)
- Robust across specifications
- Supported by parallel pre-trends
- Larger for men and those with higher education

The analysis follows best practices for difference-in-differences estimation, including appropriate weighting, clustered standard errors, and event study validation of the parallel trends assumption.
