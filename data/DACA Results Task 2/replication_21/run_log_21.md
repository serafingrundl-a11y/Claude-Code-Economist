# Run Log - DACA Replication Study (Run 21)

## Overview
This log documents all key decisions and commands executed during the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35+ hours per week)?

## Research Design
- **Treatment Group**: Individuals aged 26-30 as of June 15, 2012 (eligible for DACA)
- **Control Group**: Individuals aged 31-35 as of June 15, 2012 (too old for DACA but otherwise similar)
- **Method**: Difference-in-Differences (DiD) comparing pre-period (2006-2011) to post-period (2013-2016)
- **Note**: 2012 is excluded from analysis because the month of ACS data collection is unknown

---

## Session Log

### Step 1: Data Exploration and Understanding
**Timestamp**: Session start

**Actions**:
1. Read replication instructions from replication_instructions.docx
2. Examined data dictionary (acs_data_dict.txt)
3. Reviewed data structure of data.csv (33,851,424 total observations)

**Key Variables Identified**:
- YEAR: Census year (2006-2016 ACS files available)
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight for survey weighting

---

### Step 2: DACA Eligibility Criteria Definition

**DACA Requirements (from instructions)**:
1. Arrived in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Operationalization Decisions**:

1. **Age Calculation**:
   - Age as of June 15, 2012 = 2012 - BIRTHYR (adjusted for birth quarter)
   - BIRTHQTR 1 (Jan-Mar) or 2 (Apr-Jun): Already had birthday by June 15
   - BIRTHQTR 3 (Jul-Sep) or 4 (Oct-Dec): Had not yet had birthday, so subtract 1

2. **Treatment Group**: Age 26-30 as of June 15, 2012
   - Born between mid-1981 and mid-1986

3. **Control Group**: Age 31-35 as of June 15, 2012
   - Born between mid-1976 and mid-1981

4. **Sample Restrictions**:
   - HISPAN == 1 (Mexican ethnicity)
   - BPL == 200 (Born in Mexico)
   - CITIZEN == 3 (Not a citizen, assumed undocumented per instructions)
   - YRIMMIG <= 2007 (in US since at least 2007)
   - Age at immigration < 16 (YRIMMIG - BIRTHYR < 16)
   - Year != 2012 (excluded due to ambiguity around June 15 cutoff)

5. **Outcome Variable**:
   - Full-time employment = 1 if UHRSWORK >= 35, 0 otherwise

---

### Step 3: Sample Construction

**Filtering Process (in order)**:
1. Started with 33,851,424 total observations
2. Filtered to HISPAN == 1, BPL == 200, CITIZEN == 3, YRIMMIG <= 2007, YEAR != 2012
3. Result: 595,366 observations
4. Restricted to age at immigration < 16: 177,294 observations
5. Restricted to ages 26-35 as of June 15, 2012: 43,238 observations (final sample)

**Final Sample Composition**:
- Treatment group (ages 26-30): 25,470 observations
- Control group (ages 31-35): 17,768 observations
- Pre-period (2006-2011): 28,377 observations
- Post-period (2013-2016): 14,861 observations

---

### Step 4: Analysis Methodology

**Main Specification** (Model 1):
- Linear probability model (WLS) with survey weights (PERWT)
- fulltime ~ treat + post + treat_post
- Heteroskedasticity-robust standard errors (HC1)

**Alternative Specifications**:
- Model 2: Added controls (female, married, educ_hs, current_age)
- Model 3: Year fixed effects instead of post indicator
- Model 4: Full specification (controls + year FE)
- Model 5: Unweighted OLS
- Model 6: Clustered standard errors by state

**Robustness Checks**:
- Pre-trends test: Linear interaction of treat x year_trend in pre-period
- Event study: Year-specific treatment effects relative to 2011

---

### Step 5: Results Summary

**Preferred Estimate (Model 1 - Basic DiD, Weighted)**:
- DiD Coefficient: 0.0590
- Standard Error: 0.0117
- 95% CI: [0.036, 0.082]
- p-value: < 0.001

**Interpretation**: DACA eligibility is associated with a 5.9 percentage point increase in the probability of full-time employment, statistically significant at the 0.1% level.

**Cell Means (Weighted)**:
|                    | Pre-DACA | Post-DACA | Change |
|--------------------|----------|-----------|--------|
| Treatment (26-30)  | 0.631    | 0.660     | +0.029 |
| Control (31-35)    | 0.673    | 0.643     | -0.030 |
| **DiD**            |          |           | **+0.059** |

**Pre-Trends Test**:
- Treat x Year Trend coefficient: 0.0026 (SE: 0.0041)
- p-value: 0.528
- Conclusion: No significant differential pre-trend (parallel trends supported)

**Robustness of Main Finding**:
| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| 1. Basic DiD (weighted) | 0.0590 | 0.0117 | <0.001 |
| 2. With Controls | 0.0479 | 0.0107 | <0.001 |
| 3. Year FE | 0.0574 | 0.0117 | <0.001 |
| 4. Full Specification | 0.0472 | 0.0107 | <0.001 |
| 5. Unweighted | 0.0516 | 0.0100 | <0.001 |
| 6. Clustered SE (state) | 0.0590 | 0.0069 | <0.001 |

**Heterogeneity Analysis**:
- By Gender: Male 0.046 (SE 0.013), Female 0.047 (SE 0.019)
- By Education: Less than HS 0.035 (SE 0.018), HS+ 0.079 (SE 0.016)
- By Marital Status: Not Married 0.069 (SE 0.017), Married 0.062 (SE 0.016)

---

### Step 6: Output Files Generated

**Analysis Files**:
- `analysis_21.py`: Python script for all analysis
- `results_21.csv`: Main regression results
- `cell_means_21.csv`: 2x2 table of cell means
- `event_study_21.csv`: Event study coefficients

**Report Files**:
- `replication_report_21.tex`: LaTeX source document
- `replication_report_21.pdf`: Compiled PDF report (19 pages)

**Documentation**:
- `run_log_21.md`: This file

---

## Key Analytical Decisions

1. **Exclusion of 2012**: The ACS does not provide month of data collection, so it's impossible to distinguish pre- vs post-DACA observations in 2012. Excluded entirely.

2. **Age calculation with birth quarter adjustment**: Individuals born in Q3-Q4 (after mid-year) would not have had their birthday by June 15, so their age is calculated as 2012 - BIRTHYR - 1.

3. **Non-citizen as proxy for undocumented**: Per instructions, assumed non-citizens (CITIZEN==3) who arrived before age 16 and have been in US since 2007 are undocumented.

4. **Full-time threshold**: Defined as 35+ usual hours per week, per research question specification.

5. **Survey weights**: Used PERWT in all main specifications for nationally representative estimates.

6. **Standard errors**: Used heteroskedasticity-robust (HC1) standard errors as default; also tested clustered SE by state.

7. **Reference year for event study**: Used 2011 (last pre-treatment year) as reference category.

---

## Software Environment

- Python 3.14
- pandas, numpy, statsmodels, scipy
- pdflatex (MiKTeX)

---

## Conclusion

The analysis finds robust evidence that DACA eligibility increased full-time employment by approximately 5.9 percentage points among the target population. This finding is stable across multiple specifications and supported by pre-trends analysis validating the parallel trends assumption.

---

*End of Run Log*
