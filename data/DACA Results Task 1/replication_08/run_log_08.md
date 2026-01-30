# Run Log for DACA Replication Study - Replication 08

## Date: 2025-01-25

---

## 1. Study Overview

**Research Question:** Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program (treatment) on the probability that the eligible person is employed full-time (outcome), defined as usually working 35 hours per week or more?

**Time Period:** Examine effects on full-time employment in 2013-2016 (post-DACA implementation).

---

## 2. Data Source

- **Primary Data:** American Community Survey (ACS) 2006-2016 from IPUMS USA
- **Data File:** `data/data.csv` (pre-downloaded)
- **Data Dictionary:** `data/acs_data_dict.txt`
- **Optional:** State-level demographic and policy data (`state_demo_policy.csv`)

---

## 3. Key DACA Eligibility Criteria (from instructions)

DACA eligibility requires:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status (citizenship or legal residency) at that time

**Sample Restriction:** Hispanic-Mexican ethnicity AND born in Mexico

---

## 4. Variable Definitions

### Key Variables from Data Dictionary:
- **YEAR**: Census/survey year (2006-2016)
- **HISPAN**: Hispanic origin (1 = Mexican)
- **BPL/BPLD**: Birthplace (200/20000 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth
- **AGE**: Age at time of survey
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status
- **PERWT**: Person weight

### Constructed Variables:
- **full_time**: = 1 if UHRSWORK >= 35, = 0 otherwise
- **daca_eligible**: Based on eligibility criteria above
- **post**: = 1 if YEAR >= 2013, = 0 otherwise

---

## 5. Identification Strategy

**Difference-in-Differences (DiD) Design:**
- Treatment Group: DACA-eligible Hispanic-Mexican Mexican-born non-citizens
- Control Group: Non-DACA-eligible Hispanic-Mexican Mexican-born non-citizens (due to age/arrival timing)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA implementation)
- Note: 2012 is excluded due to mid-year DACA implementation (June 15, 2012)

**Regression Specification:**
```
Y_it = α + β₁·DACA_eligible_i + β₂·Post_t + β₃·(DACA_eligible_i × Post_t) + X_it'γ + ε_it
```

Where β₃ is the DiD estimator of the DACA effect.

---

## 6. Commands and Decisions Log

### Step 1: Data Exploration
- Examined data dictionary (`acs_data_dict.txt`) to understand variable coding
- Confirmed data covers years 2006-2016 (11 years)
- Identified key variables for eligibility determination
- Data file is 6.2 GB CSV with 33.8 million observations

### Step 2: Define Eligibility Criteria Operationalization

**Age Criteria:**
- Must be born on or after June 15, 1981 (to be under 31 as of June 15, 2012)
- Implementation: BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)

**Arrival Criteria:**
- Must have arrived before age 16
- Age at arrival = YRIMMIG - BIRTHYR
- Condition: (YRIMMIG - BIRTHYR) < 16 AND (YRIMMIG - BIRTHYR) >= 0

**Continuous Presence:**
- Lived in US since June 15, 2007 (5 years before DACA)
- YRIMMIG <= 2007

**Citizenship:**
- CITIZEN == 3 (Not a citizen, no papers received)

### Step 3: Sample Construction

| Step | Observations | Notes |
|------|-------------|-------|
| Initial ACS 2006-2016 | 33,851,424 | Full dataset |
| Hispanic-Mexican & Mexico-born | 991,261 | HISPAN=1, BPL=200 |
| Exclude 2012 | 898,879 | Mid-year DACA implementation |
| Non-citizens only | 636,722 | CITIZEN=3 |
| Ages 18-64 | 547,614 | Working-age population |

### Step 4: Analysis Commands

```bash
# Run main analysis script
python daca_analysis.py
```

Python script performs:
1. Data loading with chunked processing (1M rows at a time) to handle memory
2. Sample restrictions
3. DACA eligibility construction
4. Outcome variable construction (full_time = UHRSWORK >= 35)
5. Difference-in-differences regressions with clustered standard errors
6. Robustness checks
7. Event study analysis

---

## 7. Key Results

### Main Finding (Preferred Specification)
- **DiD Coefficient:** 0.0251
- **Standard Error:** 0.0038
- **95% CI:** [0.0176, 0.0326]
- **p-value:** < 0.001
- **Sample Size:** 547,614

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 2.5 percentage points.

### Raw DiD Calculation
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Non-Eligible | 0.604 | 0.581 | -0.023 |
| DACA-Eligible | 0.511 | 0.548 | +0.037 |
| **DiD** | | | **0.060** |

### Regression Results by Specification

| Model | DiD Coef | SE | N |
|-------|----------|-----|------|
| Basic DiD (unweighted) | 0.060 | 0.003 | 547,614 |
| Weighted DiD | 0.068 | 0.003 | 547,614 |
| With Controls | 0.026 | 0.004 | 547,614 |
| With Controls + State FE | 0.025 | 0.004 | 547,614 |

### Robustness Checks

| Specification | Coefficient | SE | N |
|--------------|-------------|------|------|
| Ages 16-35 only | 0.019 | 0.005 | 267,229 |
| Men only | 0.021 | 0.006 | 296,109 |
| Women only | 0.024 | 0.006 | 251,505 |
| Any employment | 0.030 | 0.005 | 547,614 |

### Event Study Results (Reference: 2011)

| Year | Coefficient | Significance |
|------|-------------|--------------|
| 2006 | 0.016 | |
| 2007 | 0.007 | |
| 2008 | 0.021 | |
| 2009 | 0.020 | |
| 2010 | 0.019 | |
| 2013 | 0.013 | |
| 2014 | 0.025 | * |
| 2015 | 0.042 | *** |
| 2016 | 0.042 | *** |

Pre-trends: No significant pre-treatment differences (supports parallel trends assumption)
Post-treatment: Effects emerge in 2014 and strengthen through 2016

---

## 8. Key Decisions Made

1. **Exclusion of 2012:** DACA was implemented on June 15, 2012, making it impossible to cleanly separate pre and post periods within that year.

2. **Non-citizen focus:** Only non-citizens (CITIZEN=3) are potentially DACA-eligible. Citizens and legal permanent residents are excluded.

3. **Age restriction (18-64):** Focus on working-age population to study employment outcomes.

4. **Birth quarter coding:** BIRTHQTR >= 3 corresponds to July-December (quarters 3 and 4), used for precise age determination.

5. **Continuous presence via YRIMMIG:** Assumed individuals who immigrated by 2007 were continuously present since then.

6. **Clustering:** Standard errors clustered at state level to account for within-state correlation.

7. **Weighting:** Used ACS person weights (PERWT) for population-representative estimates.

8. **Control group:** Non-eligible individuals are those who arrived at age 16+ OR were over 30 as of June 2012 OR arrived after 2007.

---

## 9. Output Files Generated

1. `summary_stats.csv` - Means by group and period
2. `regression_results.csv` - Main regression coefficients
3. `event_study_results.csv` - Year-by-year effects
4. `replication_report_08.tex` - Full LaTeX report
5. `replication_report_08.pdf` - Compiled 20-page report

---

## 10. Software and Packages

- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- statsmodels (regression analysis)
- scipy (statistical functions)
- pdflatex (LaTeX compilation)

---

## 11. Replication Instructions

1. Ensure data file `data/data.csv` is present
2. Run: `python daca_analysis.py`
3. Compile report: `pdflatex replication_report_08.tex` (run twice for references)

---

## 12. Conclusion

The analysis finds that DACA eligibility increased full-time employment probability by approximately 2.5 percentage points (p < 0.001). This effect is robust across specifications including demographic controls, state fixed effects, weighted estimation, and subgroup analyses. Event study results support the parallel trends assumption and show effects emerging gradually after 2012, consistent with the phased rollout of DACA protections.
