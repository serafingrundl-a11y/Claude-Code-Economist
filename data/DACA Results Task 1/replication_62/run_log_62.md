# DACA Replication Study - Run Log 62

## Study Overview
**Research Question**: Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability of full-time employment (defined as usually working 35 hours per week or more)?

**Policy Implementation Date**: June 15, 2012
**Analysis Period**: Effects examined in years 2013-2016

---

## Session Log

### Session Start: 2026-01-25

#### 1. Initial Data Exploration

**Data Files Available:**
- `data.csv` - Main ACS data file (~6.2 GB)
- `acs_data_dict.txt` - Data dictionary with variable definitions
- `state_demo_policy.csv` - Optional state-level data
- `State Level Data Documentation.docx` - Documentation for state data

**ACS Years in Data:** 2006-2016 (one-year ACS samples)

**Key Variables Identified:**
- YEAR: Census year
- HISPAN/HISPAND: Hispanic origin (1 = Mexican)
- BPL/BPLD: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- UHRSWORK: Usual hours worked per week (outcome: >= 35 for full-time)
- PERWT: Person weights
- AGE: Age

#### 2. DACA Eligibility Criteria (per instructions)

Individuals were eligible for DACA if they:
1. Arrived in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status
5. (Additional requirements: were in school, graduated, or honorably discharged - cannot verify in ACS)

**Key Decisions:**

1. **Sample Restriction**: Hispanic-Mexican ethnicity (HISPAN=1) AND born in Mexico (BPL=200)

2. **Identification of Non-citizens**: CITIZEN = 3 (Not a citizen). Per instructions, assume non-citizens without naturalization papers are undocumented.

3. **Age at arrival calculation**: Use YRIMMIG - BIRTHYR to determine age at arrival. Those arriving before age 16 meet criterion 1.

4. **Age criterion for DACA (born after June 15, 1981)**: For observations, check if BIRTHYR >= 1982, OR if BIRTHYR == 1981 AND BIRTHQTR >= 3 (July-Sept or Oct-Dec, as born after June 15).

5. **Continuous presence since 2007**: Use YRIMMIG <= 2007 as a proxy.

6. **Comparison Group Strategy**: Use a difference-in-differences design:
   - Treatment group: DACA-eligible Mexican-born non-citizens
   - Control group: Mexican-born non-citizens who do NOT meet DACA criteria (e.g., arrived too old, or too old as of 2012)
   - Time periods: Pre (2006-2011) vs Post (2013-2016)
   - Note: 2012 is ambiguous (DACA implemented mid-year) - will exclude or treat separately

7. **Outcome Variable**: Full-time employment = UHRSWORK >= 35

8. **Working Age Restriction**: Limit sample to working-age individuals (e.g., 18-64 or similar)

---

#### 3. Analytical Approach

**Primary Method**: Difference-in-Differences (DiD)

**Model Specification**:
Y_it = β0 + β1*DACA_eligible_i + β2*Post_t + β3*(DACA_eligible_i × Post_t) + X_it*γ + ε_it

Where:
- Y_it = Full-time employment indicator (UHRSWORK >= 35)
- DACA_eligible_i = 1 if individual meets DACA criteria, 0 otherwise
- Post_t = 1 if year >= 2013, 0 if year <= 2011
- β3 = Causal effect of interest (DiD estimator)
- X_it = Control variables (age, sex, education, marital status, state, etc.)

**Control Variables to Include**:
- AGE (and AGE^2)
- SEX
- MARST (marital status)
- EDUC (education)
- STATEFIP (state fixed effects)
- YEAR (year fixed effects)

---

#### 4. Code Development

Analysis code written in Python using:
- pandas (data manipulation)
- statsmodels (regression analysis)
- numpy (numerical operations)

**File**: `analysis.py`

---

#### 5. Analysis Results

**Sample Construction:**
- Total ACS observations scanned: 33,851,424
- After Hispanic-Mexican filter: 701,347
- After Mexico birthplace filter: (same as above, done together)
- After non-citizen filter: (same as above)
- After working age (18-64) restriction: 603,425
- After excluding 2012: 547,614
- After excluding group quarters: 529,973
- After cleaning age at arrival: 528,360 (FINAL SAMPLE)

**DACA Eligibility Identification:**
- Arrived before age 16: 120,023 (22.7%)
- Born after June 1981: 142,552 (27.0%)
- Present since 2007: 500,316 (94.7%)
- DACA eligible (all criteria met): 67,501 (12.8%)
- Non-eligible comparison group: 460,859 (87.2%)

**Descriptive Statistics:**

| Group | Period | N | Full-time Rate | Employment Rate | Mean Age |
|-------|--------|---|----------------|-----------------|----------|
| Non-eligible | Pre (2006-2011) | 289,478 | 61.1% | 66.9% | 38.3 |
| Non-eligible | Post (2013-2016) | 171,381 | 58.8% | 67.6% | 42.0 |
| DACA-eligible | Pre (2006-2011) | 36,276 | 51.7% | 60.8% | 22.2 |
| DACA-eligible | Post (2013-2016) | 31,225 | 56.0% | 68.8% | 25.2 |

**Simple DiD Calculation:**
- DACA-eligible change: 56.0% - 51.7% = +4.3 pp
- Non-eligible change: 58.8% - 61.1% = -2.3 pp
- **Simple DiD estimate: 6.6 percentage points**

---

#### 6. Main Results

**Preferred Estimate: Model 4 (with year fixed effects)**

| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0182 |
| Standard Error | 0.0046 |
| P-value | 0.0001 |
| 95% CI | [0.0092, 0.0273] |
| N | 528,360 |
| R-squared | 0.2323 |

**Interpretation:** DACA eligibility is associated with a statistically significant 1.82 percentage point increase in the probability of full-time employment.

**Model Comparison:**

| Model | DiD Coef | SE | P-value |
|-------|----------|-----|---------|
| (1) Basic DiD | 0.0702 | 0.0050 | <0.001 |
| (2) + Demographics | 0.0284 | 0.0046 | <0.001 |
| (3) + Education | 0.0269 | 0.0046 | <0.001 |
| (4) + Year FE | 0.0182 | 0.0046 | <0.001 |
| (5) + Year & State FE | 0.0175 | 0.0046 | <0.001 |

---

#### 7. Robustness Checks

| Specification | DiD Coef | SE | P-value |
|---------------|----------|-----|---------|
| Employment (any) | 0.0266 | 0.0044 | <0.001 |
| Ages 18-35 only | 0.0096 | 0.0051 | 0.061 |
| Men only | 0.0124 | 0.0059 | 0.037 |
| Women only | 0.0172 | 0.0070 | 0.014 |
| Full-time = 40+ hrs | 0.0152 | 0.0047 | 0.001 |

---

#### 8. Event Study Results (Reference: 2011)

| Year | Coefficient | SE | P-value | Interpretation |
|------|-------------|-----|---------|----------------|
| 2006 | 0.0132 | 0.0111 | 0.233 | Pre-trend |
| 2007 | 0.0086 | 0.0106 | 0.417 | Pre-trend |
| 2008 | 0.0200 | 0.0106 | 0.059 | Pre-trend |
| 2009 | 0.0205 | 0.0105 | 0.051 | Pre-trend |
| 2010 | 0.0152 | 0.0102 | 0.137 | Pre-trend |
| 2013 | 0.0135 | 0.0100 | 0.178 | Post (1st year) |
| 2014 | 0.0276 | 0.0101 | 0.006 | Post |
| 2015 | 0.0439 | 0.0099 | <0.001 | Post |
| 2016 | 0.0447 | 0.0101 | <0.001 | Post |

**Note:** Pre-trends are generally not statistically significant, supporting the parallel trends assumption. Effects grow larger over time in the post-period.

---

#### 9. Key Decisions Summary

1. **Sample restriction**: Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200), non-citizen (CITIZEN=3), ages 18-64
2. **Year exclusion**: 2012 excluded due to mid-year DACA implementation
3. **DACA eligibility**: Combination of (1) arrived before age 16, (2) born after June 15, 1981, (3) immigrated by 2007
4. **Outcome**: Full-time employment defined as UHRSWORK >= 35
5. **Comparison group**: Mexican-born non-citizens who don't meet DACA criteria
6. **Method**: Difference-in-differences with year fixed effects (preferred), robust standard errors (HC1)
7. **Weights**: Person weights (PERWT) used in all regressions

---

#### 10. Files Generated

- `analysis.py` - Main analysis script
- `analysis_results.txt` - Detailed regression output
- `descriptive_stats.csv` - Summary statistics by group
- `latex_results.json` - Results formatted for LaTeX report
- `replication_report_62.tex` - LaTeX report (37 KB)
- `replication_report_62.pdf` - Final PDF report (23 pages, 348 KB)

---

## Session Complete

**Final Deliverables Created:**
1. `replication_report_62.tex` - LaTeX source file
2. `replication_report_62.pdf` - Compiled 23-page PDF report
3. `run_log_62.md` - This run log file

**Summary of Key Finding:**
DACA eligibility is associated with a statistically significant **1.82 percentage point** increase in the probability of full-time employment among Hispanic-Mexican, Mexican-born non-citizens in the United States (95% CI: [0.92, 2.73], p < 0.001, N = 528,360).

