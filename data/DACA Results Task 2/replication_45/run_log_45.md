# Run Log for DACA Replication Study (Replication 45)

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA employment effects study.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (defined as usually working 35 hours per week or more)?

## Identification Strategy
- **Treatment group**: Individuals ages 26-30 at the time of DACA implementation (June 15, 2012)
- **Control group**: Individuals ages 31-35 at the time of DACA implementation
- **Method**: Difference-in-Differences (DiD) comparing changes in full-time employment rates before vs. after DACA between treatment and control groups
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016 (excluding 2012 due to mid-year implementation)

## Data Sources
- American Community Survey (ACS) 2006-2016 from IPUMS
- Data file: data.csv (6.26 GB)
- Data dictionary: acs_data_dict.txt

---

## Session Log

### Step 1: Data Exploration
**Date**: 2026-01-26

**Files examined**:
- replication_instructions.docx - Read and parsed research instructions
- acs_data_dict.txt - Examined variable definitions
- data.csv - Confirmed CSV structure with headers

**Key variables identified**:
- YEAR: Survey year (2006-2016)
- BIRTHYR: Birth year for calculating age at DACA implementation
- BIRTHQTR: Birth quarter (for more precise age calculation)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL/BPLD: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration (to verify continuous presence requirement)
- UHRSWORK: Usual hours worked per week (outcome: >=35 = full-time)
- PERWT: Person weight for population estimates

### Step 2: Sample Definition Decisions

**DACA Eligibility Criteria** (from instructions):
1. Arrived in US before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status

**Operationalization Decisions**:

1. **Hispanic-Mexican, Mexican-born**:
   - HISPAN = 1 (Mexican)
   - BPL = 200 (Mexico)

2. **Non-citizen (undocumented proxy)**:
   - CITIZEN = 3 (Not a citizen)
   - Per instructions: assume non-citizens without immigration papers are undocumented

3. **Age at DACA implementation** (June 15, 2012):
   - Treatment: Born 1982-1986 (ages 26-30 on June 15, 2012)
   - Control: Born 1977-1981 (ages 31-35 on June 15, 2012)

4. **Arrived before 16th birthday**:
   - YRIMMIG (year of immigration) - BIRTHYR < 16

5. **Continuous presence since June 15, 2007**:
   - YRIMMIG <= 2007 (arrived by 2007)

6. **Full-time employment outcome**:
   - UHRSWORK >= 35

### Step 3: Analysis Approach

**Primary specification**: Difference-in-Differences regression

Model: FT_employed_it = β0 + β1*Treatment_i + β2*Post_t + β3*(Treatment_i × Post_t) + ε_it

Where:
- FT_employed_it = 1 if usually works 35+ hours/week, 0 otherwise
- Treatment_i = 1 if ages 26-30 at DACA, 0 if ages 31-35
- Post_t = 1 if year >= 2013, 0 if year <= 2011 (excluding 2012)
- β3 = DiD estimate (effect of interest)

**Additional specifications**:
1. Add demographic controls (sex, education, marital status)
2. Add state fixed effects
3. Add year fixed effects
4. Clustered standard errors at state level

### Step 4: Running Analysis

**Command**: Python script for data processing and analysis
```
python daca_analysis.py
```

**Output Summary**:
- Total observations loaded: 33,851,424
- After Hispanic-Mexican filter: 2,945,521
- After Mexico birthplace filter: 991,261
- After non-citizen filter: 701,347
- After excluding 2012: 636,722
- After selecting age groups (26-35): 162,283
- After arrival before age 16 filter: 44,725
- After continuous presence filter: 44,725
- **FINAL ANALYTIC SAMPLE: 44,725**

### Step 5: Results Summary

**Sample Composition**:
- Treatment group (ages 26-30 at DACA): 26,591 observations
- Control group (ages 31-35 at DACA): 18,134 observations
- Pre-period observations: 29,326
- Post-period observations: 15,399

**Raw DiD Table**:
|                | Pre-DACA | Post-DACA | Difference |
|----------------|----------|-----------|------------|
| Control (31-35)| 0.643    | 0.611     | -0.032     |
| Treatment (26-30)| 0.611  | 0.634     | +0.023     |
| **DiD Estimate**|         |           | **0.055**  |

**Regression Results**:

| Model | Description | Coefficient | SE | 95% CI |
|-------|-------------|-------------|-----|--------|
| 1 | Basic DiD | 0.0551 | 0.0098 | [0.036, 0.074] |
| 2 | +Controls | 0.0485 | 0.0091 | [0.031, 0.066] |
| 3 | +State FE | 0.0476 | 0.0091 | [0.030, 0.066] |
| 4 | +Year FE | 0.0477 | 0.0091 | [0.030, 0.066] |
| 5 | Clustered SE | **0.0477** | **0.0094** | **[0.029, 0.066]** |
| Weighted | WLS w/ PERWT | 0.0496 | 0.0089 | [0.032, 0.067] |

**Preferred Estimate (Model 5)**:
- **Effect**: 4.77 percentage point increase in full-time employment
- **Standard Error**: 0.0094 (clustered at state level)
- **95% CI**: [0.029, 0.066]
- **p-value**: < 0.001

**Event Study (Pre-trends Check)**:
- 2006: -0.030 (p=0.100) - Not significant
- 2007: -0.024 (p=0.191) - Not significant
- 2008: 0.002 (p=0.907) - Not significant
- 2009: -0.005 (p=0.796) - Not significant
- 2010: -0.010 (p=0.589) - Not significant
- 2011: Reference year
- 2013: 0.029 (p=0.130) - Not significant
- 2014: 0.029 (p=0.140) - Not significant
- 2015: 0.036 (p=0.074) - Marginally significant
- 2016: 0.052 (p=0.011) - Significant

**Subgroup Analysis**:
- Male: 0.043 (SE=0.011)
- Female: 0.045 (SE=0.015)
- Less than HS: 0.028 (SE=0.014)
- HS or more: 0.067 (SE=0.012)

### Step 6: Generating Figures

**Command**: Python script for figure generation
```
python create_figures.py
```

**Figures created**:
1. figure1_event_study.png/pdf - Event study plot
2. figure2_trends.png/pdf - Treatment and control trends
3. figure3_did.png/pdf - DiD visualization
4. figure4_robustness.png/pdf - Coefficient comparison
5. figure5_subgroups.png/pdf - Subgroup analysis

### Step 7: Writing LaTeX Report

**File**: replication_report_45.tex

**Sections**:
1. Abstract
2. Introduction
3. Background on DACA
4. Data and Sample Construction
5. Empirical Methodology
6. Results
7. Discussion
8. Conclusion
9. Appendix: Analytical Decisions

### Step 8: Compiling PDF

**Command**:
```
pdflatex -interaction=nonstopmode replication_report_45.tex
pdflatex -interaction=nonstopmode replication_report_45.tex
```

**Output**: replication_report_45.pdf (28 pages)

---

## Final Deliverables

1. **replication_report_45.tex** - LaTeX source file
2. **replication_report_45.pdf** - Compiled report (28 pages)
3. **run_log_45.md** - This log file
4. **daca_analysis.py** - Analysis script
5. **create_figures.py** - Figure generation script
6. **analysis_results.json** - Saved results
7. **summary_statistics.csv** - Descriptive statistics
8. **yearly_means.csv** - Yearly employment rates
9. **figure1_event_study.png/pdf** - Event study plot
10. **figure2_trends.png/pdf** - Trends plot
11. **figure3_did.png/pdf** - DiD visualization
12. **figure4_robustness.png/pdf** - Robustness check
13. **figure5_subgroups.png/pdf** - Subgroup analysis

---

## Key Methodological Decisions

1. **Undocumented proxy**: Used CITIZEN = 3 (non-citizen) as proxy for undocumented status
2. **Age calculation**: Used birth year only (not quarter) for simplicity
3. **Continuous presence**: Required YRIMMIG <= 2007
4. **Full-time threshold**: UHRSWORK >= 35 hours/week
5. **Standard errors**: Clustered at state level in preferred specification
6. **Excluded 2012**: Due to mid-year DACA implementation
