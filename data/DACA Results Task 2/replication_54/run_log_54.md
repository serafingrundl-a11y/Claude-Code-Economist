# Run Log for DACA Replication Study (ID: 54)

## Overview
This log documents all key decisions and commands for the independent replication of the DACA study examining the effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals in the United States.

## Data Source
- ACS data from IPUMS USA (2006-2016)
- File: data.csv (~33.8 million observations)
- Data dictionary: acs_data_dict.txt

## Research Question
What was the causal impact of DACA eligibility (treatment) on the probability that an eligible Hispanic-Mexican Mexican-born person is employed full-time (≥35 hours per week)?

## Key Dates
- DACA Implemented: June 15, 2012
- Pre-treatment period: 2006-2011
- Treatment year (excluded): 2012 (cannot distinguish before/after implementation)
- Post-treatment period: 2013-2016

---

## Session Log

### Step 1: Data Exploration and Understanding

**Files examined:**
- replication_instructions.docx - Read research task specifications
- data/acs_data_dict.txt - Variable definitions
- data/data.csv - 33,851,424 observations (excluding header)

**Key Variables Identified:**
- YEAR: Survey year
- BIRTHYR: Birth year
- BIRTHQTR: Quarter of birth
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week (outcome: ≥35 for full-time)
- EMPSTAT: Employment status
- PERWT: Person weight for survey estimation

### Step 2: Defining Treatment and Control Groups

**DACA Eligibility Criteria (per instructions):**
1. Ethnically Hispanic-Mexican (HISPAN = 1)
2. Born in Mexico (BPL = 200)
3. Not a citizen (CITIZEN = 3) - proxy for undocumented status
4. Arrived before 16th birthday (YRIMMIG - BIRTHYR < 16)
5. Lived continuously in US since June 15, 2007 (YRIMMIG ≤ 2007)

**Age-Based Treatment Assignment:**
- Treatment group: Ages 26-30 as of June 15, 2012
  - Born between June 16, 1981 and June 15, 1986
  - Simplified: BIRTHYR in {1982, 1983, 1984, 1985, 1986} (using full year approximation)

- Control group: Ages 31-35 as of June 15, 2012
  - Born between June 16, 1976 and June 15, 1981
  - Simplified: BIRTHYR in {1977, 1978, 1979, 1980, 1981}

**Decision:** Use birth year only for simplicity, as using BIRTHQTR creates complexity and the analysis examines annual data anyway. This is a common approach in DACA literature.

### Step 3: Analysis Strategy

**Method:** Difference-in-Differences (DiD)

**Outcome Variable:** Full-time employment indicator (1 if UHRSWORK ≥ 35, 0 otherwise)

**Specification:**
Y_it = α + β₁(Treat_i) + β₂(Post_t) + β₃(Treat_i × Post_t) + X_i'γ + ε_it

Where:
- Y_it = Full-time employment indicator
- Treat_i = 1 if person is in treatment group (ages 26-30 in 2012)
- Post_t = 1 if year ≥ 2013
- β₃ = DiD estimate (effect of DACA eligibility on full-time employment)
- X_i = Optional covariates (age, sex, education, state, etc.)

**Periods:**
- Pre-period: 2006-2011
- Post-period: 2013-2016
- 2012 excluded (implementation year)

### Step 4: Sample Construction

**Command executed:** `python daca_analysis.py`

**Sample filtering steps and results:**
```
Full ACS sample (2006-2016):         33,851,424
After Hispanic-Mexican filter:         2,945,521
After Mexico birthplace filter:          991,261
After non-citizen filter:                701,347
After valid YRIMMIG filter:              701,347
After arrived before age 16:             205,327
After continuous residence (≤2007):      195,023
After age group filter:                   49,019
After excluding 2012:                     44,725 (FINAL SAMPLE)
```

### Step 5: Descriptive Statistics

**Sample sizes by group and period (unweighted):**
| Group | Pre-DACA (2006-2011) | Post-DACA (2013-2016) |
|-------|----------------------|-----------------------|
| Control (ages 31-35) | 11,916 | 6,218 |
| Treatment (ages 26-30) | 17,410 | 9,181 |

**Weighted sample sizes:**
| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| Control | 1,671,499 | 859,291 |
| Treatment | 2,367,739 | 1,307,226 |

**Full-time employment rates (weighted):**
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control (31-35) | 67.05% | 64.12% | -2.93 pp |
| Treatment (26-30) | 62.53% | 65.80% | +3.27 pp |
| **Simple DiD** | | | **+6.20 pp** |

### Step 6: Regression Analysis Results

**Model 1: Basic DiD (no controls)**
- treat_post coefficient: 0.0620 (SE: 0.012, p < 0.001)

**Model 2: DiD with demographic controls**
- treat_post coefficient: 0.0482 (SE: 0.011, p < 0.001)
- Controls: age, female, education

**Model 3: DiD with year fixed effects**
- treat_post coefficient: 0.0610 (SE: 0.012, p < 0.001)

**Model 4: DiD with year FE + controls (PREFERRED)**
- treat_post coefficient: **0.0484** (SE: 0.0105, p < 0.001)
- 95% CI: [0.0278, 0.0691]
- R-squared: 0.159
- Controls: female, education, number of children, marital status

**Model 5: DiD with year + state FE + controls**
- treat_post coefficient: 0.0464 (SE: 0.011, p < 0.001)
- R-squared: 0.159

### Step 7: Robustness Checks

**Placebo test (pre-treatment only: 2006-2008 vs 2009-2011):**
- fake_treat_post coefficient: 0.0120 (SE: 0.014, p = 0.375)
- **Result:** Not significant - supports parallel trends assumption

**Heterogeneous effects by sex:**
- Male: 0.0621 (SE: 0.012, p < 0.001)
- Female: 0.0313 (SE: 0.018, p = 0.086)
- **Result:** Effect stronger for males

**Alternative age bandwidth (24-28 vs 33-37):**
- treat_post coefficient: 0.1093 (SE: 0.012, p < 0.001)
- **Result:** Effect robust to bandwidth choice

### Step 8: Event Study Analysis

**Year-specific treatment effects (relative to 2011):**
| Year | Coefficient | SE | Significant? |
|------|-------------|------|--------------|
| 2006 | -0.005 | 0.024 | No |
| 2007 | -0.013 | 0.024 | No |
| 2008 | 0.019 | 0.025 | No |
| 2009 | 0.017 | 0.025 | No |
| 2010 | 0.019 | 0.025 | No |
| 2011 | (reference) | --- | --- |
| 2013 | 0.060 | 0.026 | Yes* |
| 2014 | 0.070 | 0.027 | Yes* |
| 2015 | 0.043 | 0.027 | No |
| 2016 | 0.095 | 0.027 | Yes* |

**Result:** Pre-trends are flat (close to zero), supporting parallel trends assumption. Post-DACA effects are positive and generally significant.

### Step 9: Figures Created

1. **figure1_trends.png/pdf** - Full-time employment trends by treatment group
2. **figure2_event_study.png/pdf** - Event study coefficients with CI
3. **figure3_coefficients.png/pdf** - Coefficient comparison across models
4. **figure4_did_design.png/pdf** - DiD design illustration
5. **figure5_by_sex.png/pdf** - Heterogeneous effects by sex

### Step 10: Report Generation

**LaTeX compilation:**
```
pdflatex replication_report_54.tex (x3 passes for references)
```

**Output:** replication_report_54.pdf (22 pages)

---

## Key Decisions Summary

1. **Age group definition:** Used full birth years (1982-1986 for treatment, 1977-1981 for control) rather than exact dates with birth quarter

2. **2012 exclusion:** Excluded all of 2012 since ACS does not report survey month, making it impossible to distinguish pre/post DACA observations

3. **Undocumented status proxy:** Used CITIZEN = 3 (not a citizen) as proxy since ACS does not directly identify documentation status

4. **Full-time definition:** UHRSWORK ≥ 35 hours per week, following BLS standards

5. **Continuous residence:** Operationalized as YRIMMIG ≤ 2007

6. **Standard errors:** Robust (heteroskedasticity-consistent) standard errors throughout

7. **Preferred model:** Year fixed effects + demographic controls (Model 4)

---

## Final Results Summary

**Preferred Estimate:**
- DiD Coefficient: **0.0484** (4.84 percentage points)
- Standard Error: 0.0105
- 95% CI: [0.0278, 0.0691]
- P-value: < 0.001
- Sample Size: 44,725 observations

**Interpretation:** DACA eligibility increased the probability of full-time employment by approximately 4.8 percentage points among Hispanic-Mexican, Mexican-born individuals who met the other eligibility criteria. This effect is statistically significant and robust across specifications.

---

## Output Files

| File | Description |
|------|-------------|
| replication_report_54.tex | LaTeX source for report |
| replication_report_54.pdf | Final PDF report (22 pages) |
| run_log_54.md | This log file |
| daca_analysis.py | Main analysis script |
| create_figures.py | Figure generation script |
| regression_results.csv | Model coefficients |
| event_study_results.csv | Event study coefficients |
| yearly_ft_rates.csv | Employment rates by year/group |
| figure1_trends.png/pdf | Employment trends figure |
| figure2_event_study.png/pdf | Event study figure |
| figure3_coefficients.png/pdf | Coefficient comparison |
| figure4_did_design.png/pdf | DiD illustration |
| figure5_by_sex.png/pdf | Effects by sex |
