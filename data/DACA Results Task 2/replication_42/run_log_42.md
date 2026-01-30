# Replication Run Log - ID 42

## Project: DACA Effect on Full-Time Employment

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for the Deferred Action for Childhood Arrivals (DACA) program on the probability that the eligible person is employed full-time (defined as usually working 35 hours per week or more)?

### Design
- **Treatment group**: Ages 26-30 at the time the policy went into place (June 15, 2012)
- **Control group**: Ages 31-35 at the time the policy went into place
- **Method**: Difference-in-differences comparing treated vs. control, pre vs. post
- **Pre-period**: 2006-2011 (excluding 2012 due to implementation mid-year)
- **Post-period**: 2013-2016

---

## Session Start: 2026-01-26

### Step 1: Data Exploration
- Read replication_instructions.docx (extracted text from Word document)
- Reviewed data dictionary (acs_data_dict.txt)
- Data files available:
  - data.csv (~6.3 GB) - main ACS data 2006-2016
  - state_demo_policy.csv (optional state-level data, not used)
  - acs_data_dict.txt - variable definitions

### Key Variables Identified:
- **YEAR**: Census year (2006-2016)
- **PERWT**: Person weight for survey weighting
- **AGE**: Age of respondent
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth (for more precise age calculation)
- **HISPAN/HISPAND**: Hispanic origin (1 = Mexican)
- **BPL/BPLD**: Birthplace (200 = Mexico)
- **CITIZEN**: Citizenship status (3 = Not a citizen)
- **YRIMMIG**: Year of immigration
- **UHRSWORK**: Usual hours worked per week (>=35 is full-time)
- **SEX**: Sex (1=Male, 2=Female)
- **EDUC/EDUCD**: Educational attainment
- **MARST**: Marital status
- **STATEFIP**: State FIPS code

### DACA Eligibility Criteria (from instructions):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status
5. Non-citizen (assume all non-citizens without papers are undocumented)

---

### Step 2: Sample Construction Decisions

**Key Decisions:**
1. **Age groups at policy implementation (June 15, 2012)**:
   - Treatment: Ages 26-30 (using BIRTHYR and BIRTHQTR for precise calculation)
   - Control: Ages 31-35

2. **Eligibility criteria operationalization**:
   - Hispanic-Mexican: HISPAN == 1
   - Born in Mexico: BPL == 200
   - Non-citizen: CITIZEN == 3
   - Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
   - Continuous residence since 2007: YRIMMIG <= 2007

3. **Outcome definition**:
   - Full-time employment: UHRSWORK >= 35

4. **Period definitions**:
   - Pre-treatment: 2006-2011
   - Post-treatment: 2013-2016
   - Exclude 2012: Policy implemented mid-year (June 15)

---

### Step 3: Data Processing

**Command**: `python analysis.py`

**Processing steps**:
1. Loaded data.csv in chunks (500,000 rows) to handle ~6GB file
2. Applied initial filters: HISPAN=1, BPL=200, CITIZEN=3, YEAR in [2006-2011, 2013-2016]
3. Calculated age as of June 15, 2012 using BIRTHYR and BIRTHQTR
4. Filtered to ages 26-35 at policy implementation
5. Applied DACA eligibility: YRIMMIG > 0, YRIMMIG <= 2007, age at immigration < 16

**Sample sizes**:
- After initial demographic filters: 636,722 observations
- After age restriction (26-35): 164,874 observations
- After DACA eligibility criteria: 43,238 observations

---

### Step 4: Analysis Results

#### Main Findings (Preferred Specification - Model 4):
- **Effect Size**: 0.0475 (4.75 percentage points)
- **Standard Error**: 0.0107 (robust HC1)
- **95% CI**: [0.0265, 0.0685]
- **p-value**: < 0.0001
- **Sample Size**: 43,238 (unweighted)
- **Weighted N**: 6,000,418

#### Simple 2x2 DiD Table (Weighted Means):
|                    | Pre-DACA | Post-DACA | Difference |
|--------------------|----------|-----------|------------|
| Treatment (26-30)  | 0.6305   | 0.6597    | +0.0292    |
| Control (31-35)    | 0.6731   | 0.6433    | -0.0299    |
| **DiD Estimate**   |          |           | **0.0590** |

Note: The weighted basic DiD is 5.9 pp; with covariates and robust SE it's 4.75 pp.

#### Event Study Results (Treatment x Year, reference=2011):
| Year | Coefficient | SE     | p-value |
|------|-------------|--------|---------|
| 2006 | 0.006       | 0.023  | 0.798   |
| 2007 | -0.032      | 0.022  | 0.158   |
| 2008 | 0.008       | 0.023  | 0.734   |
| 2009 | -0.009      | 0.024  | 0.698   |
| 2010 | -0.014      | 0.023  | 0.560   |
| 2011 | (ref)       | -      | -       |
| 2013 | 0.035       | 0.024  | 0.151   |
| 2014 | 0.037       | 0.025  | 0.134   |
| 2015 | 0.020       | 0.025  | 0.418   |
| 2016 | 0.067       | 0.025  | 0.007   |

Pre-trends appear parallel (no significant pre-treatment coefficients).

#### Robustness Checks:
| Specification                     | Estimate | SE     | p-value |
|-----------------------------------|----------|--------|---------|
| Placebo (fake treatment at 2009)  | -0.002   | 0.013  | 0.843   |
| Narrow bandwidth (28-30 vs 31-33) | 0.038    | 0.014  | 0.007   |
| Any employment (alternative DV)   | 0.044    | 0.010  | <0.001  |
| Males only                        | 0.046    | 0.012  | <0.001  |
| Females only                      | 0.047    | 0.019  | 0.012   |

---

### Step 5: Output Files Generated

1. **analysis.py** - Main analysis script
2. **main_results.json** - Key results in JSON format
3. **summary_stats.csv** - Summary statistics by group/period
4. **table1_summary.csv** - Detailed summary statistics
5. **table2_did_results.csv** - DiD regression results
6. **table3_robustness.csv** - Robustness check results
7. **event_study_results.csv** - Year-by-year treatment effects
8. **cell_means.csv** - Simple 2x2 cell means

---

### Step 6: Report Generation

- Created LaTeX report: `replication_report_42.tex`
- Compiled to PDF: `replication_report_42.pdf` (22 pages)
- Three compilation passes to resolve cross-references

---

## Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| replication_report_42.tex | LaTeX source | Complete |
| replication_report_42.pdf | Final report (22 pages) | Complete |
| run_log_42.md | This log file | Complete |

---

## Summary

**Preferred Estimate**: DACA eligibility increased full-time employment by **4.75 percentage points** (SE = 0.011, 95% CI: [0.026, 0.069]) among Hispanic-Mexican, Mexican-born non-citizens aged 26-30 at policy implementation, relative to those aged 31-35.

**Interpretation**: The effect is statistically significant at the 1% level and represents approximately a 7.5% increase relative to the treatment group's baseline full-time employment rate of 63.1%.

**Key Robustness Findings**:
- Placebo test shows no pre-trend differences
- Results consistent across sexes
- Robust to bandwidth choice
- Similar effects for any employment outcome

---

## Session End: 2026-01-26
