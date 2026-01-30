# Replication Run Log

## Session Start
Date: 2026-01-25

## Task Overview
Replicating the DACA study to estimate the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Key Decisions and Commands

### 1. Data Exploration

**Command to check data folder:**
```bash
ls -la data/
```

**Files identified:**
- `data/data.csv` - Main ACS data file (~34 million records, 6.2 GB)
- `data/acs_data_dict.txt` - Data dictionary with variable definitions
- `data/state_demo_policy.csv` - Optional state-level policy data (not used)
- `replication_instructions.docx` - Instructions document

**Key variables from data dictionary:**
- YEAR: Survey year (2006-2016)
- BIRTHYR: Year of birth
- BIRTHQTR: Quarter of birth (1-4)
- HISPAN: Hispanic origin (1 = Mexican)
- BPL: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- PERWT: Person weight for analysis
- EMPSTAT: Employment status

### 2. Sample Definition

**Treatment group (ages 26-30 as of June 15, 2012):**
- Calculated using BIRTHYR and BIRTHQTR
- If BIRTHQTR in {1, 2}: age = 2012 - BIRTHYR
- If BIRTHQTR in {3, 4}: age = 2012 - BIRTHYR - 1

**Control group (ages 31-35 as of June 15, 2012):**
- Same age calculation, restricted to ages 31-35

**DACA Eligibility Criteria Applied:**
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen and no immigration papers (CITIZEN == 3)
4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
5. Arrived by 2006 (YRIMMIG <= 2006, ensuring continuous residence since June 15, 2007)

**Pre-treatment period:** 2006-2011
**Post-treatment period:** 2013-2016
**Note:** 2012 excluded as DACA was implemented mid-year (June 15, 2012)

### 3. Outcome Variable
Full-time employment: UHRSWORK >= 35 hours per week

### 4. Methodology
Difference-in-Differences (DiD) estimation:
- Compare change in full-time employment from pre to post period
- Treatment group: Ages 26-30 on June 15, 2012 (DACA eligible by age)
- Control group: Ages 31-35 on June 15, 2012 (too old for DACA)

### 5. Analysis Commands

**Python analysis script: `analysis_02.py`**

```bash
python analysis_02.py
```

**Output:**
- Final sample size: 43,238 observations
- Treatment group: 25,470 (ages 26-30)
- Control group: 17,768 (ages 31-35)
- Weighted sample size: ~6 million person-years

### 6. Results Summary

**Main Finding (Model 4 - Full specification):**
- DiD Estimate: 0.0452 (4.52 percentage points)
- Standard Error: 0.0107
- 95% CI: [0.0242, 0.0661]
- P-value: < 0.0001

**Interpretation:** DACA eligibility increased full-time employment by approximately 4.5 percentage points among the targeted population.

**Raw DiD Table:**
|                    | Pre (2006-2011) | Post (2013-2016) | Change |
|--------------------|-----------------|------------------|--------|
| Control (31-35)    | 0.6731          | 0.6433           | -0.030 |
| Treatment (26-30)  | 0.6305          | 0.6597           | +0.029 |
| **DiD**            |                 |                  | 0.059  |

**Robustness Checks:**
- Males only: 0.0309 (SE: 0.0124)
- Females only: 0.0490 (SE: 0.0181)
- Labor force participation: 0.0359 (SE: 0.0092)
- Employment (any hours): 0.0431 (SE: 0.0101)

### 7. Model Specifications

**Model 1:** Basic DiD (no controls)
- TREATED_POST coefficient: 0.0590 (SE: 0.0117)

**Model 2:** DiD with demographics (Female, Married, HS Education)
- TREATED_POST coefficient: 0.0475 (SE: 0.0107)

**Model 3:** DiD with Year Fixed Effects
- TREATED_POST coefficient: 0.0459 (SE: 0.0107)

**Model 4 (Preferred):** DiD with Year and State Fixed Effects
- TREATED_POST coefficient: 0.0452 (SE: 0.0107)

### 8. Event Study Results

Year-by-year treatment effects (relative to 2011):
| Year | Coefficient | SE     | 95% CI              |
|------|-------------|--------|---------------------|
| 2006 | 0.0067      | 0.0227 | [-0.038, 0.051]     |
| 2007 | -0.0290     | 0.0223 | [-0.073, 0.015]     |
| 2008 | 0.0081      | 0.0228 | [-0.037, 0.053]     |
| 2009 | -0.0076     | 0.0235 | [-0.054, 0.038]     |
| 2010 | -0.0145     | 0.0233 | [-0.060, 0.031]     |
| 2013 | 0.0347      | 0.0242 | [-0.013, 0.082]     |
| 2014 | 0.0354      | 0.0246 | [-0.013, 0.084]     |
| 2015 | 0.0209      | 0.0248 | [-0.028, 0.070]     |
| 2016 | 0.0680***   | 0.0247 | [0.020, 0.116]      |

Pre-trends are not statistically significant, supporting the parallel trends assumption.

### 9. Output Files Generated

**Analysis outputs:**
- `summary_stats.csv` - Descriptive statistics
- `did_table.csv` - 2x2 DiD table
- `regression_results.csv` - Main regression results
- `event_study_results.csv` - Year-by-year effects
- `robustness_results.csv` - Robustness check results
- `yearly_summary.csv` - Summary by year

**Report files:**
- `replication_report_02.tex` - LaTeX source
- `replication_report_02.pdf` - Final PDF report (22 pages)

### 10. Key Analytical Decisions

1. **Year 2012 excluded:** DACA implemented June 15, 2012; cannot distinguish pre/post in that year
2. **Age calculation:** Used BIRTHQTR to approximate whether birthday occurred before June 15
3. **Undocumented proxy:** Used CITIZEN == 3 (non-citizen without papers)
4. **Continuous residence:** Required YRIMMIG <= 2006 to ensure presence since June 15, 2007
5. **Survey weights:** Used PERWT for all analyses
6. **Standard errors:** Heteroskedasticity-robust (HC1)
7. **Fixed effects:** Included year and state fixed effects in preferred specification

### 11. LaTeX Compilation

```bash
pdflatex -interaction=nonstopmode replication_report_02.tex
pdflatex -interaction=nonstopmode replication_report_02.tex
pdflatex -interaction=nonstopmode replication_report_02.tex
```

Three compilations for proper cross-references and table of contents.

## Session End
All required deliverables generated:
- replication_report_02.tex
- replication_report_02.pdf
- run_log_02.md

