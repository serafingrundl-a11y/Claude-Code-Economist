# Replication Run Log - Session 99

## Project Overview
Replication study examining the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control group**: Ages 31-35 at DACA implementation (otherwise eligible)
- **Method**: Difference-in-Differences (DiD)
- **Outcome**: Full-time employment (usually working 35+ hours per week)
- **Post-treatment period**: 2013-2016

---

## Session Log

### Step 1: Read Replication Instructions
**Time**: Session start
**Action**: Read `replication_instructions.docx`
**Key requirements identified**:
- Target population: Hispanic-Mexican, Mexican-born, non-citizen immigrants
- DACA implementation date: June 15, 2012
- Treatment: Ages 26-30 at implementation (born ~1982-1986)
- Control: Ages 31-35 at implementation (born ~1977-1981)
- Outcome: Full-time employment (UHRSWORK >= 35)
- Data: ACS 2006-2016 (pre: 2006-2011, post: 2013-2016, exclude 2012)

### Step 2: Explore Data Files
**Files located in `data/` folder**:
- `data.csv` (6.26 GB) - main ACS data
- `acs_data_dict.txt` - variable definitions
- `state_demo_policy.csv` - optional state-level data

**Key variables identified from data dictionary**:
- YEAR: Census year
- BIRTHYR: Birth year
- HISPAN/HISPAND: Hispanic origin (1 = Mexican)
- BPL/BPLD: Birthplace (200 = Mexico)
- CITIZEN: Citizenship status (3 = Not a citizen)
- YRIMMIG: Year of immigration
- UHRSWORK: Usual hours worked per week
- EMPSTAT: Employment status
- PERWT: Person weight for sampling

### Step 3: Define DACA Eligibility Criteria
**Decision**: Based on instructions, eligibility requires:
1. Hispanic-Mexican ethnicity (HISPAN == 1)
2. Born in Mexico (BPL == 200)
3. Not a citizen (CITIZEN == 3)
4. Arrived before age 16 (YRIMMIG - BIRTHYR < 16)
5. Present in US since June 15, 2007 (YRIMMIG <= 2007)

**Age groups at June 15, 2012**:
- Treatment: Born 1982-1986 (ages 26-30)
- Control: Born 1977-1981 (ages 31-35)

### Step 4: Create Analysis Script
**File**: `analysis.py`
**Implementation details**:
- Used Python with pandas, statsmodels
- Read data in chunks due to large file size
- Applied DACA eligibility filters
- Implemented DiD regression with state-clustered standard errors
- Created event study specification for pre-trends

### Step 5: Run Analysis
**Command**: `python analysis.py`
**Execution time**: ~10 minutes

---

## Analysis Results

### Sample Construction
| Step | Observations |
|------|-------------|
| Hispanic-Mexican, Mexican-born, non-citizens | 636,722 |
| Born 1977-1986 (target age groups) | 162,283 |
| Arrived before age 16, by 2007 | 44,725 |
| Final analytic sample | 44,725 |

### Sample Sizes by Group
| Group | Pre-DACA | Post-DACA | Total |
|-------|----------|-----------|-------|
| Control (ages 31-35) | 11,916 | 6,218 | 18,134 |
| Treatment (ages 26-30) | 17,410 | 9,181 | 26,591 |
| **Total** | 29,326 | 15,399 | 44,725 |

### Full-Time Employment Rates
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment | 0.611 | 0.634 | +0.023 |
| Control | 0.643 | 0.611 | -0.032 |
| **Simple DiD** | | | **0.055** |

### Main Regression Results
| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Basic DiD | 0.055 | 0.006 | <0.001 |
| With Demographics | 0.066 | 0.012 | <0.001 |
| With Demographics + Education | 0.066 | 0.012 | <0.001 |
| With Year FE | 0.016 | 0.011 | 0.140 |
| With State + Year FE | 0.014 | 0.011 | 0.209 |
| Weighted | 0.019 | 0.010 | 0.071 |

### Pre-Trends (Event Study)
| Year | Coefficient | SE | p-value |
|------|-------------|-----|---------|
| 2006 | -0.005 | 0.013 | 0.695 |
| 2007 | -0.004 | 0.014 | 0.761 |
| 2008 | 0.019 | 0.012 | 0.111 |
| 2009 | 0.006 | 0.015 | 0.679 |
| 2010 | -0.003 | 0.012 | 0.808 |
| **2011** | (reference) | - | - |
| 2013 | 0.022 | 0.015 | 0.135 |
| 2014 | 0.014 | 0.015 | 0.362 |
| 2015 | 0.016 | 0.018 | 0.368 |
| 2016 | 0.026 | 0.016 | 0.115 |

### Robustness Checks
| Specification | Coefficient | SE | p-value |
|---------------|-------------|-----|---------|
| Narrower age bandwidth | 0.064 | 0.012 | <0.001 |
| Males only | 0.073 | 0.030 | 0.014 |
| Females only | 0.048 | 0.017 | 0.004 |
| Placebo (citizens) | -0.016 | 0.014 | 0.261 |

---

## Key Decisions Made

1. **Year 2012 excluded**: Cannot distinguish pre/post DACA within 2012
2. **Full-time definition**: UHRSWORK >= 35 (per instructions)
3. **Non-citizen proxy**: CITIZEN == 3 used as proxy for undocumented status
4. **Immigration filters**: Arrived before age 16 AND by 2007 (proxy for DACA eligibility)
5. **Standard errors**: Clustered at state level for inference
6. **Preferred specification**: Model 5 with state and year fixed effects

---

## Preferred Estimate

| Metric | Value |
|--------|-------|
| **Effect Size** | 0.0141 |
| **Standard Error** | 0.0112 |
| **95% CI** | [-0.0079, 0.0361] |
| **p-value** | 0.209 |
| **Sample Size** | 44,725 |

---

## Output Files Generated

1. `analysis.py` - Main analysis script
2. `results_summary.csv` - Key results for reporting
3. `yearly_means.csv` - Full-time employment by year and group
4. `model_comparison.csv` - Comparison of all model specifications
5. `descriptive_stats.csv` - Descriptive statistics by group
6. `replication_report_99.tex` - LaTeX report
7. `replication_report_99.pdf` - Final PDF report (20 pages)

---

## Interpretation

The simple DiD estimate of 5.5 percentage points suggests DACA increased full-time employment. However, after including year and state fixed effects along with demographic controls, the estimate decreases to 1.4 percentage points and is no longer statistically significant (p = 0.21).

Pre-trends analysis shows no evidence of differential trends before DACA, supporting the parallel trends assumption. The placebo test using naturalized citizens (who should not be affected by DACA) yields a null result, further supporting the research design validity.

The findings suggest a modest positive effect of DACA on full-time employment that is sensitive to specification choices and imprecisely estimated.
