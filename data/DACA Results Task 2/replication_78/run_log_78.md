# Replication Run Log - Participant 78

## Project Overview
**Research Question**: Impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican Mexican-born individuals in the United States.

**Method**: Difference-in-Differences (DiD) analysis comparing treatment group (ages 26-30 at DACA implementation) to control group (ages 31-35 at DACA implementation).

---

## Session Log

### Step 1: Data Exploration
- **Timestamp**: Session start
- **Action**: Read replication instructions from `replication_instructions.docx`
- **Key details extracted**:
  - DACA implemented June 15, 2012
  - Treatment: Ages 26-30 at implementation (birth years 1982-1986)
  - Control: Ages 31-35 at implementation (birth years 1977-1981)
  - Outcome: Full-time employment (35+ hours/week)
  - Pre-period: 2006-2011
  - Post-period: 2013-2016 (excluding 2012 due to timing ambiguity)

### Step 2: Data Dictionary Review
- **Action**: Examined `data/acs_data_dict.txt`
- **Key variables identified**:
  - YEAR: Survey year
  - BIRTHYR: Birth year
  - HISPAN: Hispanic origin (1 = Mexican)
  - BPL: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - UHRSWORK: Usual hours worked per week
  - SEX, MARST, EDUC: Demographic covariates
  - PERWT: Person weight
  - STATEFIP: State FIPS code

### Step 3: Sample Construction
- **Decision**: Apply sequential restrictions to identify DACA-eligible population
- **Sample flow**:
  | Restriction | N |
  |-------------|---|
  | Raw data | 33,851,424 |
  | Years 2006-2011, 2013-2016 | 30,738,394 |
  | Hispanic-Mexican | 2,663,503 |
  | Born in Mexico | 898,879 |
  | Non-citizen | 636,722 |
  | Valid immigration year | 636,722 |
  | Arrived before age 16 | 186,357 |
  | Continuous residence since 2007 | 177,294 |
  | Birth years 1977-1986 | 44,725 |

### Step 4: Variable Definitions
- **Treatment**: `treated = 1 if BIRTHYR in [1982, 1986]`
- **Post-period**: `post = 1 if YEAR >= 2013`
- **Outcome**: `fulltime = 1 if UHRSWORK >= 35`
- **Covariates created**:
  - female: SEX == 2
  - married: MARST in [1, 2]
  - educ_hs: EDUC >= 6
  - age, age_sq: Current age and quadratic term

### Step 5: Analysis Execution
- **Script**: `analysis.py`
- **Models estimated**:
  1. Basic DiD (no covariates)
  2. DiD with year fixed effects
  3. DiD with covariates (preferred specification)
  4. DiD with state fixed effects
  5. Weighted regression (using PERWT)
  6. Subgroup analysis (by gender)
  7. Robustness with narrow bandwidth
  8. Event study specification

### Step 6: Results Summary

#### Main Results (Model 3 - Preferred)
| Statistic | Value |
|-----------|-------|
| DiD Coefficient | 0.0154 |
| Standard Error | 0.0132 |
| 95% CI | [-0.0105, 0.0413] |
| p-value | 0.2448 |
| Sample Size | 44,725 |

#### Robustness Checks
| Specification | Estimate | SE |
|--------------|----------|-----|
| Basic DiD | 0.0551*** | 0.0098 |
| Year FE | 0.0554*** | 0.0098 |
| With Covariates | 0.0154 | 0.0132 |
| State FE | 0.0138 | 0.0132 |
| Weighted | 0.0183 | 0.0157 |
| Males only | 0.0598*** | 0.0112 |
| Females only | 0.0372** | 0.0150 |
| Narrow bandwidth | 0.0480*** | 0.0128 |

### Step 7: Figures Generated
- `figure1_employment_trends.pdf` - Employment rates over time by group
- `figure2_event_study.pdf` - Event study coefficients
- `figure3_sample_sizes.pdf` - Sample sizes by year
- `figure4_did_visual.pdf` - DiD visualization

### Step 8: Report Compilation
- **LaTeX file**: `replication_report_78.tex`
- **PDF output**: `replication_report_78.pdf` (21 pages)
- Compiled 4 passes for cross-references

---

## Key Decisions and Justifications

### 1. Exclusion of 2012
**Decision**: Exclude 2012 from analysis
**Rationale**: DACA was implemented on June 15, 2012. Since ACS does not record interview month, observations from 2012 cannot be classified as pre- or post-treatment.

### 2. Non-Citizen as Proxy for Undocumented
**Decision**: Use CITIZEN = 3 (Not a citizen) as proxy for undocumented status
**Rationale**: Per instructions, assume non-citizens without immigration papers are undocumented. This is an imperfect proxy as it may include some documented non-citizens.

### 3. Continuous Residence Criterion
**Decision**: Require YRIMMIG <= 2007
**Rationale**: DACA required continuous presence since June 15, 2007. This restriction ensures individuals had been in US for at least 5 years by DACA implementation.

### 4. Arrived Before Age 16
**Decision**: Require (YRIMMIG - BIRTHYR) < 16
**Rationale**: DACA eligibility required arrival before 16th birthday. This is calculated from year of immigration minus birth year.

### 5. Preferred Specification Choice
**Decision**: Model 3 (year FE + covariates) as preferred specification
**Rationale**:
- Controls for observable demographic differences between groups
- Accounts for common year shocks
- Does not include state FE which adds minimal information for the cost of complexity
- Provides more conservative estimate than basic DiD

### 6. Event Study Reference Year
**Decision**: Use 2011 as reference year
**Rationale**: Last pre-treatment year provides cleanest comparison point for examining pre-trends and post-treatment dynamics.

---

## Potential Concerns

1. **Pre-trends**: Event study shows some evidence of differential trends in 2006 (coefficient = -0.035, p < 0.10). This warrants caution in causal interpretation.

2. **Age confounding**: Treatment and control groups differ in age by design. While covariates help adjust, perfect separation is not possible.

3. **Citizenship proxy**: Cannot directly identify undocumented immigrants; non-citizen status is imperfect proxy.

4. **No panel data**: ACS is repeated cross-section, so we observe different individuals each year, not the same people over time.

---

## Output Files

### Required Deliverables
- [x] `replication_report_78.tex` - LaTeX source file
- [x] `replication_report_78.pdf` - Compiled report (21 pages)
- [x] `run_log_78.md` - This log file

### Supporting Files
- `analysis.py` - Main analysis script
- `generate_figures.py` - Figure generation script
- `output_sample_by_year.csv` - Sample sizes by year
- `output_ft_rates_by_year.csv` - Full-time employment rates by year
- `output_event_study.csv` - Event study coefficients
- `output_summary_pre.csv` - Pre-period summary statistics
- `output_summary_post.csv` - Post-period summary statistics
- `output_results_summary.txt` - Text summary of main results
- `figure1_employment_trends.pdf/png` - Employment trends figure
- `figure2_event_study.pdf/png` - Event study figure
- `figure3_sample_sizes.pdf/png` - Sample sizes figure
- `figure4_did_visual.pdf/png` - DiD visualization figure

---

## Final Results for Qualtrics Survey

**Preferred Estimate**: Model 3 (DiD with Year FE and Covariates)
- **Effect Size**: 0.0154 (1.54 percentage points)
- **Standard Error**: 0.0132
- **95% Confidence Interval**: [-0.0105, 0.0413]
- **p-value**: 0.2448
- **Sample Size**: 44,725 observations
- **Treatment Group**: 26,591 (ages 26-30 at DACA implementation)
- **Control Group**: 18,134 (ages 31-35 at DACA implementation)

**Interpretation**: DACA eligibility is associated with a 1.54 percentage point increase in the probability of full-time employment, but this effect is not statistically significant at conventional levels (p = 0.24). The 95% confidence interval includes zero and ranges from -1.1 to 4.1 percentage points.

---

*Log completed successfully.*
