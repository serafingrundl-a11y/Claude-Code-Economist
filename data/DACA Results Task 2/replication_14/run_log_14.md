# Replication Run Log - Session 14

## Overview
This log documents the analysis of the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of being employed full-time (working 35+ hours per week)?

## Study Design
- **Treatment Group**: Ages 26-30 at DACA implementation (June 15, 2012)
- **Control Group**: Ages 31-35 at DACA implementation (otherwise eligible)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2006-2011
- **Post-period**: 2013-2016
- **Note**: 2012 excluded due to policy implementation mid-year

---

## Session Log

### Step 1: Data Exploration

**Files examined**:
- `replication_instructions.docx`: Research task specifications
- `data/acs_data_dict.txt`: Variable definitions from IPUMS
- `data/data.csv`: Main ACS dataset (2006-2016), ~6.3GB

**Key variables identified**:
- `YEAR`: Census year (2006-2016)
- `HISPAN`/`HISPAND`: Hispanic origin (1=Mexican for HISPAN)
- `BPL`/`BPLD`: Birthplace (200=Mexico)
- `CITIZEN`: Citizenship status (3=Not a citizen)
- `YRIMMIG`: Year of immigration
- `BIRTHYR`: Birth year
- `BIRTHQTR`: Birth quarter (1-4)
- `AGE`: Age at survey
- `UHRSWORK`: Usual hours worked per week
- `EMPSTAT`: Employment status
- `PERWT`: Person weight for population estimates

### Step 2: DACA Eligibility Criteria Definition

**Decision**: Define DACA eligibility based on research instructions:

1. Hispanic-Mexican ethnicity: `HISPAN == 1`
2. Born in Mexico: `BPL == 200`
3. Not a citizen: `CITIZEN == 3`
4. Arrived before age 16: `YRIMMIG - BIRTHYR < 16`
5. Continuous presence since June 2007: `YRIMMIG <= 2006`

**Rationale for continuous presence**: Since ACS only provides year of immigration (not month), I used `YRIMMIG <= 2006` as a conservative criterion. This ensures individuals arrived before January 2007, satisfying the June 15, 2007 cutoff.

**Age calculation for June 15, 2012**:
- Birth quarters 1-2 (Jan-Jun): Age = 2012 - BIRTHYR
- Birth quarters 3-4 (Jul-Dec): Age = 2012 - BIRTHYR - 1

### Step 3: Data Loading and Preparation

**Command executed**:
```python
python analysis.py
```

**Memory optimization**: Due to the large dataset (6.3GB), implemented chunk-based loading with immediate filtering to keep only DACA-eligible observations. Used memory-efficient dtypes (int8, int16, float32).

**Sample construction**:
- Total eligible observations loaded: 191,315
- Treatment group (ages 26-30): 27,903
- Control group (ages 31-35): 19,515
- After excluding 2012: 43,238

### Step 4: Variable Creation

**Outcome variable**:
- `fulltime = 1` if `UHRSWORK >= 35`, else 0
- Following BLS definition of full-time work (35+ hours/week)

**Covariates**:
- `female`: SEX == 2
- `married`: MARST == 1 (married, spouse present)
- `educ_hs`: EDUC >= 7 (high school or higher)
- `has_children`: NCHILD > 0

**Treatment indicators**:
- `treatment_group`: 1 if ages 26-30 on June 15, 2012
- `post`: 1 if YEAR >= 2013
- `treat_x_post`: interaction term

### Step 5: Descriptive Statistics

**Sample by period**:
- Pre-DACA (2006-2011): 28,377 observations
- Post-DACA (2013-2016): 14,861 observations

**Full-time employment rates**:
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Treatment (26-30) | 0.631 | 0.660 | +0.029 |
| Control (31-35) | 0.673 | 0.643 | -0.030 |
| **DiD Estimate** | | | **0.059** |

### Step 6: Regression Analysis

**Models estimated**:

1. **Basic DiD** (no covariates)
   - Coefficient: 0.0590
   - SE: 0.0117
   - p-value: < 0.001

2. **With demographic covariates**
   - Coefficient: 0.0446
   - SE: 0.0107
   - p-value: < 0.001

3. **With year fixed effects** (PREFERRED)
   - Coefficient: 0.0424
   - SE: 0.0107
   - 95% CI: [0.021, 0.063]
   - p-value: < 0.001

4. **With state and year FE**
   - Coefficient: 0.0418
   - SE: 0.0107
   - p-value: < 0.001

**Decision on preferred specification**: Selected Model 3 (year FE + covariates) as preferred because:
- Controls for year-specific shocks affecting all groups
- Adjusts for demographic composition differences
- State FE (Model 4) provides nearly identical results, suggesting state-level trends not confounding

### Step 7: Robustness Checks

**Gender heterogeneity**:
- Men: DiD = 0.0462 (SE = 0.0125, p < 0.001)
- Women: DiD = 0.0466 (SE = 0.0185, p = 0.012)
- Similar effects for both genders

**Placebo test** (2009 as fake policy year):
- Placebo DiD = 0.0058 (SE = 0.0136, p = 0.668)
- Not statistically significant, supporting parallel trends

**Event study** (relative to 2011):
- Pre-period coefficients all close to zero and insignificant
- Post-period coefficients positive, largest in 2016 (0.068, p = 0.012)
- Supports parallel trends assumption

### Step 8: Figure Generation

**Command executed**:
```python
python create_figures.py
```

**Figures created**:
1. `figure1_trends.png/pdf`: Full-time employment trends by group
2. `figure2_event_study.png/pdf`: Event study coefficients
3. `figure3_coefficient_comparison.png/pdf`: Coefficient forest plot
4. `figure4_did_visual.png/pdf`: DiD conceptual diagram

### Step 9: Report Compilation

**Command executed**:
```bash
pdflatex -interaction=nonstopmode replication_report_14.tex
pdflatex -interaction=nonstopmode replication_report_14.tex  # Second pass for cross-refs
```

**Output**: 20-page PDF report

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Continuous presence cutoff | YRIMMIG <= 2006 | Conservative; ensures arrival before June 2007 |
| Age calculation | Use BIRTHQTR to refine | More accurate age on June 15, 2012 |
| Outcome definition | UHRSWORK >= 35 | Standard BLS full-time definition |
| 2012 exclusion | Yes | Policy implemented mid-year |
| Preferred model | Year FE + covariates | Balances bias reduction and precision |
| Standard errors | HC1 robust | Accounts for heteroskedasticity |
| Weights | PERWT | Population-representative estimates |

---

## Final Results

**Preferred Estimate**:
- **Effect**: 4.24 percentage point increase in full-time employment probability
- **Standard Error**: 1.07 percentage points
- **95% Confidence Interval**: [2.15, 6.33] percentage points
- **P-value**: 0.000072
- **Sample Size**: 43,238

**Interpretation**: DACA eligibility increased the probability of full-time employment by approximately 4.2 percentage points among eligible Hispanic-Mexican immigrants, representing a 6.7% increase relative to the baseline rate. The effect is statistically significant at all conventional levels and robust across specifications.

---

## Output Files

| File | Description |
|------|-------------|
| `analysis.py` | Main analysis script |
| `create_figures.py` | Figure generation script |
| `regression_results.csv` | Regression coefficients table |
| `descriptive_stats.csv` | Sample statistics |
| `year_group_means.csv` | Full-time rates by year and group |
| `event_study_results.csv` | Event study coefficients |
| `key_results.txt` | Key results for report |
| `figure1_trends.png` | Trends figure |
| `figure2_event_study.png` | Event study figure |
| `figure3_coefficient_comparison.png` | Coefficient comparison |
| `figure4_did_visual.png` | DiD visualization |
| `replication_report_14.tex` | LaTeX source |
| `replication_report_14.pdf` | Final report (20 pages) |
| `run_log_14.md` | This run log |

---

## Session Complete

All deliverables produced:
- [x] `replication_report_14.tex`
- [x] `replication_report_14.pdf`
- [x] `run_log_14.md`
