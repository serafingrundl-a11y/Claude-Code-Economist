# Replication Run Log - ID 82

## Date: 2026-01-25

## Research Question
Estimate the causal impact of DACA eligibility on full-time employment (35+ hours/week) among Hispanic-Mexican, Mexican-born individuals in the US, examining effects in 2013-2016.

---

## Data Overview

### Source
- American Community Survey (ACS) 2006-2016, provided via IPUMS USA
- Data file: data.csv (~33.8 million observations)
- Data dictionary: acs_data_dict.txt
- Optional: state_demo_policy.csv (state-level policy data)

### Key Variables Identified
- **YEAR**: Survey year (2006-2016)
- **HISPAN/HISPAND**: Hispanic origin (1=Mexican for HISPAN; 100-107 for HISPAND)
- **BPL/BPLD**: Birthplace (200=Mexico for BPL; 20000 for BPLD)
- **CITIZEN**: Citizenship status (3=Not a citizen)
- **YRIMMIG**: Year of immigration
- **BIRTHYR**: Birth year
- **BIRTHQTR**: Quarter of birth
- **AGE**: Age at survey
- **UHRSWORK**: Usual hours worked per week
- **EMPSTAT**: Employment status
- **PERWT**: Person weight

---

## DACA Eligibility Criteria (per instructions)

DACA was implemented on June 15, 2012. Eligibility requires:
1. Arrived in US before 16th birthday
2. Had not yet turned 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (arrived by 2007 or earlier)
4. Present in US on June 15, 2012 and not a citizen/legal resident

### Operational Definitions
- **Hispanic-Mexican & Mexican-born**: HISPAN == 1 AND BPL == 200
- **Not a citizen**: CITIZEN == 3 (assume non-citizens without naturalization are undocumented)
- **Arrived before age 16**: AGE at immigration < 16 (calculated from YRIMMIG and BIRTHYR)
- **Under 31 on June 15, 2012**: BIRTHYR > 1981, or BIRTHYR == 1981 and BIRTHQTR >= 3
- **In US since June 2007**: YRIMMIG <= 2007

---

## Identification Strategy

### Approach: Difference-in-Differences (DiD)

**Treatment Group**: Hispanic-Mexican, Mexican-born non-citizens who meet all DACA eligibility criteria based on age and arrival requirements.

**Control Group**: Hispanic-Mexican, Mexican-born non-citizens who do NOT meet DACA eligibility criteria due to age cutoffs (arrived after turning 16 or too old as of June 15, 2012).

**Time Periods**:
- Pre-treatment: 2006-2011 (and potentially 2012, though it's ambiguous since DACA was announced mid-2012)
- Post-treatment: 2013-2016 (focus period per instructions)

**Outcome**: Binary indicator for full-time employment (UHRSWORK >= 35)

---

## Analysis Decisions Log

### Decision 1: Sample Restriction
Restrict to:
- Hispanic-Mexican (HISPAN == 1)
- Born in Mexico (BPL == 200)
- Non-citizens (CITIZEN == 3)
- Working-age adults (16-45 years old at survey)
- Arrived in US by 2007 (to satisfy continuous residence requirement)

Rationale: Focus on the population potentially eligible for DACA while maintaining comparable control group.

### Decision 2: Treatment Definition
- DACA-eligible = arrived before age 16 AND born after June 15, 1981
- Control = non-DACA-eligible Mexican non-citizens (similar population but failed age criteria)

### Decision 3: Year 2012 Handling
Exclude 2012 from main analysis due to mid-year implementation; include in robustness check.

### Decision 4: Outcome Variable
Full-time employment = UHRSWORK >= 35 (among those with UHRSWORK > 0 or all individuals)

### Decision 5: Regression Specification
Main model:
$$Y_{ist} = \alpha + \beta (DACA\_Eligible_i \times Post_t) + \gamma DACA\_Eligible_i + \delta Post_t + X_i'\theta + \phi_s + \epsilon_{ist}$$

Where:
- $Y_{ist}$ = full-time employment indicator
- $DACA\_Eligible_i$ = treatment group indicator
- $Post_t$ = indicator for years 2013-2016
- $X_i$ = individual controls (age, sex, education, marital status)
- $\phi_s$ = state fixed effects

---

## Commands Executed

### Step 1: Data Exploration
```bash
head -5 data.csv  # View data structure
wc -l data.csv    # Count rows: 33,851,425
```

### Step 2: Analysis Script (Python)
```bash
python analysis.py
```

Created `analysis.py` to perform:
- Data loading and filtering
- DACA eligibility variable construction
- Descriptive statistics
- DiD regression (5 specifications)
- Robustness checks
- Event study analysis

### Step 3: Figure Generation
```bash
python create_figures.py
```

Created 4 figures:
- figure1_event_study.pdf: Event study plot showing year-specific treatment effects
- figure2_trends.pdf: Full-time employment trends by group
- figure3_robustness.pdf: Bar chart of robustness check estimates
- figure4_sample.pdf: Sample composition visualization

### Step 4: LaTeX Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_82.tex
pdflatex -interaction=nonstopmode replication_report_82.tex  # Second pass for refs
pdflatex -interaction=nonstopmode replication_report_82.tex  # Third pass
```

Output: 19-page PDF report

---

## Key Results

### Main Finding (Preferred Specification)
- **Effect**: 0.0195 (1.95 percentage points)
- **Robust SE**: 0.0044
- **95% CI**: [0.0109, 0.0280]
- **p-value**: < 0.001
- **Sample size**: 399,677 observations
- **Weighted population**: 55.8 million

### Sample Construction
| Restriction | N |
|-------------|---|
| Full ACS 2006-2016 | 33,851,424 |
| Hispanic-Mexican, born in Mexico | 991,261 |
| Non-citizens | 701,347 |
| Arrived by 2007 | 654,693 |
| Ages 16-45 | 438,688 |
| Excluding 2012 | 399,677 |

### Treatment/Control Groups
- DACA Eligible (Treatment): 83,611 obs
- Not DACA Eligible (Control): 316,066 obs

### Robustness Checks
| Specification | Coefficient | SE |
|---------------|-------------|-----|
| Preferred | 0.0195 | 0.0044 |
| Include 2012 | 0.0206 | 0.0033 |
| Employment outcome | 0.0334 | 0.0034 |
| Ages 18-30 | 0.0077 | 0.0055 |
| Males only | -0.0008 | 0.0042 |
| Females only | 0.0364 | 0.0056 |

### Key Finding: Gender Heterogeneity
The DACA effect is entirely driven by women (3.64 pp effect), with no significant effect for men.

---

## Files Produced

### Required Deliverables
- `replication_report_82.tex` - LaTeX source file
- `replication_report_82.pdf` - 19-page PDF report
- `run_log_82.md` - This run log

### Supporting Files
- `analysis.py` - Main analysis script
- `create_figures.py` - Figure generation script
- `results.json` - Results in JSON format
- `figure1_event_study.pdf/png` - Event study plot
- `figure2_trends.pdf/png` - Trends plot
- `figure3_robustness.pdf/png` - Robustness checks
- `figure4_sample.pdf/png` - Sample composition

---

## Notes
- ACS is repeated cross-section, not panel data
- Cannot distinguish documented vs undocumented among non-citizens; assume undocumented
- Person weights (PERWT) should be used for population-representative estimates
- Year 2012 excluded from main analysis due to mid-year DACA implementation
- State-level policy data (state_demo_policy.csv) was not used in the analysis

---

## Interpretation

DACA eligibility is associated with a statistically significant 1.95 percentage point increase in the probability of full-time employment among Hispanic-Mexican individuals born in Mexico. This effect is economically meaningful, representing approximately a 4.5% relative increase from the pre-DACA baseline of 43%.

The effect accumulated over time, with larger impacts in 2015-2016 compared to 2013, consistent with gradual DACA uptake. Event study analysis shows no clear pre-trends, supporting the parallel trends assumption, though some fluctuation exists in early pre-period years.

The striking gender heterogeneity (effect concentrated among women) suggests DACA may have differentially impacted labor market barriers for undocumented women.

