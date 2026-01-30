# Replication Run Log - ID 98

## Task Overview
Replicating analysis of the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican Mexican-born people in the United States.

---

## Session Start
**Date:** 2026-01-27

### Initial Setup
- Read replication instructions from `replication_instructions.docx`
- Examined data files in `data/` folder:
  - `prepared_data_labelled_version.csv` - Labelled data file (18.9 MB)
  - `prepared_data_numeric_version.csv` - Numeric data file (6.5 MB)
  - `acs_data_dict.txt` - Data dictionary

### Key Research Design Elements
- **Treatment Group:** DACA-eligible individuals aged 26-30 at time of policy (June 2012), ELIGIBLE=1
- **Control Group:** Individuals aged 31-35 at time of policy who would have been eligible if not for age, ELIGIBLE=0
- **Outcome:** Full-time employment (FT=1 if usually working 35+ hours/week)
- **Method:** Difference-in-Differences comparing pre-period (2008-2011) to post-period (2013-2016)
- **Key Variables:** FT (outcome), ELIGIBLE (treatment group), AFTER (post-policy indicator)

### Data Characteristics
- Sample size: 17,382 observations (excluding header)
- Years included: 2008-2011 (pre) and 2013-2016 (post); 2012 omitted
- Population: Ethnically Hispanic-Mexican Mexican-born individuals

---

## Analysis Steps

### Step 1: Exploratory Data Analysis
- Loaded data from `prepared_data_numeric_version.csv`
- Total observations: 17,382
- Years: 2008-2011 (pre-DACA), 2013-2016 (post-DACA); 2012 excluded
- ELIGIBLE variable: 0=Control (ages 31-35), 1=Treated (ages 26-30)
- AFTER variable: 0=Pre-DACA, 1=Post-DACA
- FT outcome variable: 0=Not full-time, 1=Full-time (35+ hours/week)

### Sample Composition
| Group | Pre-DACA (2008-11) | Post-DACA (2013-16) | Total |
|-------|-------------------|---------------------|-------|
| Control (31-35) | 3,294 | 2,706 | 6,000 |
| Treated (26-30) | 6,233 | 5,149 | 11,382 |
| Total | 9,527 | 7,855 | 17,382 |

### Step 2: Basic DiD Calculation
Raw FT Employment Rates:
- Control, Pre: 66.97%
- Control, Post: 64.49%
- Treated, Pre: 62.63%
- Treated, Post: 66.58%

Manual DiD Estimate:
- Control change: -2.48 pp
- Treated change: +3.94 pp
- DiD = 6.43 pp

### Step 3: Regression Analysis

Created analysis.py with the following models:
1. **Model 1**: Basic DiD (no controls) - DiD = 0.0643 (SE: 0.0153)***
2. **Model 2**: + Demographics (sex, marital status, children, age) - DiD = 0.0549 (SE: 0.0142)***
3. **Model 3**: + Education controls - DiD = 0.0526 (SE: 0.0142)***
4. **Model 4**: + State and Year FE (HC1 SE) - DiD = 0.0512 (SE: 0.0142)***
5. **Model 5**: + State-clustered SE (PREFERRED) - DiD = 0.0512 (SE: 0.0147)***

### Step 4: Robustness Checks
- By gender: Males (6.15 pp***), Females (4.52 pp*)
- By education: Effects strongest for Some College (10.75 pp***)
- Placebo test (fake treatment at 2010): 1.57 pp (not significant) - supports parallel trends

### Key Decisions
1. Used numeric version of data for analysis
2. Used ELIGIBLE and AFTER variables as provided (did not create own)
3. Kept all observations including those not in labor force
4. Used person-level data, no weighting (noted for robustness)
5. Clustered standard errors at state level for preferred specification
6. Included state and year fixed effects to control for unobserved heterogeneity

### Preferred Estimate
- Effect Size: 5.12 percentage points
- Standard Error: 1.47 pp (clustered at state level)
- 95% CI: [2.24, 8.00] pp
- p-value: 0.0005
- Sample Size: 17,379

---

## Deliverables Generated

### Files Created
1. **analysis.py** - Python script containing all analysis code
2. **analysis_results.json** - JSON file with key results for reference
3. **replication_report_98.tex** - LaTeX source for the replication report
4. **replication_report_98.pdf** - Compiled PDF report (23 pages)
5. **run_log_98.md** - This run log documenting all commands and decisions

### Report Contents
The 23-page report includes:
- Abstract
- Introduction
- Background on DACA
- Data description
- Methodology (DiD specification)
- Main results with regression tables
- Robustness checks (gender, education, placebo)
- Discussion and limitations
- Conclusion
- Appendices with full regression output and variable definitions

---

## Session End
**Date:** 2026-01-27

### Summary
Successfully completed independent replication of DACA eligibility effects on full-time employment. Found a statistically significant positive effect of approximately 5.12 percentage points (95% CI: 2.24-8.00 pp), indicating that DACA eligibility increased full-time employment among eligible Hispanic-Mexican Mexican-born individuals.

### Interpretation
DACA eligibility is associated with a 5.12 percentage point increase in the probability of full-time employment. This represents an 8.2% relative increase from the pre-DACA baseline of 62.63% full-time employment in the treatment group. The effect is robust across model specifications and statistically significant at the 0.1% level.
