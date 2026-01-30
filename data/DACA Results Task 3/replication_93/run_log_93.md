# Replication Run Log - Session 93

## Overview
This log documents all commands and key decisions made during the independent replication of the DACA impact study on full-time employment.

## Research Question
Among ethnically Hispanic-Mexican, Mexican-born people living in the United States, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (defined as usually working 35+ hours per week)?

## Study Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at time of policy implementation (June 15, 2012)
- **Control Group**: Would-be eligible individuals aged 31-35 at time of policy (ineligible due to age cutoff)
- **Method**: Difference-in-Differences (DiD)
- **Pre-treatment period**: 2008-2011
- **Post-treatment period**: 2013-2016 (2012 excluded as treatment year)
- **Outcome**: Full-time employment (FT variable: 1 = works 35+ hours/week)

## Data
- Source: American Community Survey (ACS) via IPUMS
- Key variables provided:
  - ELIGIBLE: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
  - AFTER: 1 = post-DACA (2013-2016), 0 = pre-DACA (2008-2011)
  - FT: 1 = full-time employed, 0 = not full-time employed
  - PERWT: Person weight for survey estimation

---

## Session Log

### Step 1: Read Instructions
- Extracted text from replication_instructions.docx
- Key requirements:
  - Use provided ELIGIBLE variable (do not create own)
  - Use FT as outcome (includes those not in labor force as 0)
  - Do not drop individuals based on characteristics
  - Estimate effect for all eligible individuals aged 26-30
  - May use covariates and account for differing trends

### Step 2: Examine Data
- Data files present:
  - prepared_data_labelled_version.csv
  - prepared_data_numeric_version.csv
  - acs_data_dict.txt
- Confirmed key variables: YEAR, ELIGIBLE, AFTER, FT, PERWT, plus demographic covariates

### Step 3: Analysis Plan
1. Load and explore data
2. Generate descriptive statistics
3. Check parallel trends assumption
4. Estimate baseline DiD model
5. Estimate DiD with covariates
6. Robustness checks
7. Compile results

---

## Commands Executed

### Analysis Script (analysis.py)
```bash
python analysis.py
```

**Output Summary:**
- Dataset: 17,382 observations, 105 variables
- Years: 2008-2011 (pre), 2013-2016 (post)
- Treatment group (ELIGIBLE=1): 11,382 obs
- Control group (ELIGIBLE=0): 6,000 obs

**Key Results:**
- Simple DiD (weighted): 0.0748
- Preferred Model (Year + State FE, weighted): 0.0710
- Standard Error: 0.0152
- 95% CI: [0.0413, 0.1007]
- P-value: < 0.0001
- State-Clustered SE: 0.0202

**Parallel Trends Test:**
- Pre-treatment trend slope: 0.0172
- P-value: 0.3804 (no significant pre-trend)

### Figure Generation (analysis_figures.py)
```bash
python analysis_figures.py
```
Generated:
- figure1_parallel_trends.png
- figure2_event_study.png
- figure3_did_bars.png
- figure4_sample_size.png

---

## Key Analytical Decisions

1. **Weights**: Used PERWT (person weights) for all main specifications to ensure population-representative estimates.

2. **Fixed Effects**: Included both year and state fixed effects in preferred specification to control for time-varying aggregate shocks and time-invariant state characteristics.

3. **Standard Errors**: Reported both conventional and state-clustered standard errors. Clustering at state level accounts for within-state correlation in errors.

4. **Sample**: Used full provided sample without dropping any observations, as instructed.

5. **Event Study**: Conducted event study analysis with 2011 as reference year to examine dynamic treatment effects and pre-trend patterns.

6. **Heterogeneity**: Examined effects by sex and education level.

---

## Final Results Summary

**Preferred Estimate:**
- DiD Coefficient: 0.0710 (7.10 percentage points)
- Standard Error: 0.0152 (conventional) / 0.0202 (clustered)
- 95% CI: [0.0413, 0.1007] (conventional) / [0.0315, 0.1105] (clustered)
- Sample Size: 17,382
- Interpretation: DACA eligibility associated with ~7 percentage point increase in full-time employment probability

