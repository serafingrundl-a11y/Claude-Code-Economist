# Run Log for DACA Replication Study (Participant 91)

## Date: 2026-01-26

## Overview
Independent replication of the study examining the causal impact of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals in the United States.

---

## Task Summary
**Research Question:** What was the causal impact of DACA eligibility on full-time employment (working 35+ hours/week) among ethnically Hispanic-Mexican, Mexican-born people in the US?

**Design:** Difference-in-Differences
- Treatment group: Ages 26-30 as of June 15, 2012 (DACA eligible)
- Control group: Ages 31-35 as of June 15, 2012 (would have been eligible but for age)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (excluding 2012 due to policy implementation timing)

---

## Step 1: Read and Understand Instructions
**Timestamp:** 2026-01-26 07:00

- Read replication_instructions.docx
- Key eligibility criteria for DACA:
  1. Arrived unlawfully in the US before 16th birthday
  2. Had not yet had 31st birthday as of June 15, 2012
  3. Lived continuously in US since June 15, 2007
  4. Present in US on June 15, 2012 without lawful status
- Outcome: Full-time employment (35+ hours/week)
- Treatment: DACA eligibility (ages 26-30 in 2012)
- Control: Ages 31-35 in 2012 (ineligible due to age only)

## Step 2: Examine Data Files
**Timestamp:** 2026-01-26 07:05

- data.csv: 33,851,424 observations (ACS 2006-2016)
- Data dictionary reviewed: acs_data_dict.txt
- Variables identified for analysis:
  - YEAR: Survey year
  - BIRTHYR: Birth year
  - HISPAN/HISPAND: Hispanic origin (1 = Mexican)
  - BPL/BPLD: Birthplace (200 = Mexico)
  - CITIZEN: Citizenship status (3 = Not a citizen)
  - YRIMMIG: Year of immigration
  - UHRSWORK: Usual hours worked per week
  - AGE: Age at time of survey
  - PERWT: Person weight for statistical analysis
  - SEX, MARST, EDUC: Covariates

## Step 3: Define Sample Selection Criteria
**Timestamp:** 2026-01-26 07:10

**Decision:** Sample selection criteria based on DACA eligibility requirements

1. Hispanic-Mexican ethnicity: HISPAN == 1 (Mexican)
2. Born in Mexico: BPL == 200
3. Non-citizen: CITIZEN in {3, 4} (Not a citizen or has first papers)
4. Arrived before age 16: (YRIMMIG - BIRTHYR) < 16
5. Continuous residence since 2007: YRIMMIG <= 2007
6. Age groups: 26-30 (treatment) or 31-35 (control) in 2012
7. Exclude year 2012 (implementation year)

**Rationale:** These criteria proxy for DACA eligibility while acknowledging the ACS cannot directly identify undocumented status.

## Step 4: Create Analysis Script
**Timestamp:** 2026-01-26 07:15

Created Python script: `analysis_daca_91.py`

Key features:
- Memory-optimized data loading (dtype specifications)
- Sample selection pipeline with counts at each step
- Multiple regression specifications
- Event study analysis for parallel trends
- Robustness checks (placebo, subgroup)

## Step 5: Execute Analysis
**Timestamp:** 2026-01-26 07:20

### Sample Selection Results:
```
Initial observations: 33,851,424
After Hispanic-Mexican restriction (HISPAN==1): 2,945,521
After Mexico birthplace restriction (BPL==200): 991,261
After non-citizen restriction (CITIZEN in [3,4]): 701,347
After arrived before age 16 restriction: 205,327
After continuous presence since 2007 (YRIMMIG<=2007): 195,023
After age group restriction (26-30 or 31-35 in 2012): 49,019
After excluding 2012: 44,725

Final analysis sample: 44,725
- Treatment group (ages 26-30 in 2012): 26,591
- Control group (ages 31-35 in 2012): 18,134
```

### Descriptive Statistics:
```
                        Control (Pre)  Control (Post)  Treatment (Pre)  Treatment (Post)
Full-time rate (unwt)      0.6431         0.6108          0.6111           0.6339
Full-time rate (wt)        0.6705         0.6412          0.6253           0.6580
N                          11,916         6,218           17,410           9,181
```

### Simple DiD Calculation:
```
DiD = (0.6339 - 0.6111) - (0.6108 - 0.6431) = 0.0551
```

### Main Regression Results:

| Model | Coefficient | SE | 95% CI | p-value | N | R² |
|-------|-------------|-----|--------|---------|---|-----|
| (1) Basic DiD (OLS) | 0.055 | 0.010 | [0.036, 0.074] | <0.001 | 44,725 | 0.001 |
| (2) Basic DiD (WLS) | 0.062 | 0.012 | [0.039, 0.085] | <0.001 | 44,725 | 0.002 |
| (3) Year FE | 0.061 | 0.012 | [0.038, 0.084] | <0.001 | 44,725 | 0.005 |
| (4) Year FE + Covariates* | 0.048 | 0.011 | [0.027, 0.069] | <0.001 | 44,725 | 0.156 |
| (5) Year + State FE | 0.060 | 0.012 | [0.037, 0.082] | <0.001 | 44,725 | 0.010 |
| (6) Full Specification | 0.047 | 0.011 | [0.027, 0.068] | <0.001 | 44,725 | 0.159 |

*Preferred specification

### Event Study Results (Reference: 2011):

| Year | Coefficient | SE | 95% CI | p-value |
|------|-------------|-----|--------|---------|
| 2006 | -0.005 | 0.024 | [-0.053, 0.042] | 0.827 |
| 2007 | -0.013 | 0.024 | [-0.060, 0.034] | 0.580 |
| 2008 | 0.019 | 0.025 | [-0.030, 0.067] | 0.452 |
| 2009 | 0.017 | 0.025 | [-0.032, 0.066] | 0.503 |
| 2010 | 0.019 | 0.025 | [-0.030, 0.068] | 0.450 |
| 2011 | (Reference) | --- | --- | --- |
| 2013 | 0.060 | 0.026 | [0.008, 0.111] | 0.023 |
| 2014 | 0.070 | 0.027 | [0.017, 0.122] | 0.009 |
| 2015 | 0.043 | 0.027 | [-0.009, 0.095] | 0.108 |
| 2016 | 0.095 | 0.027 | [0.043, 0.148] | <0.001 |

**Key Finding:** No evidence of differential pre-trends; effect emerges post-DACA.

### Robustness Checks:

**Placebo Test (Pre-DACA: 2006-2008 vs 2009-2011):**
- Coefficient: 0.012 (SE: 0.013, p = 0.358)
- Interpretation: No significant placebo effect, supporting parallel trends assumption

**Subgroup Analysis:**
| Subgroup | Coefficient | SE | N |
|----------|-------------|-----|---|
| Men | 0.061*** | 0.012 | 25,058 |
| Women | 0.030 | 0.018 | 19,667 |
| Married | 0.065*** | 0.016 | 21,047 |
| Not Married | 0.069*** | 0.017 | 23,678 |

## Step 6: Generate Report
**Timestamp:** 2026-01-26 07:30

- Created LaTeX report: replication_report_91.tex
- Compiled to PDF: replication_report_91.pdf (19 pages)
- Report includes:
  - Abstract
  - Introduction
  - Background on DACA
  - Data description
  - Empirical strategy
  - Results (main, event study, robustness)
  - Discussion
  - Conclusion
  - Appendices

---

## Key Analytical Decisions

1. **Sample Definition:** Restricted to Hispanic-Mexican individuals born in Mexico (HISPAN=1, BPL=200) rather than broader definition

2. **Citizenship Proxy:** Used CITIZEN in {3, 4} (non-citizen or first papers) as proxy for undocumented status

3. **Age Groups:** Used ages 26-30 (treatment) vs 31-35 (control) as specified

4. **Time Period:** Excluded 2012 to avoid mixing pre/post observations

5. **Outcome Definition:** Full-time employment = UHRSWORK >= 35

6. **Preferred Specification:** Model 4 (Year FE + Covariates) - balances controls without overfitting

7. **Standard Errors:** Heteroskedasticity-robust (HC1) standard errors

8. **Weighting:** Survey weights (PERWT) used in all main specifications

---

## Preferred Estimate

**Effect Size:** 0.048 (4.8 percentage points)
**Standard Error:** 0.011
**95% Confidence Interval:** [0.027, 0.069]
**P-value:** <0.001
**Sample Size:** 44,725

**Interpretation:** DACA eligibility is associated with a 4.8 percentage point increase in the probability of full-time employment, statistically significant at the 1% level.

---

## Output Files Created

1. `analysis_daca_91.py` - Main analysis script
2. `results_summary_91.csv` - Summary of all model results
3. `event_study_91.csv` - Event study coefficients
4. `descriptive_stats_91.csv` - Descriptive statistics by group/period
5. `subgroup_results_91.csv` - Subgroup analysis results
6. `replication_report_91.tex` - LaTeX report source
7. `replication_report_91.pdf` - Final PDF report (19 pages)
8. `run_log_91.md` - This log file

---

## Commands Executed

```bash
# Data loading and analysis
cd "C:\Users\seraf\DACA Results Task 2\replication_91"
python analysis_daca_91.py

# LaTeX compilation
pdflatex -interaction=nonstopmode replication_report_91.tex
pdflatex -interaction=nonstopmode replication_report_91.tex  # Second pass for refs
pdflatex -interaction=nonstopmode replication_report_91.tex  # Third pass
```

---

## Session Complete
**End Timestamp:** 2026-01-26 07:45

All deliverables created:
- [x] replication_report_91.tex
- [x] replication_report_91.pdf
- [x] run_log_91.md
