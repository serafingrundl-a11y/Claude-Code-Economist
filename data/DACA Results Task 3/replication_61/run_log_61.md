# Run Log - DACA Replication Study (Replication 61)

## Session Start
Date: 2026-01-27

## Task Overview
Replicate the causal analysis of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

### Research Question
Among ethnically Hispanic-Mexican Mexican-born people living in the US, what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (35+ hours/week)?

### Identification Strategy
- Treatment group: ELIGIBLE=1, individuals ages 26-30 at time of policy (June 15, 2012)
- Control group: ELIGIBLE=0, individuals ages 31-35 at time of policy
- Method: Difference-in-differences (DiD)
- Pre-period: 2008-2011
- Post-period: 2013-2016 (2012 excluded due to ambiguity)

---

## Step 1: Data Exploration

### Data Files
- `prepared_data_labelled_version.csv` - Contains labeled categorical variables
- `prepared_data_numeric_version.csv` - Contains numeric coding
- `acs_data_dict.txt` - Data dictionary from IPUMS

### Sample Size
- Total observations: 17,382
- Treatment group (ages 26-30 in 2012): 11,382
- Control group (ages 31-35 in 2012): 6,000
- Pre-DACA period (2008-2011): 9,527
- Post-DACA period (2013-2016): 7,855

### Key Variables
| Variable | Description | Coding |
|----------|-------------|--------|
| FT | Full-time employment | 1=Yes (35+ hrs/wk), 0=No |
| ELIGIBLE | DACA eligibility | 1=Treatment (26-30), 0=Control (31-35) |
| AFTER | Post-DACA indicator | 1=2013-2016, 0=2008-2011 |
| PERWT | Person weight | Population weights for ACS |

---

## Step 2: Descriptive Statistics

### Full-Time Employment Rates (Weighted)

|                     | Pre-DACA | Post-DACA |
|---------------------|----------|-----------|
| Control (31-35)     | 68.86%   | 66.29%    |
| Treatment (26-30)   | 63.69%   | 68.61%    |

### Simple DiD Calculation
- Treatment change: 68.61% - 63.69% = +4.91 pp
- Control change: 66.29% - 68.86% = -2.57 pp
- **DiD estimate: 4.91 - (-2.57) = 7.48 pp**

---

## Step 3: Regression Analysis

### Model Specifications

1. **Model 1**: Basic DiD with robust (HC1) standard errors
2. **Model 2**: DiD with state fixed effects, clustered SEs at state level
3. **Model 3**: DiD with state + year fixed effects (PREFERRED)
4. **Model 4**: DiD with state + year FE + individual covariates
5. **Model 5**: Unweighted basic DiD

### Main Results

| Model | Coefficient | SE | 95% CI | p-value |
|-------|-------------|-----|--------|---------|
| (1) Basic DiD | 0.0748 | 0.0181 | [0.039, 0.110] | <0.001 |
| (2) State FE | 0.0737 | 0.0209 | [0.033, 0.115] | <0.001 |
| (3) State + Year FE | 0.0710 | 0.0202 | [0.032, 0.111] | <0.001 |
| (4) + Covariates | 0.0592 | 0.0211 | [0.018, 0.101] | 0.005 |
| (5) Unweighted | 0.0643 | 0.0153 | [0.034, 0.094] | <0.001 |

---

## Step 4: Preferred Estimate

**Specification**: Model 3 - State and Year Fixed Effects

- **DiD Estimate**: 0.0710 (7.10 percentage points)
- **Standard Error**: 0.0202 (clustered at state level)
- **95% Confidence Interval**: [0.0315, 0.1105]
- **p-value**: 0.0004
- **Sample Size**: 17,382

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 7.1 percentage points among Hispanic-Mexican, Mexican-born individuals. This effect is statistically significant at the 1% level.

---

## Step 5: Robustness Checks

### Placebo Test (Pre-Period Only)
- Comparing 2010-2011 vs 2008-2009
- Coefficient: 0.0178
- SE: 0.0241
- p-value: 0.461
- **Conclusion**: No significant pre-trend, supports parallel trends assumption

### Event Study Results
| Year | Coefficient | SE |
|------|-------------|-----|
| 2009 | 0.0182 | 0.0325 |
| 2010 | -0.0140 | 0.0323 |
| 2011 | 0.0681 | 0.0351 |
| 2013 | 0.0839 | 0.0344 |
| 2014 | 0.0681 | 0.0353 |
| 2015 | 0.0695 | 0.0349 |
| 2016 | 0.1422 | 0.0352 |

Pre-DACA coefficients not statistically significant; post-DACA coefficients consistently positive.

### Heterogeneous Effects by Gender
| Group | Coefficient | SE | p-value | N |
|-------|-------------|-----|---------|-----|
| Male | 0.0716 | 0.0199 | <0.001 | 9,075 |
| Female | 0.0527 | 0.0281 | 0.061 | 8,307 |

Effect stronger and statistically significant for men.

---

## Key Decisions Log

1. **Data**: Used `prepared_data_numeric_version.csv` for analysis
2. **Weighting**: Applied person weights (PERWT) for population-representative estimates
3. **Standard Errors**: Clustered at state level for models with fixed effects
4. **Model Selection**: Chose state + year fixed effects as preferred specification
   - Rationale: Controls for time-invariant state characteristics and common temporal shocks
5. **Outcome Definition**: FT=1 if UHRSWORK >= 35, includes non-labor force as 0
6. **No sample restrictions**: Used entire provided sample as instructed

---

## Output Files

1. `analysis.py` - Main analysis script (Python)
2. `results_summary.json` - Stored regression results
3. `figure1_parallel_trends.png` - Trends visualization
4. `figure2_event_study.png` - Event study results
5. `figure3_did_illustration.png` - DiD explanation figure
6. `replication_report_61.tex` - LaTeX source for report
7. `replication_report_61.pdf` - Final report (16 pages)
8. `run_log_61.md` - This log file

---

## Software Used
- Python 3.14
- pandas, numpy, statsmodels, matplotlib
- pdfLaTeX (MiKTeX)

---

## Session End
Date: 2026-01-27
Status: COMPLETE
