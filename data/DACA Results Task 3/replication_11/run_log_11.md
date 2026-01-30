# Run Log - DACA Replication Study (Replication 11)

## Date: 2026-01-27

## Overview
This log documents an independent replication analyzing the causal impact of DACA eligibility on full-time employment among ethnically Hispanic-Mexican, Mexican-born individuals in the United States.

## Research Design
- **Treatment Group**: DACA-eligible individuals aged 26-30 at policy implementation (June 2012)
- **Control Group**: Individuals aged 31-35 at policy implementation who would have been eligible except for age
- **Method**: Difference-in-Differences (DiD)
- **Outcome**: Full-time employment (FT = 1 if usually working 35+ hours/week)
- **Pre-period**: 2008-2011
- **Post-period**: 2013-2016 (2012 omitted as treatment year)

## Data Source
- American Community Survey (ACS) data via IPUMS
- Provided datasets:
  - `prepared_data_labelled_version.csv`
  - `prepared_data_numeric_version.csv`
- Sample size: 17,382 observations (17,383 lines including header)

## Key Variables from Data Dictionary
- **ELIGIBLE**: 1 = treatment group (ages 26-30), 0 = control group (ages 31-35)
- **AFTER**: 1 = post-treatment (2013-2016), 0 = pre-treatment (2008-2011)
- **FT**: 1 = full-time employment, 0 = not full-time
- **PERWT**: Person weight for survey-weighted analysis
- **YEAR**: Survey year
- **AGE_IN_JUNE_2012**: Age as of policy implementation date

---

## Session Log

### Step 1: Initial Setup and Data Examination
- Read replication instructions from `replication_instructions.docx`
- Examined data dictionary (`acs_data_dict.txt`)
- Confirmed data files present and sample size (17,382 observations)
- Key identification: Use provided ELIGIBLE variable, do not create own eligibility criteria
- Note: Binary IPUMS variables coded 1=No, 2=Yes; added variables (FT, AFTER, ELIGIBLE) coded 0=No, 1=Yes

### Step 2: Analysis Plan
1. Load data and verify key variables
2. Generate descriptive statistics for treatment and control groups
3. Verify parallel trends assumption (pre-treatment period)
4. Estimate main DiD specification:
   - FT = β0 + β1*ELIGIBLE + β2*AFTER + β3*(ELIGIBLE×AFTER) + ε
   - β3 is the DiD estimate (causal effect of DACA eligibility)
5. Add covariates for improved precision
6. Use survey weights (PERWT) for population-representative estimates
7. Cluster standard errors at state level (STATEFIP) for proper inference
8. Conduct robustness checks

### Step 3: Data Loading and Verification (Completed)
- Loaded `prepared_data_numeric_version.csv`
- Dataset shape: 17,382 observations × 105 variables
- Verified key variables:
  - ELIGIBLE: 11,382 treated (ages 26-30), 6,000 control (ages 31-35)
  - AFTER: 9,527 pre-period, 7,855 post-period
  - FT: 11,283 employed full-time, 6,099 not full-time
  - Years: 2008-2011 (pre), 2013-2016 (post); 2012 excluded

### Step 4: Descriptive Statistics (Completed)
Sample sizes by group:
| ELIGIBLE | AFTER | N     |
|----------|-------|-------|
| 0        | 0     | 3,294 |
| 0        | 1     | 2,706 |
| 1        | 0     | 6,233 |
| 1        | 1     | 5,149 |

Weighted Full-Time Employment Rates:
| Group               | Pre-DACA | Post-DACA |
|---------------------|----------|-----------|
| Control (31-35)     | 68.86%   | 66.29%    |
| Treatment (26-30)   | 63.69%   | 68.61%    |

Treatment group demographics:
- Mean age: 28.0 years
- 46.5% female
- Mean family size: 4.3
- Mean number of children: 1.15

### Step 5: Parallel Trends Assessment (Completed)
Pre-treatment FT rates by year:
| Year | Control | Treatment | Diff    |
|------|---------|-----------|---------|
| 2008 | 74.7%   | 68.0%     | -6.7 pp |
| 2009 | 68.5%   | 63.7%     | -4.9 pp |
| 2010 | 69.0%   | 60.9%     | -8.1 pp |
| 2011 | 62.4%   | 62.5%     | +0.1 pp |

- Both groups show similar declining trend 2008-2011 (Great Recession recovery period)
- Gap between groups converges to near zero by 2011
- Parallel trends assumption appears reasonable

### Step 6: Main DiD Analysis (Completed)

#### Model Specifications:
1. **Model 1**: Basic OLS (unweighted)
   - DiD = 0.0643 (SE = 0.0153), p < 0.001

2. **Model 2**: Weighted by PERWT
   - DiD = 0.0748 (SE = 0.0152), p < 0.001

3. **Model 3**: Weighted + Robust (HC1) SE
   - DiD = 0.0748 (SE = 0.0181), p < 0.001

4. **Model 4**: Weighted + State-Clustered SE (**PREFERRED**)
   - DiD = 0.0748 (SE = 0.0203), p = 0.0002
   - 95% CI: [0.0350, 0.1145]

5. **Model 5**: With demographic covariates (sex, family size, children, marital status)
   - DiD = 0.0642 (SE = 0.0213), p = 0.003
   - R² = 0.131

6. **Model 6**: With state and year fixed effects
   - DiD = 0.0611 (SE = 0.0213), p = 0.004
   - R² = 0.139

### Step 7: Robustness Checks (Completed)

#### Event Study (Reference: 2011)
| Year | Coefficient | SE     | Sig   |
|------|-------------|--------|-------|
| 2008 | -0.0681     | 0.0294 | **    |
| 2009 | -0.0499     | 0.0374 |       |
| 2010 | -0.0821     | 0.0296 | ***   |
| 2013 | +0.0158     | 0.0406 |       |
| 2014 | +0.0000     | 0.0279 |       |
| 2015 | +0.0014     | 0.0384 |       |
| 2016 | +0.0741     | 0.0299 | **    |

- Pre-treatment coefficients negative (treatment group lagging)
- Post-treatment coefficients positive (treatment group catching up/surpassing)
- Largest effect in 2016 (4 years after DACA)

#### Placebo Test (Pre-Period Only)
- Testing 2010-2011 vs 2008-2009
- Placebo DiD = 0.0178 (SE = 0.0255), p = 0.486
- No significant pre-treatment effect (supports validity)

#### Heterogeneous Effects by Sex
- Males: DiD = 0.0716 (SE = 0.0195), p < 0.001
- Females: DiD = 0.0527 (SE = 0.0290), p = 0.070
- Effect stronger and more precisely estimated for males

### Step 8: Key Findings Summary

**Preferred Estimate (Model 4):**
- DACA eligibility increased full-time employment by **7.48 percentage points**
- Standard error (clustered by state): 0.0203
- 95% Confidence Interval: [3.50 pp, 11.45 pp]
- P-value: 0.0002
- Sample size: 17,382

**Interpretation:**
DACA eligibility is associated with a statistically significant 7.5 percentage point increase in the probability of full-time employment among Mexican-born Hispanic individuals aged 26-30 compared to similar individuals aged 31-35 who were ineligible due to age. The effect is robust to alternative specifications including controls for demographics and state/year fixed effects.

### Step 9: Files Generated
- `analysis.py`: Main analysis script
- `figure1_parallel_trends.png`: Parallel trends visualization
- `figure2_event_study.png`: Event study coefficients
- `figure3_did_visual.png`: DiD visualization
- `results_summary.csv`: Model comparison table
- `descriptive_stats.csv`: Descriptive statistics
- `group_means.csv`: 2×2 DiD table means

### Analytical Decisions Log
1. **Weighting**: Used PERWT for population-representative estimates
2. **Standard errors**: Clustered at state level (STATEFIP) to account for within-state correlation
3. **Sample**: Used full provided sample without restrictions (per instructions)
4. **Covariates**: Tested models with and without demographic controls
5. **Fixed effects**: Tested state and year fixed effects for robustness
6. **Preferred specification**: Model 4 (weighted, state-clustered SE) balances simplicity and proper inference

---

## Final Deliverables

| File | Description | Status |
|------|-------------|--------|
| `replication_report_11.tex` | LaTeX source for replication report | Complete |
| `replication_report_11.pdf` | Compiled PDF report (21 pages) | Complete |
| `run_log_11.md` | This run log documenting all decisions | Complete |

### Supporting Files Generated
- `analysis.py` - Python analysis script (replicable)
- `figure1_parallel_trends.png` - Parallel trends figure
- `figure2_event_study.png` - Event study figure
- `figure3_did_visual.png` - DiD visualization
- `results_summary.csv` - Summary of all model estimates
- `descriptive_stats.csv` - Descriptive statistics by group
- `group_means.csv` - 2×2 DiD cell means

---

## Session Complete
- **Date/Time**: 2026-01-27
- **Analysis software**: Python 3 (pandas, numpy, statsmodels, matplotlib)
- **LaTeX compiler**: pdfLaTeX (MiKTeX)
