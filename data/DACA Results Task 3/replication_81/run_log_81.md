# Run Log - Replication 81

## Overview
This log documents all commands, decisions, and analyses performed for the DACA replication study examining the effect of DACA eligibility on full-time employment among Mexican-born Hispanic individuals in the United States.

## Research Question
What was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment (35+ hours/week) among eligible Mexican-born Hispanic individuals aged 26-30 in June 2012?

## Identification Strategy
- **Treatment Group**: DACA-eligible individuals aged 26-30 in June 2012 (ELIGIBLE=1)
- **Control Group**: Similar individuals aged 31-35 in June 2012 who would have been eligible but for age (ELIGIBLE=0)
- **Method**: Difference-in-Differences (DiD)
- **Pre-period**: 2008-2011 (AFTER=0)
- **Post-period**: 2013-2016 (AFTER=1)
- Note: 2012 data excluded since treatment timing within year is ambiguous

## Data Description
- Source: American Community Survey (ACS) via IPUMS USA
- File: prepared_data_labelled_version.csv
- Sample: 17,382 observations (Hispanic-Mexican, Mexican-born individuals)
- Key variables:
  - FT: Full-time employment (1=yes, 0=no)
  - ELIGIBLE: Treatment group indicator (1=ages 26-30 in June 2012, 0=ages 31-35)
  - AFTER: Post-treatment indicator (1=2013-2016, 0=2008-2011)
  - PERWT: Person weights for population representativeness

---

## Analysis Log

### Step 1: Data Loading and Initial Exploration
**Commands:**
```python
df = pd.read_csv('data/prepared_data_labelled_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
```

**Results:**
- Loaded prepared_data_labelled_version.csv
- Verified sample size: 17,382 observations
- Confirmed years: 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 (2012 excluded)
- Confirmed key variables present: FT, ELIGIBLE, AFTER, PERWT, YEAR, STATEFIP

### Step 2: Sample Distribution
**Observations by Group:**
| Group | Pre-DACA (2008-2011) | Post-DACA (2013-2016) |
|-------|---------------------|----------------------|
| Control (Ages 31-35) | 3,294 | 2,706 |
| Treatment (Ages 26-30) | 6,233 | 5,149 |

**Weighted Population:**
| Group | Pre-DACA | Post-DACA |
|-------|----------|-----------|
| Control | 449,366 | 370,666 |
| Treatment | 868,160 | 728,157 |

### Step 3: Descriptive Statistics
**Full-Time Employment Rates (Weighted):**
| Group | Pre-DACA | Post-DACA | Change |
|-------|----------|-----------|--------|
| Control | 0.689 | 0.663 | -0.026 |
| Treatment | 0.637 | 0.686 | +0.049 |

**Simple DiD Calculation:**
- Treatment change: +0.049
- Control change: -0.026
- DiD Estimate: 0.075 (7.5 percentage points)

### Step 4: Regression Analysis
**Model Specifications:**

| Model | Description | DiD Estimate | SE | p-value |
|-------|-------------|--------------|-----|---------|
| 1 | Basic OLS (unweighted) | 0.064 | 0.015 | <0.001 |
| 2 | Weighted OLS | 0.075 | 0.015 | <0.001 |
| 3 | Weighted + Clustered SE | 0.075 | 0.020 | <0.001 |
| 4 | + Demographic Controls | 0.062 | 0.021 | 0.004 |
| 5 | + State Fixed Effects | 0.061 | 0.022 | 0.005 |
| 6 | + Year Fixed Effects | 0.058 | 0.021 | 0.006 |

**Preferred Specification (Model 6):**
```
FT ~ ELIGIBLE + ELIGIBLE*AFTER + FEMALE + MARRIED + AGE +
     EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA +
     HAS_CHILDREN + C(STATEFIP) + C(YEAR)
```

### Step 5: Parallel Trends Analysis
**Pre-treatment differential trend test:**
- Coefficient: 0.017
- Standard Error: 0.010
- p-value: 0.082

Interpretation: Marginally significant at 10% level, not at 5% level. Some concern about parallel trends but magnitude is modest.

### Step 6: Heterogeneity Analysis
**By Sex:**
| Sex | DiD Estimate | SE |
|-----|--------------|-----|
| Male | 0.072 | 0.020 |
| Female | 0.053 | 0.029 |

**By Education:**
| Education | DiD Estimate | n |
|-----------|--------------|---|
| High School | 0.061 | 12,444 |
| Some College | 0.067 | 2,877 |
| Two-Year Degree | 0.182 | 991 |
| BA+ | 0.162 | 1,058 |

### Step 7: Figure Generation
Created 6 figures:
1. figure1_parallel_trends.png - Time series of FT rates by group
2. figure2_did_visualization.png - Visual DiD calculation
3. figure3_sample_distribution.png - Sample sizes by year
4. figure4_heterogeneity_sex.png - Trends by sex
5. figure5_coefficient_plot.png - Estimates across specifications
6. figure6_education_distribution.png - Education by group

### Step 8: Report Compilation
```bash
pdflatex -interaction=nonstopmode replication_report_81.tex
pdflatex -interaction=nonstopmode replication_report_81.tex
pdflatex -interaction=nonstopmode replication_report_81.tex
```
Output: replication_report_81.pdf (21 pages)

---

## Key Decisions Made

1. **Weighting**: Used PERWT (person weights) to make estimates representative of the population

2. **Standard Errors**: Clustered at state level (STATEFIP) to account for within-state correlation and potential state-level policy effects

3. **Controls**: Included demographic covariates (sex, age, marital status, education, children) to improve precision and address compositional differences

4. **Fixed Effects**: Included both state and year fixed effects in preferred specification to control for time-invariant state factors and aggregate time trends

5. **Sample**: Used entire provided sample without further restrictions per instructions. Retained non-labor force participants coded as FT=0.

6. **Model Selection**: Chose Model 6 (full controls + state FE + year FE) as preferred specification because it provides most comprehensive control for confounders

---

## Final Results

### Preferred Estimate
- **Effect Size**: 0.058 (5.8 percentage points)
- **Standard Error**: 0.021 (clustered at state level)
- **95% Confidence Interval**: [0.017, 0.100]
- **Sample Size**: 17,382
- **p-value**: 0.006

### Interpretation
DACA eligibility increased the probability of full-time employment by approximately 5.8 percentage points among Mexican-born Hispanic individuals aged 26-30 in June 2012, relative to similar individuals aged 31-35 who were ineligible due to the age cutoff. This effect is statistically significant at the 1% level.

---

## Files Generated

1. **analysis.py** - Main analysis script
2. **create_figures.py** - Figure generation script
3. **regression_results.csv** - Summary of regression estimates
4. **yearly_means.csv** - Year-by-year employment rates
5. **group_summary.csv** - Sample summary statistics
6. **figure1_parallel_trends.png** through **figure6_education_distribution.png** - Figures
7. **replication_report_81.tex** - LaTeX source
8. **replication_report_81.pdf** - Final report (21 pages)
9. **run_log_81.md** - This log file

---

## Software Used
- Python 3.x with pandas, numpy, statsmodels, matplotlib
- pdflatex (MiKTeX)
